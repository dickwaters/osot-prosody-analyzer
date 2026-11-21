#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSOT 五维播音声学深度剖析与三层解耦升级引擎 v2.0 (osot_prosody_analyzer_v2.py)
=============================================================================
v2.0 核心升级（响应学术评审与 P0/P1 改进建议）：
1. 【P0 统一协议】全面输出 osot-unified-v2.0 统一 JSON，严格分离：
   - tier1_acoustic      (客观物理测量，不可人工篡改)
   - tier2_machine_pred  (机器候选预测 + 置信度 + 白箱归因；明确不替代艺术意图)
   - tier3_human_gold    (人工专业审校终稿 + edit_history 修改轨迹)
2. 【P1 声调-语势联合解耦升级】不再宣称“CWT 单独彻底剥离四声”，改为：
   - 先做说话人半音归一化 + 同调型基线偏移补偿 (Tone-Offset Neutralization)，再执行多尺度 CWT；
   - 联合音长、能量与勒让德拟合残差输出「语势候选曲线 (yushi_candidate_curve)」、
     「语势置信度 (yushi_confidence)」与「声调混叠风险提示 (tone_confound_warning)」。
3. 【P1 候选置信度与人工审核边界】发声态 (虚/实/气/喉化) 与 B1 虚连均输出显式置信度与人工复核标记，
   明确区分“声学发声代理特征”与“播音艺术表达意图”。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt, get_window

# 允许同目录或上级 00_osot_schema 导入统一协议
_CUR_DIR = Path(__file__).resolve().parent
for _cand in (_CUR_DIR, _CUR_DIR.parent / "00_osot_schema"):
    if (_cand / "osot_schema.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from osot_schema import SCHEMA_VERSION, empty_unified_doc, migrate_to_unified  # noqa: E402

PAUSE_PUNCT_SET = set("，。！？、；：,.!?;:")
STRONG_PUNCT_SET = set("。！？!?；;")
WEAK_FUNCTION_CHARS = set("的了着过吗呢吧啊呀嘛地得之乎者也么啦哇哦嗯与和及或而")
CONTENT_BOOST_CHARS = set("不没无非最极更全所凡绝必竟却但然才就都只仅")


def load_wav_mono(wav_path: str) -> Tuple[np.ndarray, int]:
    try:
        import soundfile as sf  # type: ignore

        y, sr = sf.read(wav_path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y.astype(np.float64), int(sr)
    except Exception:
        pass
    with wave.open(wav_path, "rb") as wf:
        sr, ch, sw, nf = wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes()
        raw = wf.readframes(nf)
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"不支持的 WAV 位深: {sw * 8} bit")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, int(sr)


def save_wav_mono_16bit(wav_path: str, y: np.ndarray, sr: int) -> None:
    pcm = (np.clip(y, -0.99, 0.99) * 32767.0).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _autocorr_pitch_track(
    y: np.ndarray, sr: int, hop_sec: float = 0.005, win_sec: float = 0.040, fmin: float = 65.0, fmax: float = 550.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hop = max(1, int(round(sr * hop_sec)))
    win_len = max(64, int(round(sr * win_sec)))
    n_frames = max(1, 1 + max(0, len(y) - win_len) // hop)
    nyq = 0.5 * sr
    b, a = butter(3, [max(40.0 / nyq, 0.001), min(1200.0 / nyq, 0.95)], btype="band")
    y_bp = filtfilt(b, a, y) if len(y) > 30 else y.copy()

    window = get_window("hann", win_len)
    lag_min = max(2, int(sr / fmax))
    lag_max = min(win_len - 2, int(sr / fmin))
    times = np.zeros(n_frames, dtype=np.float64)
    f0 = np.zeros(n_frames, dtype=np.float64)
    db = np.zeros(n_frames, dtype=np.float64)
    hnr = np.zeros(n_frames, dtype=np.float64)
    energy_gate = max(1e-4, float(np.sqrt(np.mean(y ** 2) + 1e-12)) * 0.08)

    for idx in range(n_frames):
        s = idx * hop
        seg_r = y[s : s + win_len]
        seg_b = y_bp[s : s + win_len]
        if len(seg_r) < win_len:
            seg_r = np.pad(seg_r, (0, win_len - len(seg_r)))
            seg_b = np.pad(seg_b, (0, win_len - len(seg_b)))
        times[idx] = (s + 0.5 * win_len) / sr
        rms = float(np.sqrt(np.mean(seg_r ** 2) + 1e-12))
        db[idx] = 20.0 * math.log10(max(rms, 1e-6)) + 90.0
        if rms < energy_gate:
            continue
        xw = (seg_b - np.mean(seg_b)) * window
        spec = np.fft.rfft(xw, n=2 * win_len)
        acf = np.fft.irfft(spec * np.conj(spec))[:win_len]
        if acf[0] <= 1e-12:
            continue
        acf_n = acf / acf[0]
        sub = acf_n[lag_min : lag_max + 1]
        if len(sub) == 0:
            continue
        best_rel = int(np.argmax(sub))
        best_val = float(sub[best_rel])
        if best_val >= 0.35:
            best_lag = lag_min + best_rel
            if 0 < best_rel < len(sub) - 1:
                y0, y1, y2 = sub[best_rel - 1], sub[best_rel], sub[best_rel + 1]
                denom = y0 - 2.0 * y1 + y2
                delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
                best_lag_f = best_lag + float(np.clip(delta, -0.5, 0.5))
            else:
                best_lag_f = float(best_lag)
            f0[idx] = sr / max(best_lag_f, 1.0)
            rc = float(np.clip(best_val, 0.01, 0.995))
            hnr[idx] = 10.0 * math.log10(rc / (1.0 - rc))
    return times, f0, db, hnr


def extract_f0_hirst_twopass(
    y: np.ndarray, sr: int, hop_sec: float = 0.005
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    backend = "numpy-scipy-autocorr"
    try:
        import parselmouth  # type: ignore

        snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=float(sr))
        p1 = snd.to_pitch(time_step=hop_sec, pitch_floor=65.0, pitch_ceiling=550.0)
        v1 = p1.selected_array["frequency"]
        v1 = v1[v1 > 0]
        q25, q75 = np.percentile(v1, [25, 75]) if len(v1) >= 10 else (100.0, 280.0)
        floor2, ceil2 = max(55.0, float(0.75 * q25)), min(600.0, max(160.0, float(1.5 * q75)))
        p2 = snd.to_pitch(time_step=hop_sec, pitch_floor=floor2, pitch_ceiling=ceil2)
        inten = snd.to_intensity(minimum_pitch=floor2, time_step=hop_sec)
        harm = snd.to_harmonicity_cc(time_step=hop_sec, minimum_pitch=floor2)
        n_frames = max(1, int(len(y) / sr / hop_sec))
        times = np.arange(n_frames) * hop_sec + 0.5 * hop_sec
        f0 = np.array([0.0 if (v := p2.get_value_at_time(float(t))) is None or np.isnan(v) else float(v) for t in times])
        db = np.array([0.0 if (v := inten.get_value(float(t))) is None or np.isnan(v) else float(v) for t in times])
        hnr = np.array([0.0 if (v := harm.get_value(float(t))) is None or np.isnan(v) or v < -100 else float(v) for t in times])
        backend = "praat-parselmouth"
    except Exception:
        times, f0_1, db, hnr = _autocorr_pitch_track(y, sr, hop_sec=hop_sec)
        v1 = f0_1[f0_1 > 0]
        if len(v1) >= 10:
            q25, q75 = np.percentile(v1, [25, 75])
            floor2, ceil2 = max(55.0, float(0.75 * q25)), min(600.0, max(160.0, float(1.5 * q75)))
            times, f0, db, hnr = _autocorr_pitch_track(y, sr, hop_sec=hop_sec, fmin=floor2, fmax=ceil2)
        else:
            floor2, ceil2, f0 = 65.0, 550.0, f0_1

    v_idx = np.where(f0 > 0)[0]
    med_f0 = float(np.median(f0[v_idx])) if len(v_idx) >= 5 else 180.0
    for idx in v_idx:
        if f0[idx] > 1.85 * med_f0 and (f0[idx] * 0.5) >= floor2:
            f0[idx] *= 0.5
        elif f0[idx] < 0.55 * med_f0 and (f0[idx] * 2.0) <= ceil2:
            f0[idx] *= 2.0

    return times, f0, db, hnr, {
        "pitch_backend": backend,
        "hop_ms": int(round(hop_sec * 1000)),
        "speaker_f0_median_hz": round(med_f0, 2),
        "hirst_floor_hz": round(floor2, 1),
        "hirst_ceiling_hz": round(ceil2, 1),
    }


def interpolate_f0_semitone_gap_safe(
    f0_hz: np.ndarray, f0_median_hz: float, hop_sec: float = 0.005, max_interp_ms: float = 150.0
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    n = len(f0_hz)
    f0_st = np.full(n, np.nan, dtype=np.float64)
    voiced = f0_hz > 0
    if not np.any(voiced):
        return f0_st, []
    f0_st[voiced] = 12.0 * np.log2(f0_hz[voiced] / max(f0_median_hz, 1.0))
    max_gap = int(round((max_interp_ms / 1000.0) / hop_sec))
    v_idx = np.where(voiced)[0]
    segs: List[Tuple[int, int]] = []
    seg_s, prev = int(v_idx[0]), int(v_idx[0])
    for cur in v_idx[1:]:
        cur = int(cur)
        if cur - prev > max_gap:
            segs.append((seg_s, prev + 1))
            seg_s = cur
        prev = cur
    segs.append((seg_s, prev + 1))

    for s, e in segs:
        sub = f0_st[s:e]
        vl = np.where(~np.isnan(sub))[0]
        if len(vl) >= 2:
            f0_st[s:e] = PchipInterpolator(vl, sub[vl], extrapolate=False)(np.arange(len(sub)))
        elif len(vl) == 1:
            f0_st[s:e] = sub[vl[0]]
    return f0_st, segs


def infer_mandarin_tones(
    tokens: List[Dict[str, Any]], times: np.ndarray, f0_st: np.ndarray
) -> Tuple[List[int], str]:
    try:
        from pypinyin import Style, pinyin  # type: ignore

        tones = []
        for tok in tokens:
            ch = tok.get("char", "")
            py = pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True) if ch else []
            tones.append(int(py[0][0][-1]) if (py and py[0] and py[0][0] and py[0][0][-1].isdigit()) else 5)
        return tones, "pypinyin"
    except Exception:
        pass
    tones = []
    for tok in tokens:
        if tok.get("char", "") in WEAK_FUNCTION_CHARS:
            tones.append(5)
            continue
        m = (times >= tok["start_ms"] / 1000.0) & (times <= tok["end_ms"] / 1000.0) & (~np.isnan(f0_st))
        seg = f0_st[m]
        if len(seg) < 3:
            tones.append(1)
            continue
        slope = float(np.polyfit(np.linspace(-1, 1, len(seg)), seg, 1)[0])
        mean_v = float(np.mean(seg))
        tones.append(2 if slope > 1.2 else (4 if slope < -1.5 else (3 if mean_v < -1.2 else 1)))
    return tones, "acoustic-contour-fallback"


def tone_offset_neutralized_cwt(
    f0_st: np.ndarray,
    segments: List[Tuple[int, int]],
    tokens: List[Dict[str, Any]],
    tones: List[int],
    times: np.ndarray,
    hop_sec: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, float]]:
    """
    v2.0 升级：声调均值偏移预补偿 + 墨西哥帽 CWT 多尺度分解。
    先统计说话人各声调相对全篇中位数的固有高度偏置（如阴平偏高、上声偏低），
    在做宏观语势提取前温和扣除 50% 的固有调高偏置，减轻连串同调字对宏观语势拟合的拉偏。
    """
    tone_means: Dict[int, List[float]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    for tok, tn in zip(tokens, tones):
        m = (times >= tok["start_ms"] / 1000.0) & (times <= tok["end_ms"] / 1000.0) & (~np.isnan(f0_st))
        if np.any(m):
            tone_means[tn].append(float(np.mean(f0_st[m])))
    tone_bias = {tn: (round(float(np.median(vs)), 3) if len(vs) >= 2 else 0.0) for tn, vs in tone_means.items()}

    # 构造声调偏置预补偿后的序列用于宏观语势候选估计
    f0_for_macro = f0_st.copy()
    for tok, tn in zip(tokens, tones):
        m = (times >= tok["start_ms"] / 1000.0) & (times <= tok["end_ms"] / 1000.0) & (~np.isnan(f0_for_macro))
        if np.any(m):
            f0_for_macro[m] -= 0.50 * tone_bias.get(tn, 0.0)

    n = len(f0_st)
    f0_local = np.full(n, np.nan, dtype=np.float64)
    f0_macro = np.full(n, np.nan, dtype=np.float64)
    cwt_prom = np.zeros(n, dtype=np.float64)
    scales_ms = 80.0 * (2.0 ** (np.arange(0, 22) / 4.0))
    scales_ms = scales_ms[scales_ms <= 3300.0]

    for s, e in segments:
        seg_raw = f0_st[s:e]
        seg_comp = f0_for_macro[s:e]
        if len(seg_raw) < 6 or np.all(np.isnan(seg_raw)):
            continue
        raw_c = np.nan_to_num(seg_raw, nan=float(np.nanmedian(seg_raw)))
        comp_c = np.nan_to_num(seg_comp, nan=float(np.nanmedian(seg_comp)))
        sig = comp_c - float(np.mean(comp_c))
        L = len(sig)

        prom_acc = np.zeros(L, dtype=np.float64)
        macro_acc = np.zeros(L, dtype=np.float64)
        w_l, w_m = 0.0, 0.0
        for sc_ms in scales_ms:
            sigma = max(1.0, (sc_ms / 1000.0) / hop_sec / 4.0)
            hw = min(max(4, int(round(4.0 * sigma))), max(4, L))
            u = np.arange(-hw, hw + 1, dtype=np.float64) / sigma
            wav = (1.0 - u ** 2) * np.exp(-0.5 * u ** 2)
            wav -= np.mean(wav)
            wav /= np.sqrt(np.sum(wav ** 2)) + 1e-12
            pad = min(hw, L - 1)
            coeff = np.convolve(np.pad(sig, pad, mode="reflect"), wav, mode="same")[pad : pad + L]
            w = 1.0 / math.sqrt(sc_ms)
            if 150.0 <= sc_ms <= 600.0:
                prom_acc += np.maximum(coeff, 0.0) * w
                w_l += w
            elif sc_ms >= 800.0:
                macro_acc += coeff * w
                w_m += w

        # 高斯低通基底 + CWT 宏观曲率
        msig = max(2.0, 0.25 / hop_sec)
        hg = min(int(round(3.0 * msig)), max(2, L - 1))
        gk = np.exp(-0.5 * (np.arange(-hg, hg + 1) / msig) ** 2)
        gk /= np.sum(gk)
        pg = min(hg, L - 1)
        macro_sm = np.convolve(np.pad(comp_c, pg, mode="reflect"), gk, mode="same")[pg : pg + L]
        macro_fin = 0.85 * macro_sm + 0.15 * (float(np.mean(comp_c)) + (macro_acc / w_m if w_m > 0 else 0.0))

        f0_macro[s:e] = macro_fin
        f0_local[s:e] = raw_c - macro_fin
        if w_l > 0:
            cwt_prom[s:e] = prom_acc / w_l

    return f0_local, f0_macro, cwt_prom, tone_bias


def fit_legendre_yushi_with_confidence(
    f0_macro_seg: np.ndarray,
    phrase_tones: List[int],
    phrase_energies: np.ndarray,
    phrase_durs: np.ndarray,
) -> Dict[str, Any]:
    """
    v2.0：输出语势候选曲线 + 综合置信度 yushi_confidence + 声调混叠告警。
    不把勒让德拟合当成绝对定论，而是供专家审校与下游加权使用。
    """
    valid = ~np.isnan(f0_macro_seg)
    n = len(f0_macro_seg)
    voiced_ratio = float(np.mean(valid)) if n > 0 else 0.0

    # 检查声调混叠风险：若短语内连续去声(4)或上声(3)占比 >= 65%，提示声调可能拉偏斜率
    dominant_tone_ratio = max((phrase_tones.count(t) / max(1, len(phrase_tones)) for t in (1, 2, 3, 4)), default=0.0)
    tone_warning = (
        "短语内单一声调占比偏高（≥65%），宏观语势斜率可能受词汇声调连读影响，建议人工听审确认。"
        if dominant_tone_ratio >= 0.65 and len(phrase_tones) >= 3
        else ""
    )

    if np.sum(valid) < 5 or n < 5:
        return {
            "c0_level": 0.0, "c1_slope": 0.0, "c2_curvature": 0.0, "c3_inflection": 0.0,
            "rmse_st": 0.0, "pearson_r": 0.0,
            "candidate_yushi_label": "Level",
            "candidate_yushi_cn": "平直类（候选·有效帧不足）",
            "yushi_confidence": 0.15,
            "voiced_coverage": round(voiced_ratio, 3),
            "energy_slope_aligned": False,
            "tone_confound_warning": tone_warning or "有声帧不足",
            "yushi_candidate_curve": [],
        }

    t_all = np.linspace(-1.0, 1.0, n)
    t, y = t_all[valid], f0_macro_seg[valid]
    X = np.column_stack([np.ones_like(t), t, 0.5 * (3 * t ** 2 - 1), 0.5 * (5 * t ** 3 - 3 * t)])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coeffs
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    r_val = float(np.corrcoef(y, y_hat)[0, 1]) if (np.std(y) > 1e-6 and np.std(y_hat) > 1e-6) else 0.0
    c0, c1, c2, c3 = [float(c) for c in coeffs]

    # 联合能量斜率一致性校验（上山通常伴随能量抬升，下山伴随能量回落）
    eng_slope = float(np.polyfit(np.linspace(-1, 1, len(phrase_energies)), phrase_energies, 1)[0]) if len(phrase_energies) >= 2 else 0.0
    eng_aligned = bool((c1 > 0.4 and eng_slope > 0) or (c1 < -0.4 and eng_slope < 0) or abs(c1) <= 0.4)

    abs_c1, abs_c2, abs_c3 = abs(c1), abs(c2), abs(c3)
    if max(abs_c1, abs_c2, abs_c3) < 0.45:
        lbl, cn = "Level", "平直类（候选）"
    elif abs_c3 >= 1.15 and abs_c3 > max(abs_c1, abs_c2):
        lbl, cn = "Complex", "复合转折类（候选）"
    elif abs_c2 >= 0.75 and abs_c2 >= 0.85 * abs_c1:
        lbl, cn = ("Arch-Peak", "波峰类（候选·低→高→低）") if c2 < 0 else ("Arch-Valley", "波谷类（候选·高→低→高）")
    elif c1 >= 0.50:
        lbl, cn = "Rise", "上山类/起潮类（候选）"
    elif c1 <= -0.50:
        lbl, cn = "Fall", "下山类/落潮类（候选·教材定义缺口保留）"
    else:
        lbl, cn = "Level", "平直类（候选）"

    # 综合语势置信度 [0, 1]：融合拟合相关性、有声帧覆盖率、残差惩罚、能量同向加分与声调混叠惩罚
    conf = 0.45 * max(0.0, r_val) + 0.25 * voiced_ratio + 0.15 * max(0.0, 1.0 - rmse / 3.0) + (0.15 if eng_aligned else 0.05)
    if tone_warning:
        conf -= 0.15
    conf = round(float(np.clip(conf, 0.1, 0.98)), 3)

    tg = np.linspace(-1.0, 1.0, 25)
    yg = c0 + c1 * tg + c2 * 0.5 * (3 * tg ** 2 - 1) + c3 * 0.5 * (5 * tg ** 3 - 3 * tg)
    return {
        "c0_level": round(c0, 3), "c1_slope": round(c1, 3), "c2_curvature": round(c2, 3), "c3_inflection": round(c3, 3),
        "rmse_st": round(rmse, 3), "pearson_r": round(r_val, 3),
        "candidate_yushi_label": lbl,
        "candidate_yushi_cn": cn,
        "yushi_confidence": conf,
        "voiced_coverage": round(voiced_ratio, 3),
        "energy_slope_aligned": eng_aligned,
        "tone_confound_warning": tone_warning,
        "yushi_candidate_curve": [round(float(v), 2) for v in yg],
    }


def extract_voice_quality_metrics(
    y: np.ndarray, sr: int, s_sec: float, e_sec: float, f0_hz: float, hnr_db: float
) -> Dict[str, Any]:
    dur = e_sec - s_sec
    seg = y[max(0, int((s_sec + 0.30 * dur) * sr)) : min(len(y), int((s_sec + 0.70 * dur) * sr))]
    if len(seg) < 128 or f0_hz <= 50.0:
        return {"cpp_db": 0.0, "h1_h2_db": 0.0, "h1_a1_db": 0.0, "h1_a3_db": 0.0, "hnr_db": round(hnr_db, 2), "voiced": False}
    n_fft = 1
    while n_fft < max(2048, len(seg)):
        n_fft *= 2
    xw = (seg - np.mean(seg)) * get_window("hann", len(seg))
    log_mag = 20.0 * np.log10(np.abs(np.fft.rfft(xw, n=n_fft)) + 1e-12)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    cep = np.fft.irfft(log_mag)
    qmin, qmax = max(2, int(sr / 550.0)), min(len(cep) // 2 - 1, int(sr / 65.0))
    cpp = 0.0
    if qmax > qmin + 4:
        qr = np.arange(qmin, qmax + 1, dtype=np.float64)
        cr = cep[qmin : qmax + 1]
        sl, ic = np.polyfit(qr, cr, 1)
        pi = int(np.argmax(cr))
        cpp = float(cr[pi] - (sl * qr[pi] + ic))

    def pk(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs <= hi)
        return float(np.max(log_mag[m])) if np.any(m) else float(np.median(log_mag))

    bw = max(20.0, 0.12 * f0_hz)
    h1 = pk(max(50.0, f0_hz - bw), f0_hz + bw)
    h2 = pk(max(100.0, 2 * f0_hz - bw), 2 * f0_hz + bw)
    a1 = pk(300.0, 900.0)
    a3 = pk(2200.0, min(3500.0, 0.48 * sr))
    return {
        "cpp_db": round(cpp, 3),
        "h1_h2_db": round(h1 - h2, 2),
        "h1_a1_db": round(h1 - a1, 2),
        "h1_a3_db": round(h1 - a3, 2),
        "hnr_db": round(float(hnr_db), 2),
        "voiced": True,
    }


def _z(arr: np.ndarray, clip: float = 3.0) -> np.ndarray:
    if len(arr) <= 1 or float(np.std(arr)) < 1e-6:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - float(np.mean(arr))) / float(np.std(arr)), -clip, clip)


def analyze_osot_v2(doc_in: Dict[str, Any], wav_path: str, speaker_id: str = "spk_01") -> Dict[str, Any]:
    y, sr = load_wav_mono(wav_path)
    duration_sec = len(y) / float(sr) if sr > 0 else float(doc_in.get("duration_sec", 0.0))
    u_prev = migrate_to_unified(doc_in, default_speaker_id=speaker_id)
    tokens_in = u_prev["tier1_acoustic"]["tokens"]
    if not tokens_in:
        raise ValueError("输入 JSON 中 tokens 为空。")

    hop_sec = 0.005
    times, f0_hz, db_vals, hnr_vals, pitch_meta = extract_f0_hirst_twopass(y, sr, hop_sec=hop_sec)
    f0_st, segs = interpolate_f0_semitone_gap_safe(f0_hz, pitch_meta["speaker_f0_median_hz"], hop_sec=hop_sec)
    tones, tone_backend = infer_mandarin_tones(tokens_in, times, f0_st)
    f0_local, f0_macro, cwt_prom, tone_bias = tone_offset_neutralized_cwt(
        f0_st, segs, tokens_in, tones, times, hop_sec=hop_sec
    )

    n = len(tokens_in)
    spans, engs, durs, gaps, cwts, macs = [np.zeros(n, dtype=np.float64) for _ in range(6)]
    vq_raw: List[Dict[str, Any]] = []
    for i, tok in enumerate(tokens_in):
        s_ms, e_ms = float(tok["start_ms"]), float(tok["end_ms"])
        s_s, e_s = s_ms / 1000.0, e_ms / 1000.0
        durs[i] = max(20.0, e_ms - s_ms)
        gaps[i] = max(0.0, float(tokens_in[i + 1]["start_ms"]) - e_ms) if i + 1 < n else 0.0
        mf = (times >= s_s) & (times <= e_s)
        mc = (times >= s_s + 0.15 * (e_s - s_s)) & (times <= s_s + 0.85 * (e_s - s_s))
        if not np.any(mf) and len(times) > 0:
            mf[int(np.argmin(np.abs(times - 0.5 * (s_s + e_s))))] = True
            mc = mf
        lv = f0_local[mc]
        lv = lv[~np.isnan(lv)]
        spans[i] = float(np.percentile(lv, 95) - np.percentile(lv, 5)) if len(lv) >= 2 else 0.0
        mv = f0_macro[mf]
        mv = mv[~np.isnan(mv)]
        macs[i] = float(np.mean(mv)) if len(mv) > 0 else 0.0
        engs[i] = float(np.mean(db_vals[mf])) if np.any(mf) else 0.0
        cwts[i] = float(np.max(cwt_prom[mf])) if np.any(mf) else 0.0
        f0v = f0_hz[mc]
        f0v = f0v[f0v > 0]
        f0_c = float(np.median(f0v)) if len(f0v) > 0 else 0.0
        hnr_c = float(np.mean(hnr_vals[mc])) if np.any(mc) else 0.0
        vq_raw.append(extract_voice_quality_metrics(y, sr, s_s, e_s, f0_c, hnr_c))

    # 发声态声学候选（严格标明仅作声学代理候选，不等同于艺术表达意图）
    v_idx = [i for i, v in enumerate(vq_raw) if v["voiced"]]
    vq_cands = [{"label": "Unvoiced", "cn": "清音/无声", "confidence": 0.9, "breathiness_index": 0.0} for _ in range(n)]
    if len(v_idx) >= 3:
        z_cpp = _z(np.array([vq_raw[i]["cpp_db"] for i in v_idx]))
        z_hnr = _z(np.array([vq_raw[i]["hnr_db"] for i in v_idx]))
        z_tilt = _z(np.array([0.5 * vq_raw[i]["h1_h2_db"] + 0.3 * vq_raw[i]["h1_a1_db"] + 0.2 * vq_raw[i]["h1_a3_db"] for i in v_idx]))
        z_e = _z(engs)
        for k, idx in enumerate(v_idx):
            bi = float(0.45 * z_tilt[k] - 0.35 * z_cpp[k] - 0.20 * z_hnr[k])
            if macs[idx] < -2.2 and z_hnr[k] < -0.9 and z_tilt[k] < 0.2:
                lbl, cn, conf = "Creaky", "喉化/气泡音（声学候选）", min(0.92, 0.55 + 0.15 * abs(z_hnr[k]))
            elif bi >= 1.15:
                lbl, cn, conf = "Breathy", "气声（声学候选）", min(0.92, 0.55 + 0.18 * (bi - 1.0))
            elif bi >= 0.45 or (z_e[idx] < -0.8 and z_tilt[k] > 0.2):
                lbl, cn, conf = "Soft", "虚声/柔声（声学候选）", min(0.85, 0.50 + 0.15 * max(bi, 0.2))
            else:
                lbl, cn, conf = "Modal", "实声（声学候选）", min(0.90, 0.60 + 0.12 * max(0.0, -bi))
            vq_cands[idx] = {"label": lbl, "cn": cn, "confidence": round(conf, 3), "breathiness_index": round(bi, 3)}

    # 停连候选（B1 虚连显式标为候选需人工确认）
    pause_cands: List[Dict[str, Any]] = []
    dsrs = np.ones(n, dtype=np.float64)
    resets = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - 4)
        prev_d = durs[lo:i] if i > lo else durs[i : min(n, i + 3)]
        dsr = durs[i] / max(float(np.median(prev_d)), 40.0)
        dsrs[i] = dsr
        rst = float(macs[i + 1] - macs[i]) if i + 1 < n else 0.0
        resets[i] = rst
        gap = gaps[i]
        punct = tokens_in[i].get("punct_after", "")
        if i == n - 1:
            lvl, cn, conf, need_rev, rsn = "B3", "句尾落停（候选）", 0.95, False, "篇末收束"
        elif gap >= 350.0 or (gap >= 260.0 and any(p in STRONG_PUNCT_SET for p in punct)):
            lvl, cn, conf, need_rev, rsn = "B3", "句群长停（候选）", 0.90, False, f"物理静音 {gap:.0f}ms"
        elif gap >= 120.0:
            lvl, cn, conf, need_rev, rsn = "B2", "短语实停（候选）", 0.82, False, f"物理静音 {gap:.0f}ms"
        elif (gap >= 45.0 and (dsr >= 1.25 or abs(rst) >= 1.5)) or (dsr >= 1.45 and abs(rst) >= 1.2):
            lvl, cn, conf, need_rev = "B1", "虚连/微气口（候选·需人工复核）", 0.62, True
            rsn = f"微间隙{gap:.0f}ms + 延长{dsr:.2f}x + F0重置{rst:+.1f}st"
        elif any(p in PAUSE_PUNCT_SET for p in punct) and gap < 45.0:
            lvl, cn, conf, need_rev, rsn = "B0_CONNECT", "跨标点直连（候选）", 0.72, True, f"标点处无停顿({gap:.0f}ms)"
        else:
            lvl, cn, conf, need_rev, rsn = "B0", "自然连读", 0.85, False, "无边界线索"
        pause_cands.append({
            "level": lvl, "label_cn": cn, "confidence": conf, "needs_human_review": need_rev,
            "gap_ms": round(gap, 1), "dsr_lengthening_ratio": round(dsr, 3), "f0_reset_st": round(rst, 2), "reason": rsn,
        })

    # 短语切分 + v2 语势候选与置信度
    phrases = []
    ps = 0
    for i in range(n):
        if pause_cands[i]["level"] in ("B2", "B3") or i == n - 1:
            t_s, t_e = tokens_in[ps]["start_ms"] / 1000.0, tokens_in[i]["end_ms"] / 1000.0
            mp = (times >= t_s) & (times <= t_e)
            leg = fit_legendre_yushi_with_confidence(
                f0_macro[mp], tones[ps : i + 1], engs[ps : i + 1], durs[ps : i + 1]
            )
            lsr = (i - ps + 1) / max(0.08, t_e - t_s)
            cv = float(np.std(durs[ps : i + 1]) / max(np.mean(durs[ps : i + 1]), 1.0)) if i > ps else 0.0
            phrases.append({
                "phrase_id": len(phrases) + 1, "token_start": ps, "token_end": i,
                "start_ms": int(tokens_in[ps]["start_ms"]), "end_ms": int(tokens_in[i]["end_ms"]),
                "text": "".join(tokens_in[k]["char"] + tokens_in[k].get("punct_after", "") for k in range(ps, i + 1)),
                "legendre_yushi": leg,
                "rhythm": {
                    "local_speech_rate_syll_per_sec": round(lsr, 2), "duration_cv": round(cv, 3),
                    "mode": "Fast" if lsr >= 4.6 else ("Slow" if lsr <= 2.9 else "Steady"),
                },
            })
            ps = i + 1

    # 同调型归一化重音 + PFC + 白箱归因
    z_span = np.zeros(n, dtype=np.float64)
    t_arr = np.array(tones, dtype=int)
    for tc in (1, 2, 3, 4, 5):
        ic = np.where(t_arr == tc)[0]
        if len(ic) >= 3:
            z_span[ic] = _z(spans[ic])
        elif len(ic) > 0:
            z_span[ic] = _z(spans)[ic]
    z_eng, z_dur, z_cwt = _z(engs), _z(durs), _z(cwts)
    pfc = np.zeros(n, dtype=np.float64)
    for i in range(n - 1):
        if pause_cands[i]["level"] not in ("B2", "B3"):
            hi = min(n, i + 3)
            drop = 0.5 * (z_eng[i] - float(np.mean(z_eng[i + 1 : hi]))) + 0.5 * (z_span[i] - float(np.mean(z_span[i + 1 : hi])))
            if drop > 0.8:
                pfc[i] = min(0.8, 0.35 * (drop - 0.5))

    u = empty_unified_doc(
        doc_id=u_prev["meta"]["id"], title=u_prev["meta"]["title"],
        speaker_id=u_prev["meta"].get("speaker_id", speaker_id),
        speaker_group=u_prev["meta"].get("speaker_group", "unspecified"),
        genre_style=u_prev["meta"].get("genre_style", "文艺朗诵"),
        global_rhythm=u_prev["meta"].get("global_rhythm", "舒缓型"),
        audio_wav=str(Path(wav_path).name), duration_sec=duration_sec,
    )
    u["raw_text"] = u_prev["raw_text"]
    u["tier1_acoustic"]["analysis_meta"] = {
        **pitch_meta, "tone_backend": tone_backend, "speaker_tone_bias_st": tone_bias,
    }
    def _ds(arr: np.ndarray, m: int = 800) -> List[Optional[float]]:
        if len(arr) == 0:
            return []
        idx = np.linspace(0, len(arr) - 1, min(len(arr), m)).astype(int)
        return [None if np.isnan(arr[j]) else round(float(arr[j]), 2) for j in idx]
    u["tier1_acoustic"]["curves"] = {
        "times_sec": _ds(times), "f0_raw_st": _ds(f0_st),
        "f0_local_st": _ds(f0_local), "f0_macro_candidate_st": _ds(f0_macro), "energy_db": _ds(db_vals),
    }
    u["tier2_machine_pred"]["model_version"] = "osot_prosody_analyzer_v2.0"
    u["tier2_machine_pred"]["phrases"] = phrases

    old_t3 = u_prev["tier3_human_gold"]["tokens"]
    for i, tok in enumerate(tokens_in):
        ch = tok["char"]
        lp = -0.65 if ch in WEAK_FUNCTION_CHARS else (0.25 if ch in CONTENT_BOOST_CHARS else 0.0)
        cs, ce, cd, cc, cp = 0.28 * z_span[i], 0.30 * z_eng[i], 0.24 * z_dur[i], 0.18 * z_cwt[i], 0.20 * pfc[i]
        prob = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, cs + ce + cd + cc + cp + lp - 0.35))))
        slvl = "S2" if prob >= 0.72 else ("S1" if prob >= 0.52 else "S0")

        u["tier1_acoustic"]["tokens"].append({
            "index": i, "char": ch, "punct_after": tok.get("punct_after", ""),
            "start_ms": int(tok["start_ms"]), "end_ms": int(tok["end_ms"]),
            "duration_ms": int(round(durs[i])), "gap_after_ms": int(round(gaps[i])),
            "mandarin_tone": int(tones[i]), "energy_db": round(float(engs[i]), 2),
            "f0_local_span_st": round(float(spans[i]), 3), "f0_macro_mean_st": round(float(macs[i]), 3),
            "cwt_prominence_raw": round(float(cwts[i]), 3), "dsr_lengthening_ratio": round(float(dsrs[i]), 3),
            "f0_reset_after_st": round(float(resets[i]), 2),
            "voice_quality_metrics": {**vq_raw[i], "breathiness_index": vq_cands[i]["breathiness_index"]},
        })
        u["tier2_machine_pred"]["tokens"].append({
            "index": i, "char": ch,
            "candidate_stress": slvl, "stress_prob": round(prob, 3),
            "is_elongated": bool(z_dur[i] >= 1.35 or durs[i] >= 480.0),
            "feature_attribution": {
                "tone_norm_f0_span": round(float(cs), 3), "energy_lift": round(float(ce), 3),
                "duration_elongation": round(float(cd), 3), "cwt_multiscale": round(float(cc), 3),
                "post_focus_compression": round(float(cp), 3), "lexical_prior": round(float(lp), 3),
            },
            "candidate_pause": pause_cands[i],
            "candidate_voice_quality": vq_cands[i]["label"],
            "candidate_voice_quality_cn": vq_cands[i]["cn"],
            "voice_quality_confidence": vq_cands[i]["confidence"],
        })
        g_prev = old_t3[i] if i < len(old_t3) else {}
        u["tier3_human_gold"]["tokens"].append({
            "index": i, "char": ch,
            "human_stress": g_prev.get("human_stress", "UNVERIFIED"),
            "human_stress_semantic_type": g_prev.get("human_stress_semantic_type", ""),
            "human_pause": g_prev.get("human_pause", "UNVERIFIED"),
            "human_pause_method": g_prev.get("human_pause_method", ""),
            "human_voice_quality": g_prev.get("human_voice_quality", "UNVERIFIED"),
            "human_breath_action": g_prev.get("human_breath_action", ""),
            "human_sentence_contour": g_prev.get("human_sentence_contour", ""),
            "knowledge_tags": list(g_prev.get("knowledge_tags", [])),
        })

    u["tier3_human_gold"]["edit_history"] = list(u_prev["tier3_human_gold"].get("edit_history", []))
    return u


def build_demo_case(out_dir: Path) -> Tuple[Path, Path]:
    sr = 16000
    script = [
        ("让", "", 190, 20, 4, 0.85, "Modal", -1.5),
        ("暴", "", 220, 15, 4, 1.00, "Modal", -0.5),
        ("风", "", 210, 10, 1, 0.90, "Modal", +0.5),
        ("雨", "", 290, 65, 3, 1.10, "Modal", +1.2),
        ("来", "", 180, 15, 2, 0.75, "Soft", +0.8),
        ("得", "", 110, 10, 5, 0.55, "Soft", +0.6),
        ("更", "", 270, 20, 4, 1.25, "Modal", +2.5),
        ("猛", "", 250, 15, 3, 1.15, "Modal", +3.0),
        ("烈", "", 260, 15, 4, 1.10, "Modal", +3.4),
        ("些", "", 180, 10, 1, 0.70, "Breathy", +2.2),
        ("吧", "！", 320, 420, 5, 0.50, "Breathy", +0.5),
        ("夜", "", 240, 20, 4, 0.65, "Breathy", -0.8),
        ("色", "", 210, 15, 4, 0.60, "Soft", -0.4),
        ("渐", "", 260, 20, 4, 0.85, "Modal", +1.4),
        ("深", "。", 340, 200, 1, 0.55, "Creaky", -2.8),
    ]
    rng = np.random.default_rng(42)
    chunks, tokens, cursor = [np.zeros(int(0.08 * sr))], [], 80
    for idx, (ch, p, d, g, tn, amp, vq, mac) in enumerate(script):
        ns = int(sr * d / 1000.0)
        u = np.linspace(0, 1, ns)
        tst = np.full_like(u, 1.5) if tn == 1 else (-1 + 3.2 * u if tn == 2 else (-1.5 - 2 * np.sin(np.pi * u) if tn == 3 else (3 - 5.2 * u if tn == 4 else np.full_like(u, -0.5))))
        ph = 2 * np.pi * np.cumsum(175.0 * (2 ** ((mac + tst) / 12.0))) / sr
        if vq == "Breathy":
            sig = np.sin(ph) + 0.12 * np.sin(2 * ph) + 0.35 * rng.normal(0, 1, ns)
        elif vq == "Soft":
            sig = 0.9 * np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.14 * rng.normal(0, 1, ns)
        elif vq == "Creaky":
            sig = (0.7 * np.sin(ph) + 0.6 * np.sin(2 * ph)) * (0.55 + 0.45 * np.sin(0.5 * ph)) + 0.18 * rng.normal(0, 1, ns)
        else:
            sig = 0.75 * np.sin(ph) + 0.65 * np.sin(2 * ph) + 0.5 * np.sin(3 * ph) + 0.02 * rng.normal(0, 1, ns)
        sig = sig / (np.max(np.abs(sig)) + 1e-6) * amp * 0.7 * (np.sin(np.pi * u) ** 0.35)
        chunks.append(sig)
        tokens.append({"index": idx, "char": ch, "punct_after": p, "start_ms": cursor, "end_ms": cursor + d, "stress": "none", "pause_after": "none"})
        cursor += d
        if g > 0:
            chunks.append(rng.normal(0, 0.001, int(sr * g / 1000.0)))
            cursor += g
    y_all = np.concatenate(chunks)
    wout, jout = out_dir / "demo_v2.wav", out_dir / "demo_v2_input.json"
    save_wav_mono_16bit(str(wout), y_all, sr)
    jout.write_text(json.dumps({"id": "demo_v2", "title": "OSOT v2.0 演示", "speaker_id": "spk_demo_01", "tokens": tokens}, ensure_ascii=False, indent=2), encoding="utf-8")
    return jout, wout


def main() -> None:
    ap = argparse.ArgumentParser(description="OSOT 五维声学深度剖析引擎 v2.0 (统一协议)")
    ap.add_argument("--json", dest="json_path", help="输入 JSON（兼容 v0.1 / v2.0）")
    ap.add_argument("--wav", dest="wav_path", help="对应纯人声 WAV")
    ap.add_argument("--speaker-id", default="spk_01", help="说话人 ID（用于防数据泄漏划分）")
    ap.add_argument("--out-dir", default="osot_v2_out", help="输出目录")
    ap.add_argument("--demo", action="store_true", help="运行内置 Demo 自测")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.demo:
        jp, wp = build_demo_case(out_dir)
    else:
        if not args.json_path or not args.wav_path:
            ap.error("请提供 --json 与 --wav，或使用 --demo。")
        jp, wp = Path(args.json_path), Path(args.wav_path)

    doc = json.loads(jp.read_text(encoding="utf-8"))
    res = analyze_osot_v2(doc, str(wp), speaker_id=args.speaker_id)
    out_json = out_dir / f"{jp.stem}_unified_v2.json"
    out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"=== OSOT 工具 01 v2.0 完成 ({SCHEMA_VERSION}) ===")
    print(f"输出统一档案: {out_json}")
    for p in res["tier2_machine_pred"]["phrases"]:
        ly = p["legendre_yushi"]
        warn = f" | ⚠️ {ly['tone_confound_warning']}" if ly["tone_confound_warning"] else ""
        print(f"  短语#{p['phrase_id']} 「{p['text']}」 -> {ly['candidate_yushi_cn']} (conf={ly['yushi_confidence']}, r={ly['pearson_r']}){warn}")


if __name__ == "__main__":
    main()
