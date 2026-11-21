#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSOT 五维播音声学深度剖析与三层解耦升级引擎 (osot_prosody_analyzer.py)
====================================================================
专为中文专业播音与 OSOT 标注程序 (D:\\otos\\osot-annotator) 打造的原创声学分析工具。

核心能力（填补现有开源工具链针对中文播音的空白）：
1. Hirst 两遍迭代基频追踪 + 半音归一化 + 实停顿截断型单调三次 Hermite 插值
2. 墨西哥帽连续小波变换 (CWT) 多尺度解耦：分离局部声调层 F0_local 与宏观语势包络 F0_macro
3. 三阶正交勒让德多项式 (c0, c1, c2, c3) 连续语势拟合 + 中国播音学「语势五形态」映射
4. 普通话同调型内部归一化重音 + 焦点后压缩 (PFC) + 白箱特征归因向量 (Feature Attribution)
5. 播音「虚实声」发声态连续谱：CPP 倒谱峰突出度 + H1-H2/H1-A1/H1-A3 频谱倾斜 + HNR 二维联合判别
6. 播音「虚连/微气口 (B1)」与实停 (B2/B3) 声学判别（结合边界前延长比 DSR + 跨界 F0 重置 + 喉化线索）
7. 输出三层解耦 JSON (Objective / Hypothesis / Human Gold) + 6 层 Praat TextGrid + 交互式 HTML 可视化报告

零重型依赖：仅需 numpy + scipy（自动兼容使用 parselmouth / soundfile / pypinyin 增强）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt, get_window

# 中文标点与轻声虚词集合（与 osot-annotator/app/pipeline.py 保持一致）
PAUSE_PUNCT_SET = set("，。！？、；：,.!?;:")
STRONG_PUNCT_SET = set("。！？!?；;")
WEAK_FUNCTION_CHARS = set("的了着过吗呢吧啊呀嘛地得之乎者也么啦哇哦嗯与和及或而")
CONTENT_BOOST_CHARS = set("不没无非最极更全所凡绝必竟却但然才就都只仅")


# ---------------------------------------------------------------------------
# 1. 音频读取（零依赖标准库 wave + 可选 soundfile）
# ---------------------------------------------------------------------------
def load_wav_mono(wav_path: str) -> Tuple[np.ndarray, int]:
    """读取单声道浮点音频 [-1.0, 1.0]，优先 soundfile，缺失时回退标准库 wave。"""
    try:
        import soundfile as sf  # type: ignore

        y, sr = sf.read(wav_path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y.astype(np.float64), int(sr)
    except Exception:
        pass

    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    else:
        raise ValueError(f"不支持的 WAV 位深: {sampwidth * 8} bit，请先转为 16-bit WAV 或安装 soundfile。")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, int(sr)


def save_wav_mono_16bit(wav_path: str, y: np.ndarray, sr: int) -> None:
    """保存 16-bit 单声道 WAV（用于内置 Demo 演示）。"""
    y_clip = np.clip(y, -0.99, 0.99)
    pcm = (y_clip * 32767.0).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# 2. Hirst 两遍迭代基频提取 + 能量包络（5ms 帧移）
# ---------------------------------------------------------------------------
def _autocorr_pitch_track(
    y: np.ndarray,
    sr: int,
    hop_sec: float = 0.005,
    win_sec: float = 0.040,
    fmin: float = 65.0,
    fmax: float = 550.0,
    voicing_thresh: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """纯 NumPy/SciPy 自相关基频 + 谐噪比 (HNR) + 能量提取器。"""
    hop = max(1, int(round(sr * hop_sec)))
    win_len = max(64, int(round(sr * win_sec)))
    n_frames = max(1, 1 + max(0, len(y) - win_len) // hop)

    # 轻度带通滤波（50–1200Hz）用于更稳健的基频自相关
    nyq = 0.5 * sr
    low = max(40.0 / nyq, 0.001)
    high = min(1200.0 / nyq, 0.95)
    b, a = butter(3, [low, high], btype="band")
    y_bp = filtfilt(b, a, y) if len(y) > 30 else y.copy()

    window = get_window("hann", win_len)
    lag_min = max(2, int(sr / fmax))
    lag_max = min(win_len - 2, int(sr / fmin))

    times = np.zeros(n_frames, dtype=np.float64)
    f0 = np.zeros(n_frames, dtype=np.float64)
    db = np.zeros(n_frames, dtype=np.float64)
    hnr = np.zeros(n_frames, dtype=np.float64)

    global_rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    energy_gate = max(1e-4, global_rms * 0.08)

    for idx in range(n_frames):
        start = idx * hop
        end = start + win_len
        seg_raw = y[start:end]
        seg_bp = y_bp[start:end]
        if len(seg_raw) < win_len:
            seg_raw = np.pad(seg_raw, (0, win_len - len(seg_raw)))
            seg_bp = np.pad(seg_bp, (0, win_len - len(seg_bp)))

        times[idx] = (start + 0.5 * win_len) / sr
        rms = float(np.sqrt(np.mean(seg_raw ** 2) + 1e-12))
        db[idx] = 20.0 * math.log10(max(rms, 1e-6)) + 90.0

        if rms < energy_gate:
            continue

        xw = (seg_bp - np.mean(seg_bp)) * window
        spec = np.fft.rfft(xw, n=2 * win_len)
        acf = np.fft.irfft(spec * np.conj(spec))[:win_len]
        r0 = float(acf[0])
        if r0 <= 1e-12:
            continue
        acf_norm = acf / r0
        sub = acf_norm[lag_min : lag_max + 1]
        if len(sub) == 0:
            continue

        best_rel = int(np.argmax(sub))
        best_val = float(sub[best_rel])
        best_lag = lag_min + best_rel

        if best_val >= voicing_thresh:
            # 抛物线插值精细化 lag
            if 0 < best_rel < len(sub) - 1:
                y0, y1, y2 = sub[best_rel - 1], sub[best_rel], sub[best_rel + 1]
                denom = y0 - 2.0 * y1 + y2
                delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
                best_lag_f = best_lag + float(np.clip(delta, -0.5, 0.5))
            else:
                best_lag_f = float(best_lag)

            f0[idx] = sr / max(best_lag_f, 1.0)
            r_clamped = float(np.clip(best_val, 0.01, 0.995))
            hnr[idx] = 10.0 * math.log10(r_clamped / (1.0 - r_clamped))

    return times, f0, db, hnr


def extract_f0_hirst_twopass(
    y: np.ndarray, sr: int, hop_sec: float = 0.005
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Hirst 两遍迭代基频追踪：
    第 1 遍宽音域 [65, 550] Hz 探查说话人分布 (Q25, Q75)；
    第 2 遍按 floor=max(55, 0.75*Q25), ceiling=min(600, 1.5*Q75) 收紧边界，抑制八度跳变。
    """
    used_backend = "numpy-scipy-autocorr"
    try:
        import parselmouth  # type: ignore

        snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=float(sr))
        p1 = snd.to_pitch(time_step=hop_sec, pitch_floor=65.0, pitch_ceiling=550.0)
        vals1 = p1.selected_array["frequency"]
        voiced1 = vals1[vals1 > 0]
        if len(voiced1) >= 10:
            q25, q75 = np.percentile(voiced1, [25, 75])
            floor2 = max(55.0, float(0.75 * q25))
            ceil2 = min(600.0, max(floor2 + 80.0, float(1.5 * q75)))
        else:
            floor2, ceil2 = 65.0, 500.0

        p2 = snd.to_pitch(time_step=hop_sec, pitch_floor=floor2, pitch_ceiling=ceil2)
        inten = snd.to_intensity(minimum_pitch=floor2, time_step=hop_sec)
        harm = snd.to_harmonicity_cc(time_step=hop_sec, minimum_pitch=floor2)

        n_frames = max(1, int(len(y) / sr / hop_sec))
        times = np.arange(n_frames) * hop_sec + 0.5 * hop_sec
        f0 = np.zeros(n_frames, dtype=np.float64)
        db = np.zeros(n_frames, dtype=np.float64)
        hnr = np.zeros(n_frames, dtype=np.float64)
        for i, t in enumerate(times):
            pv = p2.get_value_at_time(float(t))
            f0[i] = 0.0 if (pv is None or np.isnan(pv)) else float(pv)
            iv = inten.get_value(float(t))
            db[i] = 0.0 if (iv is None or np.isnan(iv)) else float(iv)
            hv = harm.get_value(float(t))
            hnr[i] = 0.0 if (hv is None or np.isnan(hv) or hv < -100) else float(hv)
        used_backend = "praat-parselmouth"
    except Exception:
        times, f0_1, db, hnr = _autocorr_pitch_track(y, sr, hop_sec=hop_sec, fmin=65.0, fmax=550.0)
        voiced1 = f0_1[f0_1 > 0]
        if len(voiced1) >= 10:
            q25, q75 = np.percentile(voiced1, [25, 75])
            floor2 = max(55.0, float(0.75 * q25))
            ceil2 = min(600.0, max(floor2 + 80.0, float(1.5 * q75)))
            times, f0, db, hnr = _autocorr_pitch_track(
                y, sr, hop_sec=hop_sec, fmin=floor2, fmax=ceil2
            )
        else:
            floor2, ceil2 = 65.0, 550.0
            f0 = f0_1

    # 中值去毛刺 + 简单八度跳变平滑
    voiced_idx = np.where(f0 > 0)[0]
    if len(voiced_idx) >= 5:
        med_f0 = float(np.median(f0[voiced_idx]))
        for idx in voiced_idx:
            if f0[idx] > 1.85 * med_f0 and (f0[idx] * 0.5) >= floor2:
                f0[idx] *= 0.5
            elif f0[idx] < 0.55 * med_f0 and (f0[idx] * 2.0) <= ceil2:
                f0[idx] *= 2.0
    else:
        med_f0 = 180.0

    meta = {
        "pitch_backend": used_backend,
        "hop_ms": int(round(hop_sec * 1000)),
        "speaker_f0_median_hz": round(med_f0, 2),
        "hirst_floor_hz": round(floor2, 1),
        "hirst_ceiling_hz": round(ceil2, 1),
    }
    return times, f0, db, hnr, meta


# ---------------------------------------------------------------------------
# 3. 实停顿截断型单调三次插值 + 墨西哥帽 CWT 声调-语势剥离
# ---------------------------------------------------------------------------
def interpolate_f0_semitone_gap_safe(
    f0_hz: np.ndarray, f0_median_hz: float, hop_sec: float = 0.005, max_interp_ms: float = 150.0
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    将 F0 转为相对说话人中位数的半音序列：
    - 小于 max_interp_ms (150ms) 的清音辅音间隙使用单调三次 Hermite (PCHIP) 插值；
    - >= 150ms 的实停顿严格截断，返回连续呼吸段列表 [(start_idx, end_idx), ...]。
    """
    n = len(f0_hz)
    f0_st = np.full(n, np.nan, dtype=np.float64)
    voiced = f0_hz > 0
    if not np.any(voiced):
        return f0_st, []

    f0_st[voiced] = 12.0 * np.log2(f0_hz[voiced] / max(f0_median_hz, 1.0))
    max_gap_frames = int(round((max_interp_ms / 1000.0) / hop_sec))

    # 找出所有连续有声岛屿，合并间隙 < max_gap_frames 的相邻岛屿为同一发声段
    v_idx = np.where(voiced)[0]
    segments: List[Tuple[int, int]] = []
    seg_start = int(v_idx[0])
    prev = int(v_idx[0])
    for cur in v_idx[1:]:
        cur = int(cur)
        if cur - prev > max_gap_frames:
            segments.append((seg_start, prev + 1))
            seg_start = cur
        prev = cur
    segments.append((seg_start, prev + 1))

    # 段内做 PCHIP 单调三次插值
    for s, e in segments:
        sub = f0_st[s:e]
        valid_local = np.where(~np.isnan(sub))[0]
        if len(valid_local) >= 2:
            interp = PchipInterpolator(valid_local, sub[valid_local], extrapolate=False)
            xs = np.arange(len(sub))
            f0_st[s:e] = interp(xs)
        elif len(valid_local) == 1:
            f0_st[s:e] = sub[valid_local[0]]

    return f0_st, segments


def mexican_hat_cwt_decompose(
    f0_st: np.ndarray,
    segments: List[Tuple[int, int]],
    hop_sec: float = 0.005,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    参照 Suni et al. (2017) 采用墨西哥帽母小波 (Ricker) 对每段连续 F0_st 做多尺度 CWT 分解：
    - 尺度覆盖 80ms ~ 3200ms（倍频网格，每倍频程 4 个尺度）
    - 局部层 f0_local_st：150–600ms 重构（声调 + 词级焦点凸显）
    - 宏观层 f0_macro_st：>= 800ms 重构（跨短语宏观语势包络）
    - cwt_prominence：跨尺度小波正极大能量累计包络
    """
    n = len(f0_st)
    f0_local = np.full(n, np.nan, dtype=np.float64)
    f0_macro = np.full(n, np.nan, dtype=np.float64)
    cwt_prom = np.zeros(n, dtype=np.float64)

    # 构造 80ms ~ 3200ms 尺度网格（单位：帧）
    scales_ms = 80.0 * (2.0 ** (np.arange(0, 22) / 4.0))
    scales_ms = scales_ms[scales_ms <= 3300.0]

    for s, e in segments:
        seg = f0_st[s:e]
        if len(seg) < 6 or np.all(np.isnan(seg)):
            continue
        seg_clean = np.nan_to_num(seg, nan=float(np.nanmedian(seg)))
        seg_mean = float(np.mean(seg_clean))
        sig = seg_clean - seg_mean
        L = len(sig)

        local_acc = np.zeros(L, dtype=np.float64)
        macro_acc = np.zeros(L, dtype=np.float64)
        prom_acc = np.zeros(L, dtype=np.float64)
        w_local_sum = 0.0
        w_macro_sum = 0.0

        for sc_ms in scales_ms:
            # 墨西哥帽小波标准差 sigma 与特征周期 T ≈ 4*sigma 对应
            sigma_frames = max(1.0, (sc_ms / 1000.0) / hop_sec / 4.0)
            half_w = min(max(4, int(round(4.0 * sigma_frames))), max(4, L))
            t = np.arange(-half_w, half_w + 1, dtype=np.float64)
            u = t / sigma_frames
            wavelet = (1.0 - u ** 2) * np.exp(-0.5 * u ** 2)
            # 零均值化并按尺度归一化能量
            wavelet -= np.mean(wavelet)
            norm = np.sqrt(np.sum(wavelet ** 2)) + 1e-12
            wavelet /= norm

            # 反射填充避免短语边缘效应畸变
            pad = min(half_w, L - 1)
            sig_pad = np.pad(sig, pad, mode="reflect")
            conv = np.convolve(sig_pad, wavelet, mode="same")
            coeff = conv[pad : pad + L]

            weight = 1.0 / math.sqrt(sc_ms)
            if 150.0 <= sc_ms <= 600.0:
                local_acc += coeff * weight
                w_local_sum += weight
                prom_acc += np.maximum(coeff, 0.0) * weight
            elif sc_ms >= 800.0:
                macro_acc += coeff * weight
                w_macro_sum += weight

        # 使用高斯低通作为稳健宏观趋势基底，与大尺度 CWT 融合以确保语势曲线物理幅值保真
        macro_sigma = max(2.0, (0.25 / hop_sec))  # ~250ms 平滑核滤除单字声调折转
        half_g = min(int(round(3.0 * macro_sigma)), max(2, L - 1))
        tg = np.arange(-half_g, half_g + 1, dtype=np.float64)
        g_kern = np.exp(-0.5 * (tg / macro_sigma) ** 2)
        g_kern /= np.sum(g_kern)
        pad_g = min(half_g, L - 1)
        sig_pg = np.pad(seg_clean, pad_g, mode="reflect")
        macro_smooth = np.convolve(sig_pg, g_kern, mode="same")[pad_g : pad_g + L]

        if w_macro_sum > 0:
            macro_cwt = (macro_acc / w_macro_sum)
            # 保持低通包络的绝对半音尺度，同时叠加 CWT 宏观曲率
            macro_final = 0.85 * macro_smooth + 0.15 * (seg_mean + macro_cwt)
        else:
            macro_final = macro_smooth

        local_final = seg_clean - macro_final
        if w_local_sum > 0:
            cwt_prom[s:e] = prom_acc / w_local_sum

        f0_macro[s:e] = macro_final
        f0_local[s:e] = local_final

    return f0_local, f0_macro, cwt_prom


# ---------------------------------------------------------------------------
# 4. 三阶正交勒让德多项式连续语势拟合 + 中国播音学「语势五形态」映射
# ---------------------------------------------------------------------------
def fit_legendre_yushi(f0_macro_seg: np.ndarray) -> Dict[str, Any]:
    """
    在韵律短语区间上将时间归一化至 t in [-1, 1]，最小二乘拟合三阶正交勒让德多项式：
      P0(t) = 1
      P1(t) = t
      P2(t) = 0.5 * (3t^2 - 1)
      P3(t) = 0.5 * (5t^3 - 3t)
    并映射到中国播音学「语势五形态」：
      上山类(Rise) / 下山类(Fall) / 波峰类(Arch-Peak) / 波谷类(Arch-Valley) / 平直或复合类(Level/Complex)
    """
    valid = ~np.isnan(f0_macro_seg)
    n = len(f0_macro_seg)
    if np.sum(valid) < 5 or n < 5:
        return {
            "c0_level": 0.0,
            "c1_slope": 0.0,
            "c2_curvature": 0.0,
            "c3_inflection": 0.0,
            "rmse_st": 0.0,
            "pearson_r": 0.0,
            "yushi_label": "Level",
            "yushi_cn": "平直类（有效帧不足）",
            "fitted_curve_preview": [],
        }

    t_all = np.linspace(-1.0, 1.0, n)
    t = t_all[valid]
    y = f0_macro_seg[valid]

    P0 = np.ones_like(t)
    P1 = t
    P2 = 0.5 * (3.0 * t ** 2 - 1.0)
    P3 = 0.5 * (5.0 * t ** 3 - 3.0 * t)
    X = np.column_stack([P0, P1, P2, P3])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coeffs
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    if np.std(y) > 1e-6 and np.std(y_hat) > 1e-6:
        r_val = float(np.corrcoef(y, y_hat)[0, 1])
    else:
        r_val = 0.0

    c0, c1, c2, c3 = [float(c) for c in coeffs]

    # 播音学语势五形态规则映射（可解释几何判别）
    # 注意：P2(t) 在 t=0 为 -0.5，在 t=±1 为 +1.0；
    # 因此 c2 < 0 对应“两头低中间高（波峰类）”，c2 > 0 对应“两头高中间低（波谷类）”！
    abs_c1, abs_c2, abs_c3 = abs(c1), abs(c2), abs(c3)
    if max(abs_c1, abs_c2, abs_c3) < 0.45:
        label, cn = "Level", "平直类（平稳推进）"
    elif abs_c3 >= 1.15 and abs_c3 > max(abs_c1, abs_c2):
        label, cn = "Complex", "复合转折类（S型抑扬）"
    elif abs_c2 >= 0.75 and abs_c2 >= 0.85 * abs_c1:
        if c2 < 0:
            label, cn = "Arch-Peak", "波峰类（低→高→低）"
        else:
            label, cn = "Arch-Valley", "波谷类（高→低→高）"
    elif c1 >= 0.50:
        label, cn = "Rise", "上山类 / 起潮类（低→高）"
    elif c1 <= -0.50:
        label, cn = "Fall", "下山类 / 落潮类（高→低）"
    else:
        label, cn = "Level", "平直类（微幅起伏）"

    # 25 点轻量化拟合曲线，供前端/HTML 报告直接绘制
    t_grid = np.linspace(-1.0, 1.0, 25)
    y_grid = (
        c0
        + c1 * t_grid
        + c2 * 0.5 * (3.0 * t_grid ** 2 - 1.0)
        + c3 * 0.5 * (5.0 * t_grid ** 3 - 3.0 * t_grid)
    )

    return {
        "c0_level": round(c0, 3),
        "c1_slope": round(c1, 3),
        "c2_curvature": round(c2, 3),
        "c3_inflection": round(c3, 3),
        "rmse_st": round(rmse, 3),
        "pearson_r": round(r_val, 3),
        "yushi_label": label,
        "yushi_cn": cn,
        "fitted_curve_preview": [round(float(v), 2) for v in y_grid],
    }


# ---------------------------------------------------------------------------
# 5. 播音「虚实声」发声态连续谱（CPP + H1-H2 + H1-A1/A3 + HNR）
# ---------------------------------------------------------------------------
def extract_voice_quality_for_segment(
    y: np.ndarray, sr: int, start_sec: float, end_sec: float, f0_hz: float, hnr_db: float
) -> Dict[str, Any]:
    """
    在单字韵腹稳态区（中间 30%–70%）提取发声态物理参量：
    - CPP (Cepstral Peak Prominence, dB)：倒谱峰相对线性回归基底的突出度（气声显著偏低）
    - H1-H2 (dB)：基频与二次谐波能量差
    - H1-A1 / H1-A3 (dB)：基频相对第一共振峰区 (300–900Hz) 与第三共振峰区 (2200–3500Hz) 最强峰差值
      （规避单用 H1-H2 无法区分耳语态/气声/嘎裂的声学陷阱）
    """
    dur = end_sec - start_sec
    s_idx = max(0, int((start_sec + 0.30 * dur) * sr))
    e_idx = min(len(y), int((start_sec + 0.70 * dur) * sr))
    seg = y[s_idx:e_idx]

    if len(seg) < 128 or f0_hz <= 50.0:
        return {
            "cpp_db": 0.0,
            "h1_h2_db": 0.0,
            "h1_a1_db": 0.0,
            "h1_a3_db": 0.0,
            "hnr_db": round(float(hnr_db), 2),
            "breathiness_index": 0.0,
            "voice_quality": "Unvoiced",
            "voice_quality_cn": "清音/无声",
        }

    # 1. 高分辨率对数功率谱
    n_fft = 1
    while n_fft < max(2048, len(seg)):
        n_fft *= 2
    xw = (seg - np.mean(seg)) * get_window("hann", len(seg))
    mag = np.abs(np.fft.rfft(xw, n=n_fft)) + 1e-12
    log_mag_db = 20.0 * np.log10(mag)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    # 2. CPP (实倒谱峰突出度)
    cepstrum = np.fft.irfft(log_mag_db)
    q_min = max(2, int(sr / 550.0))
    q_max = min(len(cepstrum) // 2 - 1, int(sr / 65.0))
    if q_max > q_min + 4:
        q_region = np.arange(q_min, q_max + 1, dtype=np.float64)
        c_region = cepstrum[q_min : q_max + 1]
        # 线性回归基线
        slope, intercept = np.polyfit(q_region, c_region, 1)
        baseline = slope * q_region + intercept
        peak_i = int(np.argmax(c_region))
        cpp_db = float(c_region[peak_i] - baseline[peak_i])
    else:
        cpp_db = 0.0

    # 3. 谐波峰 H1, H2 与共振峰频段最强谐波 A1 (300-900Hz), A3 (2200-3500Hz)
    def peak_in_band(f_lo: float, f_hi: float) -> float:
        m = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(m):
            return float(np.median(log_mag_db))
        return float(np.max(log_mag_db[m]))

    bw = max(20.0, 0.12 * f0_hz)
    h1 = peak_in_band(max(50.0, f0_hz - bw), f0_hz + bw)
    h2 = peak_in_band(max(100.0, 2.0 * f0_hz - bw), 2.0 * f0_hz + bw)
    a1 = peak_in_band(300.0, 900.0)
    a3 = peak_in_band(2200.0, min(3500.0, 0.48 * sr))

    h1_h2 = h1 - h2
    h1_a1 = h1 - a1
    h1_a3 = h1 - a3

    return {
        "cpp_db": round(cpp_db, 3),
        "h1_h2_db": round(h1_h2, 2),
        "h1_a1_db": round(h1_a1, 2),
        "h1_a3_db": round(h1_a3, 2),
        "hnr_db": round(float(hnr_db), 2),
        "breathiness_index": 0.0,  # 待全篇归一化后填充
        "voice_quality": "Modal",
        "voice_quality_cn": "实声",
    }


# ---------------------------------------------------------------------------
# 6. 普通话声调估测（优先 pypinyin，缺失时采用字内 F0 几何调型回退）
# ---------------------------------------------------------------------------
def infer_mandarin_tones(
    tokens: List[Dict[str, Any]],
    times: np.ndarray,
    f0_local_st: np.ndarray,
) -> Tuple[List[int], str]:
    """返回每个汉字的调类 (1=阴平, 2=阳平, 3=上声, 4=去声, 5=轻声)。"""
    chars = [t.get("char", "") for t in tokens]
    try:
        from pypinyin import Style, pinyin  # type: ignore

        tones: List[int] = []
        for ch in chars:
            if not ch:
                tones.append(5)
                continue
            py = pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True)
            if py and py[0] and py[0][0] and py[0][0][-1].isdigit():
                tones.append(int(py[0][0][-1]))
            else:
                tones.append(5)
        return tones, "pypinyin"
    except Exception:
        pass

    # 纯声学轮廓回退分类（基于去宏观语势后的字内 F0 斜率与均值位置）
    tones = []
    for tok in tokens:
        ch = tok.get("char", "")
        if ch in WEAK_FUNCTION_CHARS:
            tones.append(5)
            continue
        s_sec = tok["start_ms"] / 1000.0
        e_sec = tok["end_ms"] / 1000.0
        m = (times >= s_sec) & (times <= e_sec) & (~np.isnan(f0_local_st))
        seg = f0_local_st[m]
        if len(seg) < 3:
            tones.append(1)
            continue
        x = np.linspace(-1.0, 1.0, len(seg))
        slope = float(np.polyfit(x, seg, 1)[0])
        mean_v = float(np.mean(seg))
        if slope > 1.2:
            tones.append(2)  # 阳平上扬
        elif slope < -1.5:
            tones.append(4)  # 去声全降
        elif mean_v < -1.2:
            tones.append(3)  # 上声低拐
        else:
            tones.append(1)  # 阴平高平
    return tones, "acoustic-contour-fallback"


def _safe_zscore(arr: np.ndarray, clip: float = 3.0) -> np.ndarray:
    if len(arr) <= 1:
        return np.zeros_like(arr, dtype=np.float64)
    std = float(np.std(arr))
    if std < 1e-6:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - float(np.mean(arr))) / std, -clip, clip)


# ---------------------------------------------------------------------------
# 7. 主分析流水线：将 osot-annotator JSON + WAV 升级为三层解耦五维档案
# ---------------------------------------------------------------------------
def analyze_osot_recording(doc: Dict[str, Any], wav_path: str) -> Dict[str, Any]:
    y, sr = load_wav_mono(wav_path)
    duration_sec = len(y) / float(sr) if sr > 0 else float(doc.get("duration_sec", 0.0))
    tokens_in: List[Dict[str, Any]] = doc.get("tokens", [])
    if not tokens_in:
        raise ValueError("输入 JSON 中 tokens 为空，无法进行声学分析。")

    hop_sec = 0.005
    times, f0_hz, db_vals, hnr_vals, pitch_meta = extract_f0_hirst_twopass(y, sr, hop_sec=hop_sec)
    f0_st, voiced_segments = interpolate_f0_semitone_gap_safe(
        f0_hz, pitch_meta["speaker_f0_median_hz"], hop_sec=hop_sec, max_interp_ms=150.0
    )
    f0_local_st, f0_macro_st, cwt_prom = mexican_hat_cwt_decompose(
        f0_st, voiced_segments, hop_sec=hop_sec
    )
    tones, tone_backend = infer_mandarin_tones(tokens_in, times, f0_local_st)

    n = len(tokens_in)
    # 逐字基础物理量收集
    spans_st = np.zeros(n, dtype=np.float64)
    energies_db = np.zeros(n, dtype=np.float64)
    durations_ms = np.zeros(n, dtype=np.float64)
    gaps_ms = np.zeros(n, dtype=np.float64)
    cwt_scores = np.zeros(n, dtype=np.float64)
    macro_means = np.zeros(n, dtype=np.float64)
    vq_list: List[Dict[str, Any]] = []

    for i, tok in enumerate(tokens_in):
        s_ms = float(tok["start_ms"])
        e_ms = float(tok["end_ms"])
        s_sec, e_sec = s_ms / 1000.0, e_ms / 1000.0
        dur_ms = max(20.0, e_ms - s_ms)
        gap_ms = (
            max(0.0, float(tokens_in[i + 1]["start_ms"]) - e_ms) if i + 1 < n else 0.0
        )

        # 韵腹核心区 20%–85% 避开声母微韵律干扰
        core_s = s_sec + 0.15 * (e_sec - s_sec)
        core_e = s_sec + 0.85 * (e_sec - s_sec)
        m_full = (times >= s_sec) & (times <= e_sec)
        m_core = (times >= core_s) & (times <= core_e)
        if not np.any(m_full) and len(times) > 0:
            nearest = int(np.argmin(np.abs(times - 0.5 * (s_sec + e_sec))))
            m_full[nearest] = True
            m_core[nearest] = True

        seg_local = f0_local_st[m_core]
        seg_local_v = seg_local[~np.isnan(seg_local)]
        if len(seg_local_v) >= 2:
            span = float(np.percentile(seg_local_v, 95) - np.percentile(seg_local_v, 5))
        else:
            span = 0.0

        seg_macro = f0_macro_st[m_full]
        seg_macro_v = seg_macro[~np.isnan(seg_macro)]
        macro_m = float(np.mean(seg_macro_v)) if len(seg_macro_v) > 0 else 0.0

        seg_db = db_vals[m_full]
        eng = float(np.mean(seg_db)) if len(seg_db) > 0 else 0.0

        seg_cwt = cwt_prom[m_full]
        cwt_v = float(np.max(seg_cwt)) if len(seg_cwt) > 0 else 0.0

        seg_f0_hz = f0_hz[m_core]
        seg_f0_hz_v = seg_f0_hz[seg_f0_hz > 0]
        f0_char_hz = float(np.median(seg_f0_hz_v)) if len(seg_f0_hz_v) > 0 else 0.0

        seg_hnr = hnr_vals[m_core]
        hnr_v = float(np.mean(seg_hnr)) if len(seg_hnr) > 0 else 0.0

        spans_st[i] = span
        energies_db[i] = eng
        durations_ms[i] = dur_ms
        gaps_ms[i] = gap_ms
        cwt_scores[i] = cwt_v
        macro_means[i] = macro_m
        vq_list.append(extract_voice_quality_for_segment(y, sr, s_sec, e_sec, f0_char_hz, hnr_v))

    # --- A. 虚实声全篇说话人相对归一化判别（二维：频谱倾斜 × 周期性 CPP/HNR）---
    voiced_vq_idx = [i for i, v in enumerate(vq_list) if v["voice_quality"] != "Unvoiced"]
    if len(voiced_vq_idx) >= 3:
        cpp_arr = np.array([vq_list[i]["cpp_db"] for i in voiced_vq_idx], dtype=np.float64)
        hnr_arr = np.array([vq_list[i]["hnr_db"] for i in voiced_vq_idx], dtype=np.float64)
        tilt_arr = np.array(
            [
                0.5 * vq_list[i]["h1_h2_db"]
                + 0.3 * vq_list[i]["h1_a1_db"]
                + 0.2 * vq_list[i]["h1_a3_db"]
                for i in voiced_vq_idx
            ],
            dtype=np.float64,
        )
        z_cpp = _safe_zscore(cpp_arr)
        z_hnr = _safe_zscore(hnr_arr)
        z_tilt = _safe_zscore(tilt_arr)
        z_eng_all = _safe_zscore(energies_db)

        for k, idx in enumerate(voiced_vq_idx):
            # 气声/虚声指数：高倾斜 (+z_tilt) + 低周期性 (-z_cpp, -z_hnr)
            breathiness = float(0.45 * z_tilt[k] - 0.35 * z_cpp[k] - 0.20 * z_hnr[k])
            vq_list[idx]["breathiness_index"] = round(breathiness, 3)

            # 喉化/气泡音：极低 F0 (<-3st) + 低 HNR + 非高倾斜（偏紧压）
            is_low_f0 = macro_means[idx] < -2.2
            if is_low_f0 and z_hnr[k] < -0.9 and z_tilt[k] < 0.2:
                vq_list[idx]["voice_quality"] = "Creaky"
                vq_list[idx]["voice_quality_cn"] = "喉化/气泡音"
            elif breathiness >= 1.15:
                vq_list[idx]["voice_quality"] = "Breathy"
                vq_list[idx]["voice_quality_cn"] = "气声"
            elif breathiness >= 0.45 or (z_eng_all[idx] < -0.8 and z_tilt[k] > 0.2):
                vq_list[idx]["voice_quality"] = "Soft"
                vq_list[idx]["voice_quality_cn"] = "虚声/柔声"
            else:
                vq_list[idx]["voice_quality"] = "Modal"
                vq_list[idx]["voice_quality_cn"] = "实声"

    # --- B. 停连检测（区分 B0 紧密连读 / B1 虚连微气口 / B2 短语停连 / B3 句群长停）---
    pause_hypotheses: List[Dict[str, Any]] = []
    dsr_ratios = np.ones(n, dtype=np.float64)
    f0_resets = np.zeros(n, dtype=np.float64)
    for i in range(n):
        # 边界前音节延长比 DSR（相对前序 4 音节滑窗中位数）
        lo = max(0, i - 4)
        prev_durs = durations_ms[lo:i] if i > lo else durations_ms[max(0, i) : min(n, i + 3)]
        local_med_dur = float(np.median(prev_durs)) if len(prev_durs) > 0 else durations_ms[i]
        dsr = durations_ms[i] / max(local_med_dur, 40.0)
        dsr_ratios[i] = dsr

        # 跨边界宏观 F0 重置阶跃
        f0_reset = float(macro_means[i + 1] - macro_means[i]) if i + 1 < n else 0.0
        f0_resets[i] = f0_reset

        gap = gaps_ms[i]
        punct = tokens_in[i].get("punct_after", "")
        has_punct = any(p in PAUSE_PUNCT_SET for p in punct)
        is_strong_punct = any(p in STRONG_PUNCT_SET for p in punct)
        is_creaky_end = vq_list[i]["voice_quality"] == "Creaky"

        if i == n - 1:
            level, cn, reason = "B3", "句尾落停", "篇末收束"
        elif gap >= 350.0 or (gap >= 260.0 and is_strong_punct):
            level, cn, reason = "B3", "句群长停顿 (//)", f"物理静音 {gap:.0f}ms"
        elif gap >= 120.0:
            level, cn, reason = "B2", "短语实停 (/)", f"物理静音 {gap:.0f}ms"
        elif (
            (gap >= 45.0 and (dsr >= 1.25 or abs(f0_reset) >= 1.5))
            or (dsr >= 1.45 and abs(f0_reset) >= 1.2)
            or (is_creaky_end and abs(f0_reset) >= 1.3)
        ):
            level = "B1"
            cn = "虚连 / 微气口 (B1)"
            cues = []
            if gap >= 45.0:
                cues.append(f"微间隙 {gap:.0f}ms")
            if dsr >= 1.25:
                cues.append(f"边界前延长 {dsr:.2f}x")
            if abs(f0_reset) >= 1.2:
                cues.append(f"跨界F0重置 {f0_reset:+.1f}st")
            if is_creaky_end:
                cues.append("句末喉化线索")
            reason = " + ".join(cues)
        elif has_punct and gap < 45.0:
            level, cn, reason = "B0_CONNECT", "跨标点直连 (^)", f"标点处无停顿 ({gap:.0f}ms)"
        else:
            level, cn, reason = "B0", "自然连读", "无边界线索"

        pause_hypotheses.append(
            {
                "level": level,
                "label_cn": cn,
                "gap_ms": round(gap, 1),
                "dsr_lengthening_ratio": round(dsr, 3),
                "f0_reset_st": round(f0_reset, 2),
                "reason": reason,
            }
        )

    # --- C. 切分韵律短语并拟合三阶正交勒让德「连续语势」与滑窗语速节奏 ---
    phrases: List[Dict[str, Any]] = []
    p_start = 0
    for i in range(n):
        lvl = pause_hypotheses[i]["level"]
        is_cut = (lvl in ("B2", "B3")) or (i == n - 1)
        if is_cut:
            p_end = i
            t_s = tokens_in[p_start]["start_ms"] / 1000.0
            t_e = tokens_in[p_end]["end_ms"] / 1000.0
            m_p = (times >= t_s) & (times <= t_e)
            leg = fit_legendre_yushi(f0_macro_st[m_p])

            # 节奏：局部语速 (音节/秒) 与时长变异系数 CV
            p_durs = durations_ms[p_start : p_end + 1]
            span_sec = max(0.08, t_e - t_s)
            lsr = (p_end - p_start + 1) / span_sec
            cv = float(np.std(p_durs) / max(np.mean(p_durs), 1.0)) if len(p_durs) >= 2 else 0.0
            if lsr >= 4.6:
                rhythm_mode, rhythm_cn = "Fast", "紧凑/轻快"
            elif lsr <= 2.9:
                rhythm_mode, rhythm_cn = "Slow", "舒缓/凝重"
            else:
                rhythm_mode, rhythm_cn = "Steady", "匀速稳重"

            text_str = "".join(
                tokens_in[k].get("char", "") + tokens_in[k].get("punct_after", "")
                for k in range(p_start, p_end + 1)
            )
            phrases.append(
                {
                    "phrase_id": len(phrases) + 1,
                    "token_start": p_start,
                    "token_end": p_end,
                    "start_ms": int(tokens_in[p_start]["start_ms"]),
                    "end_ms": int(tokens_in[p_end]["end_ms"]),
                    "text": text_str,
                    "legendre_yushi": leg,
                    "rhythm": {
                        "local_speech_rate_syll_per_sec": round(lsr, 2),
                        "duration_cv": round(cv, 3),
                        "mode": rhythm_mode,
                        "mode_cn": rhythm_cn,
                    },
                }
            )
            p_start = i + 1

    # --- D. 普通话同调型内部归一化重音 + 焦点后压缩 (PFC) + 白箱归因 ---
    z_span_tone = np.zeros(n, dtype=np.float64)
    tone_arr = np.array(tones, dtype=int)
    for tone_cls in (1, 2, 3, 4, 5):
        idx_cls = np.where(tone_arr == tone_cls)[0]
        if len(idx_cls) >= 3:
            z_span_tone[idx_cls] = _safe_zscore(spans_st[idx_cls])
        elif len(idx_cls) > 0:
            # 样本过少时退回全篇均值归一化
            z_span_tone[idx_cls] = _safe_zscore(spans_st)[idx_cls]

    z_eng = _safe_zscore(energies_db)
    z_dur = _safe_zscore(durations_ms)
    z_cwt = _safe_zscore(cwt_scores)

    # 焦点后压缩 (PFC)：若当前字后面 1~2 字在同一短语内音域与能量显著收缩，提升当前字重音置信度
    pfc_bonus = np.zeros(n, dtype=np.float64)
    for i in range(n - 1):
        if pause_hypotheses[i]["level"] in ("B2", "B3"):
            continue
        post_hi = min(n, i + 3)
        post_eng = float(np.mean(z_eng[i + 1 : post_hi]))
        post_span = float(np.mean(z_span_tone[i + 1 : post_hi]))
        drop = (z_eng[i] - post_eng) * 0.5 + (z_span_tone[i] - post_span) * 0.5
        if drop > 0.8:
            pfc_bonus[i] = min(0.8, 0.35 * (drop - 0.5))

    # 透明可解释逻辑斯蒂重音打分 + 白箱特征归因
    tier1_tokens: List[Dict[str, Any]] = []
    tier2_tokens: List[Dict[str, Any]] = []
    tier3_tokens: List[Dict[str, Any]] = []

    for i, tok in enumerate(tokens_in):
        ch = tok.get("char", "")
        lex_prior = -0.65 if ch in WEAK_FUNCTION_CHARS else (0.25 if ch in CONTENT_BOOST_CHARS else 0.0)

        c_span = 0.28 * float(z_span_tone[i])
        c_eng = 0.30 * float(z_eng[i])
        c_dur = 0.24 * float(z_dur[i])
        c_cwt = 0.18 * float(z_cwt[i])
        c_pfc = 0.20 * float(pfc_bonus[i])
        logit = c_span + c_eng + c_dur + c_cwt + c_pfc + lex_prior - 0.35
        prob = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, logit))))

        if prob >= 0.72:
            stress_lvl, stress_cn = "S2", "核心主重音"
        elif prob >= 0.52:
            stress_lvl, stress_cn = "S1", "常规次重音"
        else:
            stress_lvl, stress_cn = "S0", "非重音"

        # 播音学重音三大表达法自动提示
        vq_tag = vq_list[i]["voice_quality"]
        if stress_lvl != "S0":
            if vq_tag in ("Soft", "Breathy") and z_eng[i] < 0.2:
                method_hint = "虚实转换法（轻虚凸显）"
            elif z_dur[i] >= max(z_eng[i], z_span_tone[i]) and z_dur[i] >= 0.7:
                method_hint = "快慢停连法（拖音拉长）"
            else:
                method_hint = "高低强弱法（音高/响度凸显）"
        else:
            method_hint = "无"

        is_elong = bool(z_dur[i] >= 1.35 or durations_ms[i] >= 480.0)

        tier1_tokens.append(
            {
                "index": i,
                "char": ch,
                "punct_after": tok.get("punct_after", ""),
                "start_ms": int(tok["start_ms"]),
                "end_ms": int(tok["end_ms"]),
                "duration_ms": int(round(durations_ms[i])),
                "gap_after_ms": int(round(gaps_ms[i])),
                "mandarin_tone": int(tones[i]),
                "energy_db": round(float(energies_db[i]), 2),
                "f0_local_span_st": round(float(spans_st[i]), 3),
                "f0_macro_mean_st": round(float(macro_means[i]), 3),
                "cwt_prominence_raw": round(float(cwt_scores[i]), 3),
                "dsr_lengthening_ratio": round(float(dsr_ratios[i]), 3),
                "f0_reset_after_st": round(float(f0_resets[i]), 2),
                "voice_quality_metrics": {
                    "cpp_db": vq_list[i]["cpp_db"],
                    "h1_h2_db": vq_list[i]["h1_h2_db"],
                    "h1_a1_db": vq_list[i]["h1_a1_db"],
                    "h1_a3_db": vq_list[i]["h1_a3_db"],
                    "hnr_db": vq_list[i]["hnr_db"],
                    "breathiness_index": vq_list[i]["breathiness_index"],
                },
            }
        )

        tier2_tokens.append(
            {
                "index": i,
                "char": ch,
                "candidate_stress": stress_lvl,
                "candidate_stress_cn": stress_cn,
                "stress_prob": round(prob, 3),
                "stress_method_hint": method_hint,
                "is_elongated": is_elong,
                "feature_attribution": {
                    "tone_norm_f0_span": round(c_span, 3),
                    "energy_lift": round(c_eng, 3),
                    "duration_elongation": round(c_dur, 3),
                    "cwt_multiscale": round(c_cwt, 3),
                    "post_focus_compression": round(c_pfc, 3),
                    "lexical_prior": round(lex_prior, 3),
                },
                "candidate_pause": pause_hypotheses[i],
                "candidate_voice_quality": vq_list[i]["voice_quality"],
                "candidate_voice_quality_cn": vq_list[i]["voice_quality_cn"],
            }
        )

        # 保留 osot-annotator 中已有的人工标注结果作为第三层初始态
        old_stress = tok.get("stress", "none")
        old_pause = tok.get("pause_after", "none")
        tier3_tokens.append(
            {
                "index": i,
                "char": ch,
                "human_stress": {"primary": "S2", "secondary": "S1"}.get(old_stress, "UNVERIFIED"),
                "human_stress_semantic_type": "",  # 留待专家填写：Logic(逻辑重音) / Emotion(感情重音)
                "human_pause": {"long": "B3", "short": "B2", "connect": "B0_CONNECT"}.get(
                    old_pause, "UNVERIFIED"
                ),
                "human_voice_quality": "UNVERIFIED",
                "knowledge_tags": list(tok.get("knowledge_tags", [])),
                "audit_action": "IMPORTED_FROM_MVP" if (old_stress != "none" or old_pause != "none") else "PENDING",
            }
        )

    # 下采样全局曲线供可视化（最多 1200 点）
    def _ds(arr: np.ndarray, max_pts: int = 1200) -> List[Optional[float]]:
        if len(arr) == 0:
            return []
        idx = np.linspace(0, len(arr) - 1, min(len(arr), max_pts)).astype(int)
        return [None if np.isnan(arr[j]) else round(float(arr[j]), 2) for j in idx]

    result = {
        "schema_version": "osot-3tier-prosody-v1.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_id": doc.get("id", Path(wav_path).stem),
        "title": doc.get("title", Path(wav_path).stem),
        "duration_sec": round(duration_sec, 3),
        "analysis_meta": {
            **pitch_meta,
            "tone_backend": tone_backend,
            "total_tokens": n,
            "total_phrases": len(phrases),
        },
        "tier1_objective_acoustic": {
            "immutable_notice": "客观物理测量层：严禁人工直接篡改，仅在边界时间戳修正后由脚本重算。",
            "curves": {
                "times_sec": _ds(times, 800),
                "f0_raw_st": _ds(f0_st, 800),
                "f0_local_st": _ds(f0_local_st, 800),
                "f0_macro_st": _ds(f0_macro_st, 800),
                "energy_db": _ds(db_vals, 800),
            },
            "tokens": tier1_tokens,
        },
        "tier2_algorithmic_hypothesis": {
            "tokens": tier2_tokens,
            "phrases": phrases,
        },
        "tier3_human_gold": {
            "annotator_id": "",
            "review_status": "pending_expert_review",
            "tokens": tier3_tokens,
        },
    }
    return result


# ---------------------------------------------------------------------------
# 8. 导出 6 层 Praat TextGrid & 交互式 HTML 五维可视化声学报告
# ---------------------------------------------------------------------------
def export_6tier_textgrid(res: Dict[str, Any]) -> str:
    """导出 6 层专业播音 Praat TextGrid：chars / pause_B0_B3 / stress_S0_S2 / yushi_legendre / voice_quality / rhythm"""
    duration = max(0.1, float(res.get("duration_sec", 0.1)))
    t1 = res["tier1_objective_acoustic"]["tokens"]
    t2 = res["tier2_algorithmic_hypothesis"]["tokens"]
    phrases = res["tier2_algorithmic_hypothesis"]["phrases"]

    def build_intervals(items: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
        out: List[Tuple[float, float, str]] = []
        cursor = 0.0
        for xmin, xmax, label in items:
            xmin = max(cursor, min(xmin, duration))
            xmax = max(xmin, min(xmax, duration))
            if xmin - cursor > 1e-4:
                out.append((cursor, xmin, ""))
            if xmax - xmin > 1e-4:
                out.append((xmin, xmax, label))
                cursor = xmax
        if duration - cursor > 1e-4:
            out.append((cursor, duration, ""))
        return out or [(0.0, duration, "")]

    chars_it, pause_it, stress_it, vq_it, yushi_it, rhythm_it = [], [], [], [], [], []
    for i, (a, h) in enumerate(zip(t1, t2)):
        s, e = a["start_ms"] / 1000.0, a["end_ms"] / 1000.0
        chars_it.append((s, e, f"{a['char']}(T{a['mandarin_tone']})"))
        if h["candidate_stress"] != "S0":
            stress_it.append((s, e, f"{h['candidate_stress']}:{h['stress_prob']:.2f}"))
        if h["candidate_voice_quality"] not in ("Modal", "Unvoiced"):
            vq_it.append((s, e, f"{h['candidate_voice_quality']}({a['voice_quality_metrics']['breathiness_index']:+.1f})"))
        p_lvl = h["candidate_pause"]["level"]
        if p_lvl != "B0":
            nxt_s = t1[i + 1]["start_ms"] / 1000.0 if i + 1 < len(t1) else min(duration, e + 0.08)
            p_end = nxt_s if nxt_s > e + 0.01 else min(duration, e + 0.04)
            pause_it.append((e, p_end, p_lvl))

    for ph in phrases:
        ps, pe = ph["start_ms"] / 1000.0, ph["end_ms"] / 1000.0
        leg = ph["legendre_yushi"]
        yushi_it.append(
            (ps, pe, f"{leg['yushi_label']}[c1={leg['c1_slope']:+.2f},c2={leg['c2_curvature']:+.2f}]")
        )
        rh = ph["rhythm"]
        rhythm_it.append((ps, pe, f"{rh['mode']}({rh['local_speech_rate_syll_per_sec']}syl/s)"))

    tiers = [
        ("chars_tone", build_intervals(chars_it)),
        ("pause_B0_B3", build_intervals(pause_it)),
        ("stress_S0_S2", build_intervals(stress_it)),
        ("yushi_legendre", build_intervals(yushi_it)),
        ("voice_quality", build_intervals(vq_it)),
        ("rhythm_LSR", build_intervals(rhythm_it)),
    ]

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {duration:.6f}",
        "tiers? <exists>",
        f"size = {len(tiers)}",
        "item []:",
    ]
    for idx, (name, ivs) in enumerate(tiers, 1):
        lines.extend(
            [
                f"    item [{idx}]:",
                '        class = "IntervalTier"',
                f'        name = "{name}"',
                "        xmin = 0",
                f"        xmax = {duration:.6f}",
                f"        intervals: size = {len(ivs)}",
            ]
        )
        for j, (xmin, xmax, txt) in enumerate(ivs, 1):
            esc = txt.replace('"', '""')
            lines.extend(
                [
                    f"        intervals [{j}]:",
                    f"            xmin = {xmin:.6f}",
                    f"            xmax = {xmax:.6f}",
                    f'            text = "{esc}"',
                ]
            )
    return "\n".join(lines) + "\n"


def export_interactive_html_report(res: Dict[str, Any]) -> str:
    """生成零外部 CDN 依赖的交互式五维播音声学可视化审校报告（HTML）。"""
    payload_json = json.dumps(res, ensure_ascii=False)
    title = res.get("title", "OSOT 五维播音声学报告")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — OSOT 五维声学剖析</title>
<style>
  :root {{ --bg:#0f172a; --card:#1e293b; --line:#334155; --txt:#f1f5f9; --muted:#94a3b8;
           --red:#ef4444; --amber:#f59e0b; --blue:#38bdf8; --green:#10b981; --purple:#a78bfa; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:"PingFang SC","Microsoft YaHei",system-ui,sans-serif; background:var(--bg); color:var(--txt); padding:24px; line-height:1.6; }}
  .wrap {{ max-width:1180px; margin:0 auto; }}
  h1 {{ font-size:22px; margin:0 0 6px; }}
  .sub {{ color:var(--muted); font-size:13px; margin-bottom:20px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-bottom:20px; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
  .kpi .k {{ font-size:12px; color:var(--muted); }}
  .kpi .v {{ font-size:22px; font-weight:700; margin-top:4px; }}
  .panel {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px; margin-bottom:18px; }}
  .panel h2 {{ font-size:16px; margin:0 0 12px; display:flex; justify-content:space-between; align-items:center; }}
  canvas {{ width:100%; height:240px; background:#0b1220; border-radius:8px; display:block; }}
  .script {{ font-size:20px; line-height:2.4; display:flex; flex-wrap:wrap; align-items:baseline; gap:2px; }}
  .tok {{ display:inline-flex; flex-direction:column; align-items:center; padding:2px 6px; border-radius:6px; cursor:pointer; border:1px solid transparent; transition:all .15s; }}
  .tok:hover {{ border-color:var(--blue); background:#1e3a5f55; }}
  .tok.active {{ border-color:#fff; box-shadow:0 0 0 2px #38bdf866; }}
  .tok.S2 {{ background:#7f1d1d; color:#fecaca; font-weight:700; }}
  .tok.S1 {{ background:#713f12; color:#fef08a; font-weight:600; }}
  .vq-dot {{ width:6px; height:6px; border-radius:50%; margin-top:2px; }}
  .vq-Breathy {{ background:#38bdf8; }} .vq-Soft {{ background:#a78bfa; }} .vq-Creaky {{ background:#fb923c; }} .vq-Modal {{ background:transparent; }}
  .pause-tag {{ font-size:13px; font-weight:700; padding:0 5px; border-radius:4px; margin:0 2px; }}
  .pause-B3 {{ color:#f87171; }} .pause-B2 {{ color:#38bdf8; }} .pause-B1 {{ color:#c084fc; background:#581c8744; }} .pause-B0_CONNECT {{ color:#34d399; }}
  .inspector {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .bar-row {{ display:flex; align-items:center; gap:8px; margin:6px 0; font-size:13px; }}
  .bar-lbl {{ width:130px; color:var(--muted); text-align:right; }}
  .bar-track {{ flex:1; height:10px; background:#0f172a; border-radius:5px; overflow:hidden; position:relative; }}
  .bar-fill {{ height:100%; border-radius:5px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; }}
  th {{ color:var(--muted); font-weight:600; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>OSOT 五维播音声学剖析报告：{title}</h1>
  <div class="sub" id="metaLine"></div>
  <div class="grid" id="kpiGrid"></div>

  <div class="panel">
    <h2><span>CWT 声调-语势剥离曲线（蓝色：原始 F0 半音 ｜ 金色加粗：宏观语势包络 F0_macro）</span></h2>
    <canvas id="f0Canvas" width="1120" height="240"></canvas>
  </div>

  <div class="panel">
    <h2><span>交互式五维播音稿（点击任意汉字查看白箱声学归因与虚实声指标）</span>
      <span style="font-size:12px;color:var(--muted)">红=S2主重音 黄=S1次重音 紫底B1=虚连微气口 蓝点=气声 紫点=虚声 橙点=喉化</span>
    </h2>
    <div class="script" id="scriptBox"></div>
  </div>

  <div class="panel" id="inspectorPanel">
    <h2><span id="inspTitle">点击上方任意汉字查看白箱声学归因</span></h2>
    <div class="inspector" id="inspBody"><div style="color:var(--muted)">等待选字…</div></div>
  </div>

  <div class="panel">
    <h2><span>短语级三阶勒让德连续语势 (c0, c1, c2, c3) 与节奏分析表</span></h2>
    <table>
      <thead><tr><th>#</th><th>韵律短语文本</th><th>播音语势形态</th><th>c1 斜率</th><th>c2 拱度</th><th>c3 转折</th><th>拟合 r</th><th>语速 (字/秒)</th><th>节奏</th></tr></thead>
      <tbody id="phraseTable"></tbody>
    </table>
  </div>
</div>
<script>
const DATA = {payload_json};
const t1 = DATA.tier1_objective_acoustic.tokens;
const t2 = DATA.tier2_algorithmic_hypothesis.tokens;
const phrases = DATA.tier2_algorithmic_hypothesis.phrases;
const meta = DATA.analysis_meta;

document.getElementById("metaLine").textContent =
  `时长 ${{DATA.duration_sec}}s ｜ 说话人基频中位数 ${{meta.speaker_f0_median_hz}} Hz (Hirst音域 ${{meta.hirst_floor_hz}}–${{meta.hirst_ceiling_hz}} Hz) ｜ 基频引擎: ${{meta.pitch_backend}} ｜ 声调引擎: ${{meta.tone_backend}}`;

const s2Cnt = t2.filter(x => x.candidate_stress === "S2").length;
const s1Cnt = t2.filter(x => x.candidate_stress === "S1").length;
const b1Cnt = t2.filter(x => x.candidate_pause.level === "B1").length;
const nonModalCnt = t2.filter(x => !["Modal","Unvoiced"].includes(x.candidate_voice_quality)).length;

const kpis = [
  ["总汉字 / 韵律短语", `${{t1.length}} 字 / ${{phrases.length}} 段`],
  ["主重音 S2 / 次重音 S1", `${{s2Cnt}} / ${{s1Cnt}}`],
  ["检出「虚连/微气口 B1」", `${{b1Cnt}} 处`],
  ["非实声发声态 (气/虚/喉化)", `${{nonModalCnt}} 字`],
];
document.getElementById("kpiGrid").innerHTML = kpis.map(([k,v]) =>
  `<div class="kpi"><div class="k">${{k}}</div><div class="v">${{v}}</div></div>`).join("");

// 绘制 F0_raw vs F0_macro 曲线
(function drawF0(){{
  const cv = document.getElementById("f0Canvas");
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  const raw = DATA.tier1_objective_acoustic.curves.f0_raw_st;
  const mac = DATA.tier1_objective_acoustic.curves.f0_macro_st;
  const vals = raw.filter(v => v !== null);
  const ymin = Math.min(-6, ...(vals.length ? vals : [-6]));
  const ymax = Math.max(6, ...(vals.length ? vals : [6]));
  const mapY = v => H - 24 - ((v - ymin) / (ymax - ymin || 1)) * (H - 44);
  const mapX = i => 42 + (i / Math.max(1, raw.length - 1)) * (W - 58);

  ctx.strokeStyle = "#1e293b"; ctx.lineWidth = 1;
  for (let st = Math.ceil(ymin/3)*3; st <= ymax; st += 3) {{
    const y = mapY(st);
    ctx.beginPath(); ctx.moveTo(42, y); ctx.lineTo(W-16, y); ctx.stroke();
    ctx.fillStyle = "#64748b"; ctx.font = "11px monospace";
    ctx.fillText((st>=0?"+":"")+st+"st", 4, y+4);
  }}
  function plotLine(arr, color, lw) {{
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.beginPath();
    let started = false;
    arr.forEach((v, i) => {{
      if (v === null) {{ started = false; return; }}
      const x = mapX(i), y = mapY(v);
      if (!started) {{ ctx.moveTo(x, y); started = true; }} else ctx.lineTo(x, y);
    }});
    ctx.stroke();
  }}
  plotLine(raw, "#38bdf888", 1.5);
  plotLine(mac, "#f59e0b", 3.0);
}})();

// 渲染播音稿
const box = document.getElementById("scriptBox");
t1.forEach((a, i) => {{
  const h = t2[i];
  const sp = document.createElement("span");
  sp.className = `tok ${{h.candidate_stress}}`;
  sp.innerHTML = `<span>${{a.char}}</span><span class="vq-dot vq-${{h.candidate_voice_quality}}"></span>`;
  sp.onclick = () => selectTok(i, sp);
  box.appendChild(sp);
  if (a.punct_after) {{
    const p = document.createElement("span"); p.textContent = a.punct_after; box.appendChild(p);
  }}
  const pl = h.candidate_pause.level;
  if (pl !== "B0") {{
    const sym = {{B3:"//", B2:"/", B1:"┆B1", B0_CONNECT:"^"}}[pl] || pl;
    const pt = document.createElement("span");
    pt.className = `pause-tag pause-${{pl}}`;
    pt.title = h.candidate_pause.reason;
    pt.textContent = sym;
    box.appendChild(pt);
  }}
}});

function selectTok(i, el) {{
  document.querySelectorAll(".tok.active").forEach(x => x.classList.remove("active"));
  el.classList.add("active");
  const a = t1[i], h = t2[i], attr = h.feature_attribution, vq = a.voice_quality_metrics;
  document.getElementById("inspTitle").textContent =
    `单字「${{a.char}}」(#${{i+1}} · ${{a.start_ms}}–${{a.end_ms}}ms · 第${{a.mandarin_tone}}声) — 重音候选: ${{h.candidate_stress_cn}} (p=${{h.stress_prob}})`;
  const rows = [
    ["同调型音域跨度", attr.tone_norm_f0_span, "#38bdf8"],
    ["短语相对能量抬升", attr.energy_lift, "#f59e0b"],
    ["音节时长延长", attr.duration_elongation, "#10b981"],
    ["CWT 多尺度凸显", attr.cwt_multiscale, "#a78bfa"],
    ["焦点后压缩 PFC", attr.post_focus_compression, "#f43f5e"],
    ["词性先验修正", attr.lexical_prior, "#94a3b8"],
  ];
  const barsHtml = rows.map(([lbl, val, col]) => {{
    const pct = Math.min(100, Math.abs(val) * 100);
    return `<div class="bar-row"><div class="bar-lbl">${{lbl}}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${{pct}}%;background:${{val>=0?col:'#64748b'}}"></div></div>
      <div style="width:52px;font-family:monospace">${{val>=0?"+":""}}${{val.toFixed(2)}}</div></div>`;
  }}).join("");

  document.getElementById("inspBody").innerHTML = `
    <div>
      <div style="font-weight:600;margin-bottom:8px">白箱重音归因向量（推荐表达法：${{h.stress_method_hint}}）</div>
      ${{barsHtml}}
    </div>
    <div style="font-size:13px;line-height:1.9">
      <div><b>发声态（虚实声）：</b>${{h.candidate_voice_quality_cn}} (${{h.candidate_voice_quality}}) ｜ 气声指数 <code>${{vq.breathiness_index}}</code></div>
      <div><b>声门频谱参量：</b>CPP=<code>${{vq.cpp_db}}</code> dB ｜ H1-H2=<code>${{vq.h1_h2_db}}</code> dB ｜ H1-A1=<code>${{vq.h1_a1_db}}</code> dB ｜ H1-A3=<code>${{vq.h1_a3_db}}</code> dB ｜ HNR=<code>${{vq.hnr_db}}</code> dB</div>
      <div><b>停连判别：</b>${{h.candidate_pause.label_cn}}（${{h.candidate_pause.reason}}）</div>
      <div><b>物理时长与延长：</b>时长 <code>${{a.duration_ms}}ms</code> ｜ 边界前延长比 DSR=<code>${{a.dsr_lengthening_ratio}}x</code> ｜ 跨界 F0 重置=<code>${{a.f0_reset_after_st}}st</code></div>
    </div>`;
}}

document.getElementById("phraseTable").innerHTML = phrases.map(p => {{
  const l = p.legendre_yushi, r = p.rhythm;
  return `<tr><td>${{p.phrase_id}}</td><td>${{p.text}}</td>
    <td><span class="badge" style="background:#1e3a5f;color:#7dd3fc">${{l.yushi_cn}}</span></td>
    <td>${{l.c1_slope}}</td><td>${{l.c2_curvature}}</td><td>${{l.c3_inflection}}</td><td>${{l.pearson_r}}</td>
    <td>${{r.local_speech_rate_syll_per_sec}}</td><td>${{r.mode_cn}}</td></tr>`;
}}).join("");

if (t1.length > 0) document.querySelector(".tok").click();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 9. 内置 Demo 合成器（生成带真实语势/四声/虚实声/虚连的测试录音 + JSON）
# ---------------------------------------------------------------------------
def build_demo_case(out_dir: Path) -> Tuple[Path, Path]:
    """合成一段 4.5 秒中文播音测试波形（含上山/波峰语势、气声虚声段与 B1 虚连微气口）。"""
    sr = 16000
    script = [
        ("让", "", 190, 20, 4, 0.85, "Modal", -1.5),
        ("暴", "", 220, 15, 4, 1.00, "Modal", -0.5),
        ("风", "", 210, 10, 1, 0.90, "Modal", +0.5),
        ("雨", "", 290, 65, 3, 1.10, "Modal", +1.2),  # 边界前延长 + B1 虚连
        ("来", "", 180, 15, 2, 0.75, "Soft", +0.8),
        ("得", "", 110, 10, 5, 0.55, "Soft", +0.6),
        ("更", "", 270, 20, 4, 1.25, "Modal", +2.5),  # 核心主重音 + 上山高潮
        ("猛", "", 250, 15, 3, 1.15, "Modal", +3.0),
        ("烈", "", 260, 15, 4, 1.10, "Modal", +3.4),
        ("些", "", 180, 10, 1, 0.70, "Breathy", +2.2),
        ("吧", "！", 320, 420, 5, 0.50, "Breathy", +0.5),  # 长停 //
        ("夜", "", 240, 20, 4, 0.65, "Breathy", -0.8),  # 气声虚声起句
        ("色", "", 210, 15, 4, 0.60, "Soft", -0.4),
        ("渐", "", 260, 20, 4, 0.85, "Modal", +1.4),  # 波峰中心
        ("深", "。", 340, 200, 1, 0.55, "Creaky", -2.8),  # 句尾喉化落停
    ]

    rng = np.random.default_rng(42)
    chunks: List[np.ndarray] = []
    tokens: List[Dict[str, Any]] = []
    cursor_ms = 80
    chunks.append(np.zeros(int(0.08 * sr), dtype=np.float64))

    base_f0 = 175.0
    for idx, (ch, punct, dur_ms, gap_ms, tone, amp, vq, macro_st) in enumerate(script):
        n_s = int(sr * dur_ms / 1000.0)
        t = np.arange(n_s, dtype=np.float64) / sr
        u = np.linspace(0.0, 1.0, n_s)

        # 叠加普通话四声轮廓 + 宏观语势偏置
        if tone == 1:
            tone_st = np.full_like(u, 1.5)
        elif tone == 2:
            tone_st = -1.0 + 3.2 * u
        elif tone == 3:
            tone_st = -1.5 - 2.0 * np.sin(np.pi * u)
        elif tone == 4:
            tone_st = 3.0 - 5.2 * u
        else:
            tone_st = np.full_like(u, -0.5)

        f0_t = base_f0 * (2.0 ** ((macro_st + tone_st) / 12.0))
        phase = 2.0 * np.pi * np.cumsum(f0_t) / sr

        # 根据发声态合成不同谐波斜率与气声噪声比例
        if vq == "Breathy":
            sig = 1.0 * np.sin(phase) + 0.12 * np.sin(2 * phase) + 0.04 * np.sin(3 * phase)
            sig += 0.35 * rng.normal(0, 1, n_s)
        elif vq == "Soft":
            sig = 0.9 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.08 * np.sin(3 * phase)
            sig += 0.14 * rng.normal(0, 1, n_s)
        elif vq == "Creaky":
            sub_mod = 0.55 + 0.45 * np.sin(0.5 * phase)
            sig = (0.7 * np.sin(phase) + 0.6 * np.sin(2 * phase) + 0.45 * np.sin(3 * phase)) * sub_mod
            sig += 0.18 * rng.normal(0, 1, n_s)
        else:
            sig = (
                0.75 * np.sin(phase)
                + 0.65 * np.sin(2 * phase)
                + 0.50 * np.sin(3 * phase)
                + 0.35 * np.sin(4 * phase)
                + 0.20 * np.sin(5 * phase)
            )
            sig += 0.02 * rng.normal(0, 1, n_s)

        env = np.sin(np.pi * np.clip(u, 0, 1)) ** 0.35
        sig = sig / (np.max(np.abs(sig)) + 1e-6) * amp * 0.7 * env
        chunks.append(sig)

        tokens.append(
            {
                "index": idx,
                "char": ch,
                "punct_after": punct,
                "start_ms": cursor_ms,
                "end_ms": cursor_ms + dur_ms,
                "stress": "none",
                "pause_after": "none",
                "knowledge_tags": [],
            }
        )
        cursor_ms += dur_ms
        if gap_ms > 0:
            chunks.append(rng.normal(0, 0.001, int(sr * gap_ms / 1000.0)))
            cursor_ms += gap_ms

    y_all = np.concatenate(chunks)
    wav_out = out_dir / "demo_osot_broadcast.wav"
    json_out = out_dir / "demo_osot_mvp.json"
    save_wav_mono_16bit(str(wav_out), y_all, sr)
    doc = {
        "id": "demo_osot_broadcast",
        "title": "OSOT 内置演示：上山语势与气声虚实对照",
        "duration_sec": round(len(y_all) / sr, 3),
        "tokens": tokens,
    }
    json_out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_out, wav_out


# ---------------------------------------------------------------------------
# 10. CLI 入口
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="OSOT 五维播音声学深度剖析与三层解耦升级引擎"
    )
    parser.add_argument("--json", dest="json_path", help="osot-annotator 导出的 .json 文件路径")
    parser.add_argument("--wav", dest="wav_path", help="对应的纯人声 .wav 文件路径（16kHz 推荐）")
    parser.add_argument("--out-dir", dest="out_dir", default="osot_analysis_out", help="输出目录")
    parser.add_argument("--demo", action="store_true", help="运行内置播音合成样例自测并生成全套输出")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        json_path, wav_path = build_demo_case(out_dir)
        print(f"[DEMO] 已生成测试样例: {json_path.name} + {wav_path.name}")
    else:
        if not args.json_path or not args.wav_path:
            parser.error("请提供 --json 与 --wav 路径，或使用 --demo 运行内置自测。")
        json_path = Path(args.json_path)
        wav_path = Path(args.wav_path)

    doc = json.loads(json_path.read_text(encoding="utf-8"))
    result = analyze_osot_recording(doc, str(wav_path))

    stem = json_path.stem
    out_json = out_dir / f"{stem}_osot_3tier.json"
    out_tg = out_dir / f"{stem}_osot_6tier.TextGrid"
    out_html = out_dir / f"{stem}_osot_report.html"

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    out_tg.write_text(export_6tier_textgrid(result), encoding="utf-8")
    out_html.write_text(export_interactive_html_report(result), encoding="utf-8")

    phrases = result["tier2_algorithmic_hypothesis"]["phrases"]
    t2 = result["tier2_algorithmic_hypothesis"]["tokens"]
    print("=== OSOT 五维声学分析完成 ===")
    print(f"三层解耦 JSON : {out_json}")
    print(f"6层 TextGrid  : {out_tg}")
    print(f"交互可视化HTML: {out_html}")
    print(
        f"摘要: 共 {len(t2)} 字 / {len(phrases)} 个韵律短语 | "
        f"S2主重音={sum(1 for x in t2 if x['candidate_stress']=='S2')} | "
        f"S1次重音={sum(1 for x in t2 if x['candidate_stress']=='S1')} | "
        f"B1虚连={sum(1 for x in t2 if x['candidate_pause']['level']=='B1')} | "
        f"非实声={sum(1 for x in t2 if x['candidate_voice_quality'] not in ('Modal','Unvoiced'))}"
    )
    for p in phrases:
        leg = p["legendre_yushi"]
        print(
            f"  短语#{p['phrase_id']} 「{p['text']}」 -> {leg['yushi_cn']} "
            f"(c1={leg['c1_slope']:+.2f}, c2={leg['c2_curvature']:+.2f}, r={leg['pearson_r']:.2f})"
        )


if __name__ == "__main__":
    main()
