# OSOT 声学与韵律分析器

分析播音录音和对应标注，生成声学特征、三层标注数据及可视化报告，服务于 OSOT 播音表达研究。

## 快速体验

需要 Python、NumPy 和 SciPy。运行：

```bash
pip install numpy scipy
python osot_prosody_analyzer.py --demo --out-dir demo_out
```

分析真实数据：

```bash
python osot_prosody_analyzer.py --json 标注.json --wav 人声.wav --out-dir output
```

Windows 用户也可运行 `run_analyzer.bat`。仓库保留原版、v2 脚本和共用数据协议；启动脚本使用原版。
