@echo off
chcp 65001 >nul
echo ====================================================
echo   OSOT 工具 01：五维播音声学分析与三层解耦升级引擎
echo ====================================================
echo.
set "PY_BIN=D:\otos\osot-annotator\.venv\Scripts\python.exe"
if not exist "%PY_BIN%" set "PY_BIN=python"

set /p JSON_FILE=请拖入 osot-annotator 导出的 JSON 文件（直接回车运行 Demo）: 
if "%JSON_FILE%"=="" (
    "%PY_BIN%" "%~dp0osot_prosody_analyzer.py" --demo --out-dir "%~dp0demo_out"
    echo.
    echo 已完成 Demo 自测，正在打开可视化报告...
    start "" "%~dp0demo_out\demo_osot_mvp_osot_report.html"
    pause
    exit /b 0
)

set /p WAV_FILE=请拖入对应的纯人声 WAV 文件: 
"%PY_BIN%" "%~dp0osot_prosody_analyzer.py" --json %JSON_FILE% --wav %WAV_FILE% --out-dir "%~dp0output"
echo.
echo 分析完成！结果已保存至 %~dp0output
pause
