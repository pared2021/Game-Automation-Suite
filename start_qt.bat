@echo off
echo 正在启动游戏自动化控制面板...

:: 激活虚拟环境
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo 创建虚拟环境...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo 安装依赖...
    pip install -r requirements.txt
    pip install PyQt6
)

:: 启动Qt GUI
python qt_gui.py

:: 如果发生错误，暂停以查看错误信息
if errorlevel 1 pause
