@echo off
chcp 65001
cls

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo 请求管理员权限...
    powershell Start-Process -FilePath "%0" -Verb RunAs
    exit /b
)

setlocal enabledelayedexpansion

echo 正在启动游戏自动化程序...

REM Check if Python is installed
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误：未找到 Python。请确保 Python 已安装并添加到 PATH 中。
    pause
    exit /b 1
)

REM Create virtual environment (if it doesn't exist)
if not exist venv (
    echo 正在创建虚拟环境...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo 错误：创建虚拟环境失败。
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo 错误：激活虚拟环境失败。
    pause
    exit /b 1
)

REM Install necessary packages
echo 正在检查并安装必要的包...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo 错误：安装必要的包失败。请检查网络连接或尝试手动安装。
    pause
    exit /b 1
)

REM Check if necessary files exist
for %%F in (main/full_feature_launcher.py config/game_settings.yaml) do (
    if not exist %%F (
        echo 错误：未找到 %%F 文件。请确保所有必要文件都在正确的位置。
        pause
        exit /b 1
    )
)

REM Database initialization and optimization
echo 正在初始化和优化数据库...
python -c "from utils.data_handler import DataHandler; DataHandler().initialize_database()" 2>error.log
if %errorlevel% neq 0 (
    echo 错误：初始化和优化数据库失败。
    echo 错误详情：
    type error.log
    pause
    exit /b 1
)

REM Detect ADB devices
echo 正在检测 ADB 设备...
adb devices > adb_devices.txt
set "device_count=0"
for /f "skip=1 tokens=1,2" %%a in (adb_devices.txt) do (
    set /a "device_count+=1"
    set "device[!device_count!]=%%a"
    set "status[!device_count!]=%%b"
)

if %device_count% equ 0 (
    echo 错误：未检测到 ADB 设备。请确保模拟器或设备已正确连接。
    pause
    exit /b 1
)

if %device_count% equ 1 (
    set "selected_device=!device[1]!"
    echo 已检测到一个 ADB 设备：!selected_device!
) else (
    echo 检测到多个 ADB 设备，请选择要使用的设备：
    for /l %%i in (1,1,%device_count%) do (
        echo %%i. !device[%%i]! (!status[%%i]!)
    )
    set /p "choice=请输入设备编号（1-%device_count%）: "
    set "selected_device=!device[%choice%]!"
)

echo 选择的设备：%selected_device%

REM Set ADB_DEVICE environment variable
set ADB_DEVICE=%selected_device%

REM Run Python script
echo 正在启动游戏自动化程序...
python main/full_feature_launcher.py 2>error.log
if %errorlevel% neq 0 (
    echo 程序运行出错。错误详情：
    type error.log
    pause
    exit /b 1
)

echo 程序已成功运行完毕。
pause