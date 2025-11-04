@echo off
REM 创建 Python 3.10 虚拟环境并安装依赖
REM 使用 pip 而不是 conda

REM 检查 Python 是否已安装
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到 Python，请先安装 Python 3.10 或更高版本
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 检查 Python 版本
python --version
if %errorlevel% neq 0 (
    echo 错误: Python 命令执行失败
    pause
    exit /b 1
)

echo.
echo 正在创建 Python 虚拟环境...
python -m venv venv_pcb
if %errorlevel% neq 0 (
    echo 错误: 创建虚拟环境失败
    pause
    exit /b 1
)

echo.
echo 激活虚拟环境...
if not exist "venv_pcb\Scripts\activate.bat" (
    echo 错误: 虚拟环境激活脚本不存在
    pause
    exit /b 1
)
call venv_pcb\Scripts\activate.bat

echo.
echo 升级 pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo 警告: pip 升级失败，继续执行...
)

echo.
echo 安装 uv...
python -m pip install uv
if %errorlevel% neq 0 (
    echo 错误: 安装 uv 失败
    pause
    exit /b 1
)

echo.
echo 安装依赖包...
if not exist "requirements.txt" (
    echo 错误: 未找到 requirements.txt 文件
    pause
    exit /b 1
)
uv pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo 错误: 安装依赖包失败
    pause
    exit /b 1
)

echo.
echo ============================================
echo 环境设置完成！
echo ============================================
echo.
echo 使用方法：
echo   1. 激活环境: venv_pcb\Scripts\activate.bat
echo   2. 运行检测: python pcb-detection.py --subset train --save-dir results/
echo   3. 退出环境: deactivate
echo.

pause

