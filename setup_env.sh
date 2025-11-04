#!/bin/bash
# 创建 Python 3.10 虚拟环境并安装依赖
# 使用 pip 而不是 conda

echo "正在创建 Python 3.10 虚拟环境..."
python3.10 -m venv venv_pcb

echo "激活虚拟环境..."
source venv_pcb/bin/activate

echo "升级 pip..."
python -m pip install --upgrade pip

echo "安装 uv..."
python -m pip install uv

echo "安装依赖包..."
uv pip install -r requirements.txt

echo ""
echo "============================================"
echo "环境设置完成！"
echo "============================================"
echo ""
echo "使用方法："
echo "  1. 激活环境: source venv_pcb/bin/activate"
echo "  2. 运行检测: python pcb-detection.py --subset train --save-dir results/"
echo "  3. 退出环境: deactivate"
echo ""

