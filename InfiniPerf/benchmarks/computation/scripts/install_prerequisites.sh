#!/bin/bash

# 检查 xmake 是否安装
if ! command -v xmake &> /dev/null; then
    echo "xmake 未安装，正在安装..."
    curl -fsSL https://xmake.io/shget.text | bash
    source ~/.xmake/profile
else
    echo "xmake 已安装。"
fi

# 检查 Python gguf 包是否安装
if ! python -c "import gguf" &> /dev/null; then
    echo "gguf 未安装，正在安装..."
    pip install gguf -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
else
    echo "gguf 已安装。"
fi
