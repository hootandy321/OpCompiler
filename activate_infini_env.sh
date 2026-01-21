#!/bin/bash
# Infini环境快速激活脚本
# 使用方法: source /home/qy/src/Infini/activate_infini_env.sh

# 激活虚拟环境
source /home/qy/src/Infini/infini_venv/bin/activate

# 设置环境变量
export INFINI_ROOT="$HOME/.infini"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$LD_LIBRARY_PATH"

# 显示当前状态
echo "======================================================================"
echo "Infini 开发环境已激活"
echo "======================================================================"
echo ""
echo "Python 环境:"
python --version
echo ""
echo "Python 路径:"
which python
echo ""
echo "已安装的关键包:"
pip list | grep -E "(ninetoothed|ntops|torch|infinicore)" || echo "  某些包可能尚未安装"
echo ""
echo "环境变量:"
echo "  INFINI_ROOT=$INFINI_ROOT"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo ""
echo "======================================================================"
echo ""
echo "提示: 运行 'deactivate' 可退出虚拟环境"
echo ""
