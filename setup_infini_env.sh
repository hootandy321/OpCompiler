#!/bin/bash

# Infini项目完整环境安装脚本
# 根据各项目README文档安装整个Infini生态

set -e  # 遇到错误立即退出

echo "======================================================================"
echo "Infini 项目环境配置脚本"
echo "======================================================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 项目根目录
INFINI_ROOT_DIR="$(pwd)"
VENV_DIR="$INFINI_ROOT_DIR/infini_venv"

# 检查是否在正确的目录
if [ ! -d "ninetoothed" ] || [ ! -d "InfiniCore" ] || [ ! -d "ntops" ]; then
    echo -e "${RED}错误: 请在 Infini 项目根目录运行此脚本${NC}"
    echo "当前目录: $INFINI_ROOT_DIR"
    exit 1
fi

# ============================================================================
# 步骤 1: 创建并激活虚拟环境
# ============================================================================
echo -e "${GREEN}[1/7] 创建Python虚拟环境...${NC}"
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}虚拟环境已存在，跳过创建${NC}"
else
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ 虚拟环境创建成功: $VENV_DIR${NC}"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ 虚拟环境已激活${NC}"
echo ""

# ============================================================================
# 步骤 2: 配置环境变量 (INFINI_ROOT 和 LD_LIBRARY_PATH)
# ============================================================================
echo -e "${GREEN}[2/7] 配置环境变量...${NC}"
export INFINI_ROOT="$HOME/.infini"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:$LD_LIBRARY_PATH"

# 创建目录
mkdir -p "$INFINI_ROOT/lib"
mkdir -p "$INFINI_ROOT/bin"
mkdir -p "$INFINI_ROOT/include"

echo "INFINI_ROOT=$INFINI_ROOT"
echo "LD_LIBRARY_PATH已更新"
echo -e "${GREEN}✓ 环境变量配置完成${NC}"
echo ""

# ============================================================================
# 步骤 3: 安装九齿编译器 (ninetoothed)
# ============================================================================
echo -e "${GREEN}[3/7] 安装 ninetoothed (九齿编译器)...${NC}"
cd "$INFINI_ROOT_DIR/ninetoothed"

echo "安装ninetoothed及其依赖..."
pip install --upgrade pip
pip install -e .

echo -e "${GREEN}✓ ninetoothed 安装完成${NC}"
echo ""

# ============================================================================
# 步骤 4: 安装九齿算子库 (ntops)
# ============================================================================
echo -e "${GREEN}[4/7] 安装 ntops (九齿算子库)...${NC}"
cd "$INFINI_ROOT_DIR/ntops"

echo "安装ntops..."
pip install -e .

echo -e "${GREEN}✓ ntops 安装完成${NC}"
echo ""

# ============================================================================
# 步骤 5: 安装 InfiniCore 底层库
# ============================================================================
echo -e "${GREEN}[5/7] 安装 InfiniCore 底层库...${NC}"
cd "$INFINI_ROOT_DIR/InfiniCore"

# 检查是否有GPU
if [ -n "$CUDA_HOME" ] || [ -n "$CUDA_PATH" ]; then
    echo -e "${YELLOW}检测到CUDA环境，将编译GPU支持${NC}"
    HAS_GPU=true
else
    echo -e "${YELLOW}未检测到CUDA环境，仅编译CPU版本${NC}"
    HAS_GPU=false
fi

# 使用安装脚本
if [ "$HAS_GPU" = true ]; then
    echo "运行安装脚本（包含GPU支持）..."
    # 如果设置了CUDA_HOME
    if [ -n "$CUDA_HOME" ]; then
        python scripts/install.py --nv-gpu=y --cuda=$CUDA_HOME
    elif [ -n "$CUDA_PATH" ]; then
        python scripts/install.py --nv-gpu=y --cuda=$CUDA_PATH
    else
        python scripts/install.py --nv-gpu=y
    fi
else
    echo "运行安装脚本（仅CPU）..."
    python scripts/install.py
fi

echo -e "${GREEN}✓ InfiniCore 底层库安装完成${NC}"
echo ""

# ============================================================================
# 步骤 6: 安装 InfiniCore C++ 库
# ============================================================================
echo -e "${GREEN}[6/7] 安装 InfiniCore C++ 库...${NC}"
cd "$INFINI_ROOT_DIR/InfiniCore"

xmake build _infinicore
xmake install _infinicore

echo -e "${GREEN}✓ InfiniCore C++ 库安装完成${NC}"
echo ""

# ============================================================================
# 步骤 7: 安装 InfiniCore Python 包
# ============================================================================
echo -e "${GREEN}[7/7] 安装 InfiniCore Python 包...${NC}"
cd "$INFINI_ROOT_DIR/InfiniCore"

pip install -e .

echo -e "${GREEN}✓ InfiniCore Python 包安装完成${NC}"
echo ""

# ============================================================================
# 安装验证
# ============================================================================
echo "======================================================================"
echo "安装验证"
echo "======================================================================"
echo ""

# 验证Python包
echo "检查已安装的Python包..."
pip list | grep -E "(ninetoothed|ntops|infinicore)" || echo "警告: 某些包未正确安装"
echo ""

# 验证环境变量
echo "环境变量:"
echo "  INFINI_ROOT=$INFINI_ROOT"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo ""

# 验证库文件
echo "检查库文件:"
if [ -d "$INFINI_ROOT/lib" ]; then
    echo -e "${GREEN}✓ 库目录存在: $INFINI_ROOT/lib${NC}"
    ls -lh "$INFINI_ROOT/lib" | head -10
else
    echo -e "${RED}✗ 库目录不存在${NC}"
fi
echo ""

# ============================================================================
# 完成
# ============================================================================
echo "======================================================================"
echo -e "${GREEN}安装完成！${NC}"
echo "======================================================================"
echo ""
echo "后续步骤:"
echo ""
echo "1. 激活虚拟环境（在新终端中）:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo "2. 配置环境变量（如果还没有添加到 ~/.bashrc）:"
echo "   export INFINI_ROOT=\"$HOME/.infini\""
echo "   export LD_LIBRARY_PATH=\"\$INFINI_ROOT/lib:\$LD_LIBRARY_PATH\""
echo ""
echo "3. 运行测试:"
echo "   cd InfiniCore"
echo "   python test/infinicore/run.py --cpu"
echo ""
echo "4. 永久保存环境变量（可选）:"
echo "   echo 'export INFINI_ROOT=\"$HOME/.infini\"' >> ~/.bashrc"
echo "   echo 'export LD_LIBRARY_PATH=\"\$INFINI_ROOT/lib:\$LD_LIBRARY_PATH\"' >> ~/.bashrc"
echo ""
echo "虚拟环境位置: $VENV_DIR"
echo ""
echo "======================================================================"
