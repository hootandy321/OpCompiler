#!/bin/bash

# --- 脚本设置 ---
# set -e: 当任何命令以非零状态退出时，立即退出脚本。
# set -u: 当使用未定义的变量时，视为错误并立即退出。
# set -o pipefail: 如果管道中的任何命令失败，则整个管道的返回值为失败。
set -euo pipefail

# --- 彩色输出定义 ---
C_RED='\033[0;31m'
C_GREEN='\033[0;32m'
C_YELLOW='\033[0;33m'
C_RESET='\033[0m'

# --- 日志函数 ---
info() {
    echo -e "${C_GREEN}[INFO]${C_RESET} $1"
}
warn() {
    echo -e "${C_YELLOW}[WARN]${C_RESET} $1"
}
error() {
    echo -e "${C_RED}[ERROR]${C_RESET} $1"
}

# --- 1. 参数校验 ---
if [ "$#" -ne 1 ]; then
    error "参数数量错误。"
    echo "用法: $0 <model_path>"
    echo "示例: $0 /workspace/9G7B_MHA"
    echo "注意: 设备类型将通过环境变量 INFINIPERF_PLATFORM 自动检测。"
    exit 1
fi

# 将参数赋值给更易读的变量
MODEL_PATH=$1
INFER_SCRIPT_NAME="infer_jiuge.py" 
INFINILM_ROOT_PATH="../../InfiniLM"

# --- 脚本主逻辑 ---
info "Step 1: 设置环境变量..."
source env.sh
info "环境变量设置完毕。"

info "Step 1.5: 自动检测平台..."
DEVICE_FLAG=""
# 使用 ${VAR:-} 来安全地处理未设置的变量
case "${INFINIPERF_PLATFORM:-}" in
    "NVIDIA_GPU")    DEVICE_FLAG="--nvidia" ;;
    "CAMBRICON_MLU") DEVICE_FLAG="--cambricon" ;;
    "ASCEND_NPU")    DEVICE_FLAG="--ascend" ;;
    "METAX_GPU")     DEVICE_FLAG="--metax" ;;
    "MOORE_GPU")     DEVICE_FLAG="--moore" ;;
    "ILLUVATAR_GPU") DEVICE_FLAG="--iluvatar" ;;
    "CPU")           DEVICE_FLAG="--cpu" ;; # 为CPU添加一个映射
    "")
        error "环境变量 INFINIPERF_PLATFORM 未设置。"
        echo "请在运行前设置该变量, 例如: export INFINIPERF_PLATFORM=\"NVIDIA_GPU\""
        exit 1
        ;;
    *)
        error "不支持的环境变量值: INFINIPERF_PLATFORM=\"${INFINIPERF_PLATFORM}\""
        echo "支持的值包括: NVIDIA_GPU, CAMBRICON_MLU, ASCEND_NPU, METAX_GPU, MOORE_GPU, ILLUVATAR_GPU, CPU."
        exit 1
        ;;
esac
info "检测到平台: ${INFINIPERF_PLATFORM} -> 将使用设备标志: ${DEVICE_FLAG}"


warn "请确保InfiniCore已预先构建，否则脚本可能会失败。"

info "Step 2: 准备推理脚本..."
# 检查Python脚本是否存在
if [ ! -f "$INFER_SCRIPT_NAME" ]; then
    error "推理脚本 '$INFER_SCRIPT_NAME' 未在当前目录找到。"
    exit 1
fi
cp "$INFER_SCRIPT_NAME" "${INFINILM_ROOT_PATH}/scripts/"
info "已将 '$INFER_SCRIPT_NAME' 复制到 '${INFINILM_ROOT_PATH}/scripts/'"

info "Step 3: 进入项目目录并进行构建..."
export XMAKE_ROOT=y
cd "$INFINILM_ROOT_PATH"
# 检查并执行构建和安装
if xmake && xmake install; then
    info "项目构建和安装成功。"
else
    error "项目构建或安装失败。请检查上面的xmake输出。"
    # 清理复制的脚本并退出
    rm -f "scripts/$INFER_SCRIPT_NAME"
    exit 1
fi
cd scripts/

info "Step 4: 执行推理脚本..."
info "设备参数: $DEVICE_FLAG"
info "模型路径: $MODEL_PATH"
# 使用自动检测到的设备标志和传入的模型路径执行Python脚本
python "$INFER_SCRIPT_NAME" "$DEVICE_FLAG" "$MODEL_PATH"

info "Step 5: 清理临时文件..."
rm -f "$INFER_SCRIPT_NAME"
info "已删除临时推理脚本。"

# 返回到原始目录
cd - > /dev/null

info "脚本执行完毕！"


