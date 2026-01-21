#!/bin/bash

# ==============================================================================
# 脚本配置与错误处理
# ==============================================================================
set -e
set -o pipefail

# ==============================================================================
# 用于打印日志
# ==============================================================================
log() {
    # 打印带有高亮和时间戳的标题
    echo -e "\n\e[33m[$(date +'%Y-%m-%d %H:%M:%S')] --- $1 ---\e[0m"
}

run_build() {
    local card_type="$1" # 将传入的第一个参数作为加速卡类型

    log "正在准备为加速卡 [${card_type}] 编译 InfiniTensor (执行 build_infinitensor.sh)"
    if [ ! -f "./build_infinitensor.sh" ]; then
        echo -e "\e[31m错误：找不到编译脚本 build_infinitensor.sh，程序终止。\e[0m"
        exit 1
    fi
    
    # 将接收到的 card_type 参数传递给 build_infinitensor.sh
    source "./build_infinitensor.sh" "${card_type}"
    
    echo -e "\e[32m成功：编译脚本执行完毕。\e[0m"
}


# ==============================================================================
# 主执行流程
# ==============================================================================
main() {
    log "自动化流程开始"

    # --- 步骤 1: 检查并处理输入参数 ---
    if [ "$#" -lt 2 ]; then
        echo -e "\e[31m错误：参数不足。\e[0m"
        echo "用法: bash $(basename "$0") <加速卡类型> <设备号>"
        echo "例如: bash $(basename "$0") cuda 0"
        exit 1
    fi

    local accelerator_card="$1"
    local device_number="$2"

    log "接收到参数 -> 加速卡: ${accelerator_card}, 设备号: ${device_number}"


    # --- 步骤 2: 编译 InfiniTensor ---
    run_build "${accelerator_card}"


    # --- 步骤 3: 运行模型性能测试 ---
    log "正在准备运行模型性能测试 (执行 run_model_perf_test.sh)"
    if [ ! -f "./run_model_perf_test.sh" ]; then
        echo -e "\e[31m错误：找不到测试脚本 run_model_perf_test.sh，程序终止。\e[0m"
        exit 1
    fi
    source "./run_model_perf_test.sh" "${accelerator_card}" "${device_number}"
    echo -e "\e[32m成功：性能测试脚本执行完毕。\e[0m"


    log "所有任务已成功完成！"
}

# ==============================================================================
# 脚本入口
#
# 将所有收到的参数 ("$@") 原封不动地传递给 main 函数
# ==============================================================================
main "$@"

