#!/bin/bash

# ==============================================================================
# 脚本配置与全局变量
# ==============================================================================

# SCRIPT_DIR 获取的是当前脚本所在的目录的绝对路径
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# PROJECT_ROOT 是脚本目录的上一级目录 (即 "scripts" 目录的父目录)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")


# 定义两个数组，用于在脚本结束时生成总结报告
SUCCESSFUL_TESTS=()
FAILED_TESTS=()
# 总测试计数器
TOTAL_TESTS=0

# 定义一个函数，用于打印带有时间戳和高亮标题的日志
log() {
    echo -e "\n\e[33m[$(date +'%Y-%m-%d %H:%M:%S')] --- $1 ---\e[0m"
}

show_help() {
    echo "用法: bash $(basename "$0") [测试名称_1] [参数...] [测试名称_2] [参数...]"
    echo
    echo "这个脚本可以运行一个或所有性能/兼容性测试。"
    echo "您可以提供一个或多个测试名称来按顺序执行它们。"
    echo
    echo "可用选项:"
    echo "  [无参数]                    运行所有下面列出的测试 (infinitensor 和 infinilm_infer 将使用默认参数)。"
    echo "  computation                 运行 Computation 性能测试。"
    echo "  paddle                      运行 PaddlePaddle 兼容性测试。"
    echo "  megatron                    运行 Megatron 兼容性测试。"
    echo "  cuda_samples                运行 CUDA Samples 兼容性测试。"
    echo "  infinilm_infer <path>       运行 InfiniLM 大模型推理测试。"
    echo "                              例如: bash $(basename "$0") infinilm_infer /workspace/models/9G7B_MHA"
    echo "  infinitensor <card> <dev>   运行 InfiniTensor 兼容性测试。"
    echo "                              例如: bash $(basename "$0") infinitensor cuda 0"
    echo "  --help, -h                  显示此帮助信息。"
    echo
}

# ==============================================================================
# 各个测试的执行函数
# ==============================================================================

# --- 测试 1: Computation ---
run_computation_test() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    log "开始执行: Computation 测试"

    (
        set -e # 在子 Shell 中，我们希望它在失败时立即退出
        cd "${PROJECT_ROOT}/benchmarks/computation/scripts"
        source run_computation_test.sh
    )

    # 检查子 Shell 的退出码
    if [ $? -eq 0 ]; then
        echo -e "\e[32m成功: Computation 测试已完成。\e[0m"
        SUCCESSFUL_TESTS+=("Computation")
    else
        echo -e "\e[31m失败: Computation 测试在执行过程中发生错误。\e[0m"
        FAILED_TESTS+=("Computation")
    fi
}

# --- 测试 2: PaddlePaddle ---
run_paddle_test() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    log "开始执行: PaddlePaddle 测试"
    (
        set -e
        cd "${PROJECT_ROOT}/benchmarks/compatibility/scripts/paddle"
        source run_paddle_test.sh
    )
    if [ $? -eq 0 ]; then
        echo -e "\e[32m成功: PaddlePaddle 测试已完成。\e[0m"
        SUCCESSFUL_TESTS+=("PaddlePaddle")
    else
        echo -e "\e[31m失败: PaddlePaddle 测试在执行过程中发生错误。\e[0m"
        FAILED_TESTS+=("PaddlePaddle")
    fi
}

# --- 测试 3: Megatron ---
run_megatron_test() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    log "开始执行: Megatron 测试"
    (
        set -e
        cd "${PROJECT_ROOT}/benchmarks/compatibility/scripts/megatron"
        source run_megatron_test.sh
    )
    if [ $? -eq 0 ]; then
        echo -e "\e[32m成功: Megatron 测试已完成。\e[0m"
        SUCCESSFUL_TESTS+=("Megatron")
    else
        echo -e "\e[31m失败: Megatron 测试在执行过程中发生错误。\e[0m"
        FAILED_TESTS+=("Megatron")
    fi
}

# --- 测试 4: CUDA Samples ---
run_cuda_samples_test() {
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    log "开始执行: CUDA Samples 测试"
    (
        set -e
        cp "${PROJECT_ROOT}/benchmarks/compatibility/scripts/cuda-samples/run_cuda_samples_test.sh" "${PROJECT_ROOT}/benchmarks/compatibility/cuda-samples/Samples"
        cd "${PROJECT_ROOT}/benchmarks/compatibility/cuda-samples/Samples"
        source run_cuda_samples_test.sh
    )
    if [ $? -eq 0 ]; then
        echo -e "\e[32m成功: CUDA Samples 测试已完成。\e[0m"
        SUCCESSFUL_TESTS+=("CUDA Samples")
    else
        echo -e "\e[31m失败: CUDA Samples 测试在执行过程中发生错误。\e[0m"
        FAILED_TESTS+=("CUDA Samples")
    fi
}

# --- 测试 5: InfiniLM Infer ---
run_infinilm_infer_test() {
    # 如果第一个参数为空，使用默认路径
    local model_path="${1:-/workspace/9G7B_MHA}" 

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    log "开始执行: InfiniLM Infer 测试 (模型路径: ${model_path})"
    (
        set -e
        cd "${PROJECT_ROOT}/benchmarks/llm/scripts/InfiniLM"
        source env.sh
        bash run_infinilm_infer_test.sh "${model_path}"
    )
    if [ $? -eq 0 ]; then
        echo -e "\e[32m成功: InfiniLM Infer 测试已完成。\e[0m"
        SUCCESSFUL_TESTS+=("InfiniLM Infer (路径: ${model_path})")
    else
        echo -e "\e[31m失败: InfiniLM Infer 测试在执行过程中发生错误。\e[0m"
        FAILED_TESTS+=("InfiniLM Infer (路径: ${model_path})")
    fi
}

# --- 测试 6: InfiniTensor ---
run_infinitensor_test() {
    local accelerator_card="${1:-gpu}" # 如果第一个参数为空，默认为 'gpu'
    local device_number="${2:-0}"    # 如果第二个参数为空，默认为 '0'

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    log "开始执行: InfiniTensor 测试 (加速卡: ${accelerator_card}, 设备号: ${device_number})"
    (
        set -e
        cd "${PROJECT_ROOT}/benchmarks/compatibility/scripts/infinitensor"
        source run_infinitensor_test.sh ${accelerator_card} ${device_number}
    )
    if [ $? -eq 0 ]; then
        echo -e "\e[32m成功: InfiniTensor 测试已完成。\e[0m"
        SUCCESSFUL_TESTS+=("InfiniTensor (卡: ${accelerator_card}, 设备: ${device_number})")
    else
        echo -e "\e[31m失败: InfiniTensor 测试在执行过程中发生错误。\e[0m"
        FAILED_TESTS+=("InfiniTensor (卡: ${accelerator_card}, 设备: ${device_number})")
    fi
}

print_summary_report() {
    log "所有测试已执行完毕，生成总结报告"
    echo "======================================================"
    echo "测试结果总结:"
    echo "  总共执行测试数: $TOTAL_TESTS"
    echo -e "  \e[32m成功: ${#SUCCESSFUL_TESTS[@]}\e[0m"
    echo -e "  \e[31m失败: ${#FAILED_TESTS[@]}\e[0m"
    echo
    if [ ${#FAILED_TESTS[@]} -ne 0 ]; then
        echo -e "\e[31m以下测试项失败:"
        for test_name in "${FAILED_TESTS[@]}"; do
            echo "  - $test_name"
        done
        echo -e "\e[0m"
    else
        echo -e "\e[32m所有测试项均成功通过！\e[0m"
    fi
    echo "======================================================"
}

# ==============================================================================
# 主执行逻辑
# ==============================================================================
main() {
    # 步骤 1: 初始化顶层环境
    log "初始化顶层环境 (env.sh)..."
    source env.sh
    echo -e "\e[32m成功: 环境初始化完毕。\e[0m"

    # 步骤 2: 根据输入参数决定执行哪个测试
    if [ $# -eq 0 ]; then
        # 如果没有参数，则运行所有测试
        log "未指定特定测试，将运行所有测试..."
        run_computation_test
        run_paddle_test
        run_megatron_test
        run_cuda_samples_test
        run_infinilm_infer_test # 将使用默认参数
        run_infinitensor_test   # 将使用默认参数
    else
        # --- 使用 while 循环处理多个参数 ---
        while [[ $# -gt 0 ]]; do
            case "$1" in
                computation)
                    run_computation_test
                    shift # 处理完一个参数后，移到下一个
                    ;;
                paddle)
                    run_paddle_test
                    shift
                    ;;
                megatron)
                    run_megatron_test
                    shift
                    ;;
                cuda_samples)
                    run_cuda_samples_test
                    shift
                    ;;
                infinilm_infer)
                    # 检查 infinilm_infer 是否有提供额外的参数
                    if [ -z "$2" ] || [[ "$2" == -* ]]; then
                        echo -e "\e[31m错误: 'infinilm_infer' 测试需要一个额外参数: <模型路径>。\e[0m"
                        show_help
                        exit 1
                    fi
                    run_infinilm_infer_test "$2"
                    shift 2 # 移动 2 个位置 (test_name + path)
                    ;;
                infinitensor)
                    # 检查 infinitensor 是否有提供额外的参数
                    if [ -z "$2" ] || [ -z "$3" ] || [[ "$2" == -* ]] || [[ "$3" == -* ]]; then
                        echo -e "\e[31m错误: 'infinitensor' 测试需要两个额外参数: <加速卡> 和 <设备号>。\e[0m"
                        show_help
                        exit 1
                    fi
                    run_infinitensor_test "$2" "$3"
                    shift 3 # 移动 3 个位置 (test_name + card + dev)
                    ;;
                --help|-h)
                    show_help
                    exit 0
                    ;;
                *)
                    echo -e "\e[31m错误: 未知的测试名称 '$1'\e[0m"
                    show_help
                    exit 1
                    ;;
            esac
        done
    fi

    # 步骤 3: 打印最终的总结报告
    print_summary_report

    # 如果有任何测试失败，脚本最终以失败状态退出
    if [ ${#FAILED_TESTS[@]} -ne 0 ]; then
        exit 1
    fi
}

# 调用 main 函数来启动整个流程
main "$@"
