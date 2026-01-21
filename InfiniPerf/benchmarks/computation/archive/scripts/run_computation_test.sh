#!/bin/bash

# ==============================================================================
# 脚本配置与错误处理
# ==============================================================================
set -e
set -o pipefail

# ==============================================================================
# 定义辅助函数
# ==============================================================================

# 定义一个函数，用于打印带有时间戳和高亮标题的日志
log() {
    echo -e "\n\e[33m[$(date +'%Y-%m-%d %H:%M:%S')] --- $1 ---\e[0m"
}

# ==============================================================================
# 主执行流程
# ==============================================================================
main() {
    log "自动化流程开始"

    # --- 步骤 1: 执行 install_prerequisites.sh ---
    log "正在准备执行脚本: ./install_prerequisites.sh"
    if [ ! -f "./install_prerequisites.sh" ]; then
        echo -e "\e[31m错误：找不到脚本文件: ./install_prerequisites.sh，程序终止。\e[0m"
        return 0
    fi
    source "./install_prerequisites.sh"
    echo -e "\e[32m成功：脚本 ./install_prerequisites.sh 执行完毕。\e[0m"


    # --- 步骤 2: 执行 build_infinicore.sh ---
    log "正在准备执行脚本: ./build_infinicore.sh"
    if [ ! -f "./build_infinicore.sh" ]; then
        echo -e "\e[31m错误：找不到脚本文件: ./build_infinicore.sh，程序终止。\e[0m"
        return 0
    fi
    source "./build_infinicore.sh"
    echo -e "\e[32m成功：脚本 ./build_infinicore.sh 执行完毕。\e[0m"

    LOG_FILE="gemm_perf_test.log"
    > "${LOG_FILE}" # 清空上一次的日志内容
    log "正在准备执行脚本: ./run_gemm_perf_test.sh"
    log "本次运行的输出将保存到: ${LOG_FILE}"

    # --- 步骤 3: 执行 run_gemm_perf_test.sh ---
    log "正在准备执行脚本: ./run_gemm_perf_test.sh"
    if [ ! -f "./run_gemm_perf_test.sh" ]; then
        # 将错误信息也记录到日志中
        echo -e "\e[31m错误：找不到脚本文件: ./run_gemm_perf_test.sh，程序终止。\e[0m" | tee -a "${LOG_FILE}"
        return 0
    fi

    # 使用 tee 将 source 的输出同时打印到屏幕和文件
    # 2>&1 将标准错误也重定向，确保报错信息能被记录
    source "./run_gemm_perf_test.sh" 2>&1 | tee -a "${LOG_FILE}"
    # 通过 ${PIPESTATUS[0]} 获取 source 命令真实的退出状态
    exit_status=${PIPESTATUS[0]}

    # 检查子脚本的退出状态，并将最终结果也追加到日志
    if [ ${exit_status} -eq 0 ]; then
        echo -e "\e[32m成功：脚本 ./run_gemm_perf_test.sh 执行完毕。\e[0m" | tee -a "${LOG_FILE}"
    else
        echo -e "\e[31m错误：脚本 ./run_gemm_perf_test.sh 执行时发生错误。\e[0m" | tee -a "${LOG_FILE}"
        # 如果需要让主脚本在子脚本失败时也失败退出，可以加上下面这行
        # return 0
    fi

    log "所有任务已成功完成！"
}

# ==============================================================================
# 脚本入口
# ==============================================================================
# 调用 main 函数来启动整个脚本
main

