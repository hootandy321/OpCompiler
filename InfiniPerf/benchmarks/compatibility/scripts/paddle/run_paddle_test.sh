#!/bin/bash

# ==============================================================================
# 1. 初始化和定义变量
# ==============================================================================
echo "正在初始化环境..."
cd ../../../../ && source env.sh && cd -

# 定义统一的路径和日志文件名称
LOG_FILENAME="all_models_execution.log"
SCRIPT_BASE_DIR="$INFINIPERF_ROOT/benchmarks/compatibility/scripts/paddle"
MODEL_ROOT_DIR="$INFINIPERF_ROOT/benchmarks/compatibility/paddle"

LOG_FILE_PATH="$SCRIPT_BASE_DIR/$LOG_FILENAME"

# 合并两个脚本中的模型列表，方便统一管理和扩展
ALL_MODELS=(
    "lstm"
    "gan"
    "transformer"
    "vgg11"
    "resnet18"
    "yolov3"
)

# ==============================================================================
# 2. 准备工作
# ==============================================================================
# 切换到脚本所在的基础目录，这能确保后续的相对路径（如 requirements.txt）是正确的
cd "$SCRIPT_BASE_DIR" || { echo "严重错误: 无法进入脚本目录: $SCRIPT_BASE_DIR，程序退出。"; exit 1; }

echo "正在安装 Python 依赖..."
pip install -r requirements.txt || {
    echo "依赖安装失败，请检查网络连接和 pip 版本。"
    exit 1
}
echo "所有依赖安装成功!"

# 清空或创建日志文件
echo "脚本执行开始于: $(date)" > "$LOG_FILE_PATH"
echo "=========================================" >> "$LOG_FILE_PATH"
echo "" >> "$LOG_FILE_PATH"

# ==============================================================================
# 3. 定义可重用的核心执行函数
#    这个函数封装了处理单个模型的所有逻辑
# ==============================================================================
process_model() {
    # 将传入的第一个参数赋值给局部变量 model_name
    local model_name="$1"
    
    echo "-----------------------------------------"
    echo "开始处理模型: $model_name"

    local model_path="$MODEL_ROOT_DIR/$model_name"
    local python_script="$model_name.py"

    # 检查模型目录是否存在，如果不存在则记录日志并跳过
    if [ ! -d "$model_path" ]; then
        echo "错误: 找不到模型目录 '$model_path'。跳过。"
        echo "--- [错误] 找不到目录: $model_name ---" >> "$LOG_FILE_PATH"
        return # 从函数返回，继续下一个模型的处理
    fi

    # 进入模型目录
    cd "$model_path" || {
        echo "错误: 无法进入目录 '$model_path'。跳过。"
        cd "$SCRIPT_BASE_DIR" # 确保返回基础目录
        return
    }

    # 检查Python脚本是否存在
    if [ ! -f "$python_script" ]; then
        echo "错误: 找不到 Python 脚本 '$python_script'。跳过。"
        echo "--- [错误] 找不到脚本: $model_name/$python_script ---" >> "$LOG_FILE_PATH"
        cd "$SCRIPT_BASE_DIR" # 确保返回基础目录
        return
    fi

    # 执行脚本并记录日志
    echo "正在执行脚本: $python_script ..."
    echo "--- [日志开始] 模型: $model_name ---" >> "$LOG_FILE_PATH"

    # 使用 &>> 将标准输出(stdout)和标准错误(stderr)都追加到日志文件
    python3 "$python_script" &>> "$LOG_FILE_PATH"
    local exit_code=$? # 保存刚刚执行的命令的退出码

    # 检查退出状态并记录结果
    if [ $exit_code -eq 0 ]; then
        echo "模型 '$model_name' 执行成功。"
        echo "--- [成功] 模型: $model_name ---" >> "$LOG_FILE_PATH"
    else
        echo "错误: 模型 '$model_name' 执行失败 (退出码: $exit_code)。详情请见日志。"
        echo "--- [失败] 模型: $model_name (退出码: $exit_code) ---" >> "$LOG_FILE_PATH"
    fi
    echo "" >> "$LOG_FILE_PATH" # 在日志中添加空行，方便阅读

    # 任务完成后，安全地返回到脚本的基础目录
    cd "$SCRIPT_BASE_DIR" || { echo "严重错误: 无法返回基础目录。程序退出。"; exit 1; }
    echo "处理完成: $model_name"
    echo ""
}

# ==============================================================================
# 4. 主循环：遍历所有模型并执行
# ==============================================================================
for model in "${ALL_MODELS[@]}"
do
    process_model "$model"
done

# ==============================================================================
# 5. 结束
# ==============================================================================
echo "-----------------------------------------"
echo "所有任务已完成。详细日志请查看: $LOG_FILE_PATH"


