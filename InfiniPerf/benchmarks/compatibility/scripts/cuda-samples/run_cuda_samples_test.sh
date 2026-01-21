#!/bin/bash

# --- Configuration ---
# List of parent directories to process (modify or confirm based on your setup)
PARENT_DIRS=(
    "0_Introduction"
    "1_Utilities"
    "2_Concepts_and_Techniques"
    "3_CUDA_Features"
    "4_CUDA_Libraries"
    "5_Domain_Specific"
    "6_Performance"
)

# Specific subdirectory names to skip
SKIP_SUBDIRS=("common" "utils")

# Log files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG_FILE="make_only_all_$TIMESTAMP.log"
FAILURES_DETAIL_LOG_FILE="make_failed_details_$TIMESTAMP.log"

# Timeout for a single compilation step in seconds (10 minutes = 600 seconds)
TIMEOUT_DURATION=600

# Statistics
SUCCESS_COUNT=0
FAILURE_COUNT=0
declare -a FAILED_DIRS

# --- Initialize Logs ---
# Main log file
echo "Script start time: $(date)" > "$MAIN_LOG_FILE"
echo "Main log file: $MAIN_LOG_FILE" | tee -a "$MAIN_LOG_FILE"
# Failures detail log file
echo "Failures detail log start time: $(date)" > "$FAILURES_DETAIL_LOG_FILE"
echo "Failures detail log file: $FAILURES_DETAIL_LOG_FILE" | tee -a "$MAIN_LOG_FILE"

echo "Timeout set to: $TIMEOUT_DURATION seconds" | tee -a "$MAIN_LOG_FILE"
echo "Parent directories to check: ${PARENT_DIRS[*]}" | tee -a "$MAIN_LOG_FILE"
echo "Subdirectories to skip: ${SKIP_SUBDIRS[*]}" | tee -a "$MAIN_LOG_FILE"
echo "==================================================" | tee -a "$MAIN_LOG_FILE"

# --- Main Logic ---
# Get the script's base directory for returning and log file paths
SCRIPT_BASE_DIR=$(pwd)

for p_dir in "${PARENT_DIRS[@]}"; do
    echo "Processing parent directory: $p_dir" | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"
    if [ ! -d "$p_dir" ]; then
        echo "Error: Parent directory '$p_dir' does not exist, skipping." | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"
        continue
    fi

    # Enter parent directory for easier subdirectory handling
    cd "$p_dir" || { echo "Error: Cannot enter directory '$p_dir' from '$(pwd)'"; exit 1; }

    for sub_dir_name in *; do
        # Check if it's a directory
        if [ ! -d "$sub_dir_name" ]; then
            continue
        fi

        current_processing_sub_dir_abspath="$(pwd)/$sub_dir_name" # For logging

        # Check if it's a directory to skip
        skip_this_dir=false
        for skip_name in "${SKIP_SUBDIRS[@]}"; do
            if [ "$sub_dir_name" == "$skip_name" ]; then
                skip_this_dir=true
                break
            fi
        done

        if $skip_this_dir; then
            echo "Info: Skipping special directory '$current_processing_sub_dir_abspath' (in skip list)" | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"
            continue
        fi

        executable_name="$sub_dir_name"

        echo "------------------------------------------" | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"
        echo "Start processing subdirectory: $current_processing_sub_dir_abspath" | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"
        echo "Time: $(date)" | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"

        # Enter subdirectory for compilation
        cd "$sub_dir_name" || {
            echo "Error: Cannot enter subdirectory '$sub_dir_name' from '$(pwd)'" | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"
            FAILURE_COUNT=$((FAILURE_COUNT + 1))
            FAILED_DIRS+=("$current_processing_sub_dir_abspath (Cannot enter directory)")
            echo "--- Log for failed attempt: $current_processing_sub_dir_abspath (Cannot enter directory) ---" >> "$SCRIPT_BASE_DIR/$FAILURES_DETAIL_LOG_FILE"
            echo "Error: Cannot enter subdirectory '$sub_dir_name' from '$(pwd)'" >> "$SCRIPT_BASE_DIR/$FAILURES_DETAIL_LOG_FILE"
            echo "--- End log for $current_processing_sub_dir_abspath ---" >> "$SCRIPT_BASE_DIR/$FAILURES_DETAIL_LOG_FILE"
            echo "" >> "$SCRIPT_BASE_DIR/$FAILURES_DETAIL_LOG_FILE"
            continue
        }

        # Clean up previous CMake cache and build files
        echo "Info: Clearing previous CMake cache and build artifacts..." | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"
        rm -f CMakeCache.txt
        rm -f cmake_install.cmake
        rm -f CTestTestfile.cmake
        rm -f "$executable_name" # Remove old executable
        rm -rf CMakeFiles/

        # Temporary log file to capture output for the current subdirectory
        TEMP_BUILD_LOG="current_make_output.tmp"
        # Clear temporary log
        > "$TEMP_BUILD_LOG"

        echo "Info: Executing 'make clean && make' in '$current_processing_sub_dir_abspath'" | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"

        # Execute clean and compile with timeout
        # 'make clean' is usually fast, timeout is mainly for 'make'
        ( make clean && timeout "$TIMEOUT_DURATION" make SMS=80) >> "$TEMP_BUILD_LOG" 2>&1
        make_exit_status=$?

        local_failure_reason="" # To record the reason for failure

        if [ $make_exit_status -eq 0 ]; then
            # Compilation was successful. No further action needed.
            echo "Success: Compilation successful in '$current_processing_sub_dir_abspath'" | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        elif [ $make_exit_status -eq 124 ]; then
            local_failure_reason=" (Compilation timed out > $TIMEOUT_DURATION seconds)"
        else
            local_failure_reason=" (make command failed, exit code: $make_exit_status)"
        fi

        # Append temp log content to the main log
        cat "$TEMP_BUILD_LOG" >> "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"

        if [ -n "$local_failure_reason" ]; then # If there is any reason for failure
            echo "Failure: '$current_processing_sub_dir_abspath'$local_failure_reason" | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"
            FAILURE_COUNT=$((FAILURE_COUNT + 1))
            FAILED_DIRS+=("$current_processing_sub_dir_abspath$local_failure_reason")
            # Append temp log content to the failures detail log
            echo "--- Log for failed attempt: $current_processing_sub_dir_abspath$local_failure_reason ---" >> "$SCRIPT_BASE_DIR/$FAILURES_DETAIL_LOG_FILE"
            cat "$TEMP_BUILD_LOG" >> "$SCRIPT_BASE_DIR/$FAILURES_DETAIL_LOG_FILE"
            echo "--- End log for $current_processing_sub_dir_abspath ---" >> "$SCRIPT_BASE_DIR/$FAILURES_DETAIL_LOG_FILE"
            echo "" >> "$SCRIPT_BASE_DIR/$FAILURES_DETAIL_LOG_FILE"
        fi

        # Clean up the generated temp file
        rm -f "$TEMP_BUILD_LOG"

        # Return from the subdirectory to the parent directory
        cd ..

    done
    # Return from the parent directory to the script's initial execution directory
    cd "$SCRIPT_BASE_DIR" || { echo "Error: Cannot return to script base directory '$SCRIPT_BASE_DIR' from '$(pwd)'"; exit 1; }
done

# --- Summary Report ---
echo "==================================================" | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"
echo "Script end time: $(date)" | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"
echo "Total successful directories (compiled): $SUCCESS_COUNT" | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"
echo "Total failed directories (compile or timeout): $FAILURE_COUNT" | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"
echo "" | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"

if [ ${#FAILED_DIRS[@]} -ne 0 ]; then
    echo "The following directories failed or had issues:" | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"
    for failed_dir_info in "${FAILED_DIRS[@]}"; do
        echo "   - $failed_dir_info" | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"
    done
else
    echo "All eligible subdirectories were compiled successfully!" | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"
fi
echo "==================================================" | tee -a "$SCRIPT_BASE_DIR/$MAIN_LOG_FILE"
echo "For full logs, see: $SCRIPT_BASE_DIR/$MAIN_LOG_FILE"
echo "For failure details, see: $SCRIPT_BASE_DIR/$FAILURES_DETAIL_LOG_FILE"

