# 兼容性测评

包括 CUDA 兼容性和框架兼容性测评。

## CUDA 兼容性测评

## 框架兼容性测评

### Megatron-LM

脚本位于 `megatron/` 内，执行八卡 Llama2-7B 预训练任务。进入 `megatron/` 目录下，依次运行：

```
. build.sh
. process_data.sh
. pretrain_llama.sh
```

如果需要删除运行过程中产生的 log、data 以及 checkpoint 等缓存，运行：

```
. remove_cache_data.sh
```

### nccl-tests

脚本位于 `nccl-tests @ b4300cc/` 所链接仓库，支持单机多卡nccl-tests：
```
. 进入链接仓库并 clone
. 切换到 batch_test_zs 分支
. 执行 batch_test/test_nccl.sh (注意CUDA_HOME、NCCL_HOME 和 LD_LIBRARY_PAT 环境变量设置)
```
如果需要删除运行过程中产生的 log、build 文件，运行：

```
. remove_cache_data.sh
```

### cuda-samples

脚本位于 `cuda-samples/` 所链接仓库：
```
. 进入链接仓库并 clone
. 切换到 batch_test 分支
. 执行 batch_test/build_cuda_samples.sh 
. 可利用 batch_test/check_files.py 查看 cuda-samples 的运行结果与状态
```

如果需要删除运行过程中产生的 log、build 文件，运行：
```
. remove_cache_data.sh
```
测试环境（供参考）：Ubuntu 20.04 LTS, nvcc --version = 12.2, gcc --version = 9.4.0, cmake version = 3.27.9


### InfiniTensor

采用四个领域共16个模型进行性能测试：(测试模型路径"/data5/shared/InfiniPerfModels/")

| 模型名称     | 图像分类 | 自然语言处理 | 对抗生成 | 超分辨率 |
| ------------ | -------- | ------------ | -------- | -------- |
| ResNet101    | ✅        |              |          |          |
| GoogleNet    | ✅        |              |          |          |
| InceptionV3  | ✅        |              |          |          |
| ShuffleNetV2 | ✅        |              |          |          |
| DenseNet121  | ✅        |              |          |          |
| Bert         |          | ✅            |          |          |
| GPT2         |          | ✅            |          |          |
| OPT          |          | ✅            |          |          |
| Ernie        |          | ✅            |          |          |
| Splinter     |          | ✅            |          |          |
| BGAN         |          |              | ✅        |          |
| CGAN         |          |              | ✅        |          |
| DCGAN        |          |              | ✅        |          |
| SRCNN        |          |              |          | ✅        |
| ESPCN        |          |              |          | ✅        |
| SRGAN        |          |              |          | ✅        |

运行方式：
```(shell)
. build_infinitensor.sh [gpu/mlu/npu/xpu]
. run_model_perf_test.sh [gpu/mlu/npu/xpu] [device_id]
```
