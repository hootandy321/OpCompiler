# 计算性能测评

包括不同精度峰值算力与算子性能测评。

`scripts/` 目录下为使用 infinicore 测试框架进行算子测试的脚本。 `archive/scripts` 目录下为使用旧 infiniop-test 框架进行算子测试的脚本。

## 峰值算力测试

采用 8192 x 8192 的矩阵乘计算，涵盖 FP32、FP16 两种精度。

运行方式（运行下述脚本将一同进行此项和矩阵乘算子性能的测试）:

```(shell)
. install_prerequisites.sh
. build_infinicore.sh
. run_gemm_perf_test.sh
```

## 算子性能测试

### 矩阵乘

采用下述两个测例的矩阵乘计算，涵盖 FP32、FP16 两种精度。
- [512, 5120] x [5120, 5120]
- [512, 5120] x [5120, 13824]

运行方式（运行下述脚本将一同进行此项和峰值算力性能的测试）:

```(shell)
. install_prerequisites.sh
. build_infinicore.sh
. run_gemm_perf_test.sh
```

