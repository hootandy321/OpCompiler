# 寒武纪平台硬件性能测评

## memcpy 访存测试

调用 cnrt 库接口进行 memcpy 访存测试。

进入 `./cnrt-memcpy` 目录，执行 `./build.sh` 进行编译。

运行 `./main <bytes-transferred, e.g. 10k, 16M, 1g>` 执行测试。

## CNVS 工具测试

使用 CNVS 工具进行 pcie 通信带宽、mlulink 通信带宽以及 matmul 矩阵乘性能测试。

### 硬件性能测试

进入 `./cnvs` 目录，执行 `./run_cnvs_test.sh [pcie|mlulink|memory]`。


### 矩阵乘性能测试

进入 `./cnvs` 目录，执行 `./run_matmul_test.sh`。
