import argparse
import os
import onnx
from pyinfinitensor.onnx import OnnxStub, backend
import numpy as np
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run model test and compare performance with pytorch."
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Path to the ONNX model file."
    )
    parser.add_argument(
        "--device", type=str, required=True, 
        choices=["cuda", "mlu", "npu", "kunlun"],
        help="Device type: cuda, mlu, npu, kunlun."
    )
    parser.add_argument(
        "--repeat", type=int, required=True, help="Repeat times to calculate average time."
    )

    args = parser.parse_args()
    print("arg setting: ", args)

    return args.model, args.device, args.repeat


def initInputs(model):
    inputs = []
    for input in model.graph.input:
        shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        data_type = input.type.tensor_type.elem_type
        input = np.random.random(shape).astype(toNumpyType(data_type))
        inputs.append(input)
    return inputs


def toNumpyType(typecode: int):
    type_map = {
        1: np.float32,
        2: np.uint8,
        3: np.int8,
        4: np.uint16,
        5: np.int16,
        6: np.int32,
        7: np.int64,
        8: np.string_,
        9: np.bool_,
        10: np.float16,
        11: np.double,
        12: np.uint32,
        13: np.uint64,
        14: np.uint16
    }
    if typecode not in type_map:
        raise RuntimeError("Unsupported data type.")
    return type_map[typecode]


def run_infinitensor(model, repeat, device):
    # Select runtime based on device
    if device == "cuda":
        runtime = backend.cuda_runtime()
    elif device == "mlu":
        runtime = backend.bang_runtime()
    elif device == "npu":
        runtime = backend.ascend_runtime()
    elif device == "kunlun":
        runtime = backend.kunlun_runtime()
    else:
        raise ValueError(f"Unsupported device: {device}")

    # Initialize stub
    stub = OnnxStub(model, runtime)
    
    # Warm up (with newly generated inputs each time)
    for _ in range(20):
        inputs = initInputs(model)  # Regenerate inputs for warmup
        for idata, itensor in zip(inputs, stub.inputs.items()):
            itensor[1].copyin_numpy(idata)
        stub.run()
    
    # Benchmark (with newly generated inputs each iteration)
    total_time = 0.0
    for _ in range(repeat):
        inputs = initInputs(model)  # Regenerate inputs for each run
        for idata, itensor in zip(inputs, stub.inputs.items()):
            itensor[1].copyin_numpy(idata)
        
        begin = time.time()
        stub.run()
        end = time.time()
        total_time += (end - begin)
    
    avg_time = total_time / repeat
    print(f"InfiniTensor ({device}): {avg_time * 1000:.3f} ms (with random inputs each run)")


def main():
    model_path, device, N = parse_args()
    model = onnx.load(model_path)
    run_infinitensor(model, N, device)


if __name__ == "__main__":
    main()
