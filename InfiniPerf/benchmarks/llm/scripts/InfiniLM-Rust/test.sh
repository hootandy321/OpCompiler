cp test.py ../../InfiniLM-Rust/

cd ../../InfiniLM-Rust/

echo "开始单卡推理测试..."
python test.py "/fm9g-7B-sft-v0.0-F16.gguf" --gpus 1
echo "单卡推理测试完成。"

echo "开始多卡推理测试..."
python test.py "/fm9g-7B-sft-v0.0-F16.gguf" --gpus 2,3 5,6,7,8
echo "多卡推理测试完成。"

rm -rf test.py
cd -

