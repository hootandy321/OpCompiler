# 1. 标准库导入
from collections import defaultdict
from functools import partial
import os
import random

# 2. 第三方库导入
import jieba
import numpy as np
import pandas as pd

# 3. PaddlePaddle 及 PaddleNLP 相关导入
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Dropout, Embedding, Linear, LSTM 

import paddlenlp as ppnlp 
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab 
from paddlenlp.datasets import load_dataset
from paddlenlp.utils.downloader import get_path_from_url


URL = "https://bj.bcebos.com/paddlenlp/datasets/ChnSentiCorp.zip"
# 如果数据集不存在，就下载数据集并解压
if(not os.path.exists('ChnSentiCorp.zip')):
    get_path_from_url(URL,root_dir='.')

def read(split='train'):
    data_dict={'train':'ChnSentiCorp/ChnSentiCorp/train.tsv',
               "dev":'ChnSentiCorp/ChnSentiCorp/dev.tsv',
               'test':'ChnSentiCorp/ChnSentiCorp/test.tsv'}
    with open(data_dict[split],'r', encoding='utf-8') as f:
        head = None
        # 一行一行的读取数据
        for line in f.readlines():
            data = line.strip().split("\t")
            # 跳过第一行，因为第一行是列名
            if not head:
                head = data
            else:
                # 从第二行还是一行一行的返回数据
                if split == 'train':
                    if len(data) == 2:
                        label, text = data
                        yield {"text": text, "label": label, "qid": ''}
                elif split == 'dev':
                    if len(data) == 3:
                        qid, label, text = data
                        yield {"text": text, "label": label, "qid": qid}
                elif split == 'test':
                    if len(data) == 2: # 测试集通常是 qid, text
                        qid, text = data
                        yield {"text": text, "label": '', "qid": qid} # label 为空字符串
                    elif len(data) == 1: # 有时测试集可能只有 text
                        text = data[0]
                        yield {"text": text, "label": '', "qid": ''} # qid 也可能为空


train_ds= load_dataset(read, split="train",lazy=False)
dev_ds= load_dataset(read, split="dev",lazy=False)
test_ds= load_dataset(read, split="test",lazy=False)

for data in train_ds.data[:5]:
    print(data)

def build_vocab(texts,
                stopwords=[],
                num_words=None,
                min_freq=10,
                unk_token="[UNK]",
                pad_token="[PAD]"):
    word_counts = defaultdict(int)
    for text in texts:
        if not text:
            continue
        # # 统计词频
        for word in jieba.cut(text):
            if word in stopwords:
                continue
            word_counts[word] += 1
    # 过滤掉词频小于min_freq的单词
    wcounts = []
    for word, count in word_counts.items():
        if count < min_freq:
            continue
        wcounts.append((word, count))
    # 把单词按照词频从大到小进行排序
    wcounts.sort(key=lambda x: x[1], reverse=True)
    # 把对齐的字符和unk字符加入到词表中
    if num_words is not None and len(wcounts) > (num_words - 2):
        wcounts = wcounts[:(num_words - 2)]
    # pad字符和unk字符
    sorted_voc = [pad_token, unk_token]
    sorted_voc.extend(wc[0] for wc in wcounts)
    # 给每个单词一个编号
    word_index = dict(zip(sorted_voc, list(range(len(sorted_voc)))))
    return word_index

texts = []
for data in train_ds:
    texts.append(data["text"])
for data in dev_ds:
    texts.append(data["text"])

# 以下停用词仅用作示例，具体停用词的选择需要根据具体语料库调整。
stopwords = set(["的", "吗", "吧", "呀", "呜", "呢", "呗"])
# 构建词汇表
word2idx = build_vocab(
    texts, stopwords, min_freq=5, unk_token="[UNK]", pad_token="[PAD]")
vocab = Vocab.from_dict(word2idx, unk_token="[UNK]", pad_token="[PAD]")
# 保存词汇表
# 确保 checkpoint 目录存在
os.makedirs("./checkpoint", exist_ok=True)
res=vocab.to_json("./checkpoint/vocab.json") 

def get_idx_from_word(word, word_to_idx, unk_word_idx): # 修改unk_word为unk_word_idx
    if word in word_to_idx:
        return word_to_idx[word]
    return unk_word_idx # 直接返回unk_word的索引

# 把词汇表加载到结巴分词器中
tokenizer = JiebaTokenizer(vocab)
text='选择珠江花园的原因就是方便，有电动扶梯直接到达海边，周围餐馆、食廊、商场、超市、摊位一应俱全。'
segment_text=tokenizer.cut(text)
print("分词后的文本:{}".format(segment_text))
input_ids_manual = vocab.to_indices(segment_text)
print("把分词后的文本转换成id (manual):{}".format(input_ids_manual))

input_ids_encoded = tokenizer.encode(text) # tokenizer.encode 返回的是一个包含 token_ids 和 seq_len 的字典
print("encode 编码后的id: {}".format(input_ids_encoded))

# 把语料转换为id序列
def convert_example(example, tokenizer, is_test=False):

    input_ids = tokenizer.encode(example["text"])

    # 计算出数据转换成id后的长度，并转换成numpy的格式
    valid_length = np.array(len(input_ids), dtype='int64')
    # 把id形式的数据转换成numpy的形式
    input_ids = np.array(input_ids, dtype='int64')
    if not is_test:
        label = np.array(example["label"], dtype="int64")
        return input_ids, valid_length, label
    else:
        return input_ids, valid_length

# partial函数的意思是把tokenizer=tokenizer, is_test=False赋值给当前的convert_example函数
trans_fn_train_dev = partial(convert_example, tokenizer=tokenizer, is_test=False)
trans_fn_test = partial(convert_example, tokenizer=tokenizer, is_test=True)

# 训练数据转换成id的形式
train_ds_processed = train_ds.map(trans_fn_train_dev)
# 验证集转换成id的形式
dev_ds_processed = dev_ds.map(trans_fn_train_dev)


# 构建a，b，c三个向量
a = [1, 2, 3, 4]
b = [3, 4, 5, 6]
c = [5, 6, 7, 8]
result = Stack()([a, b, c])
print("堆叠（Stacked）数据后 : \n", result)
print()

# 构建a，b，c三个向量
a = [1, 2, 3, 4]
b = [5, 6, 7]
c = [8, 9]
result = Pad(pad_val=0)([a, b, c])
print("对齐（Padded）数据后: \n", result)
print()

# 构造一个小的样本，包含输入id和label
data = [
        [[1, 2, 3, 4], [1]], # label 应该是单个值，而不是列表，除非是多标签
        [[5, 6, 7], [0]],
        [[8, 9], [1]],
       ]
batchify_fn_example = Tuple(Pad(pad_val=0), Stack())
corrected_data = [ (item[0], item[1][0]) for item in data]
ids, labels = batchify_fn_example(corrected_data)
print("id的输出: \n", ids)
print()
print("标签的输出: \n", labels)
print()

batch_size = 64 # 在前面定义过，这里可以不用重复定义
# batchify_fn 现在应该处理 convert_example 返回的 (input_ids, valid_length, label)
# valid_length 也需要被 Stack
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=vocab.token_to_idx.get('[PAD]', 0)), # input_ids
    Stack(dtype="int64"),  # valid_length
    Stack(dtype="int64")   # label
): fn(samples) # 这里不需要 [data for data in fn(samples)]，fn(samples) 直接就是列表


# 训练集的sampler，迭代式获取mini-batch的样本下标数组，数组长度与 batch_size 一致
train_sampler = paddle.io.BatchSampler(
        dataset=train_ds_processed, batch_size=batch_size, shuffle=True)
# 测试集的sampler，迭代式获取mini-batch的样本下标数组，数组长度与 batch_size 一致
dev_sampler = paddle.io.BatchSampler( # Renamed from test_sampler to dev_sampler for clarity
        dataset=dev_ds_processed, batch_size=batch_size, shuffle=False) # shuffle=False for dev/test

# 使用paddle.io.DataLoader接口多线程异步加载数据
# DataLoader根据 batch_sampler 给定的顺序迭代一次给定的 dataset
train_loader = paddle.io.DataLoader(
        train_ds_processed, batch_sampler=train_sampler, collate_fn=batchify_fn, return_list=True)
# 使用验证集作为测试集，因为验证集包含label。而原来的测试集没有label，不方便算指标
dev_loader = paddle.io.DataLoader( # Renamed from test_loader to dev_loader
        dev_ds_processed, batch_sampler=dev_sampler, collate_fn=batchify_fn, return_list=True)

# 打印输出一个mini-batch的数据
for idx,item in enumerate(train_loader):
    if(idx==0):
        print("Sample batch from train_loader:")
        print(f"Input IDs shape: {item[0].shape}")
        print(f"Valid lengths shape: {item[1].shape}")
        print(f"Labels shape: {item[2].shape}")
        # print(item) # 打印整个item可能会很长
        break


# LSTMModel 使用 paddlenlp.seq2vec.LSTMEncoder
class LSTMModel(nn.Layer):
    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=128,
                 padding_idx=0,
                 lstm_hidden_size=198,
                 direction='forward', # PaddleNLP LSTMEncoder uses 'forward', 'backward', 'bidirectional'
                 lstm_layers=1,
                 dropout_rate=0.0,
                 # pooling_type=None, # ppnlp.seq2vec.LSTMEncoder has internal pooling
                 fc_hidden_size=96):
        super().__init__()
        # 文本向量化
        # 首先将输入word id 查表后映射成 word embedding
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=padding_idx)
        # 序列学习
        # 将word embedding经过LSTMEncoder变换到文本语义表征空间中
        self.lstm_encoder = ppnlp.seq2vec.LSTMEncoder( # 使用 ppnlp.seq2vec.LSTMEncoder
            emb_dim,
            lstm_hidden_size,
            num_layers=lstm_layers,
            direction=direction,
            dropout=dropout_rate,
            # pooling_type is handled internally by ppnlp.seq2vec.LSTMEncoder
            # default pooling_type is 'last' which takes the output of the last time step
        )
        # 特征学习
        # LSTMEncoder.get_output_dim()方法可以获取经过encoder之后的文本表示hidden_size
        self.fc = nn.Linear(self.lstm_encoder.get_output_dim(), fc_hidden_size)
        # 输出层
        # 最后的分类器
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    # forwad函数即为模型前向计算的函数，它有两个输入，分别为：
    # input为输入的训练文本，其shape为[batch_size, max_seq_len]
    # seq_len训练文本对应的真实长度，其shape维[batch_size]
    def forward(self, text, seq_len):
        # 输入的文本的维度(batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # lstm的输出的维度: (batch_size, num_tokens, num_directions*lstm_hidden_size)
        # 如果lstm是双向的，则num_directions = 2，如果是单向的则num_directions的维度是1
        text_repr = self.lstm_encoder(embedded_text, sequence_length=seq_len)
        # 全连接层的的维度是(batch_size, fc_hidden_size)
        fc_out = paddle.tanh(self.fc(text_repr))
        # 输出层的维度(batch_size, num_classes)
        logits = self.output_layer(fc_out)
        # probs 分类概率值 (Softmax 通常在计算损失函数时由 F.cross_entropy 内部处理，或者在推理时显式调用)
        # 在训练时，模型通常直接返回 logits
        return logits # 返回 logits 更常见，softmax 在损失函数或评估时处理

# 定义训练参数
epoch_num = 4
learning_rate = 5e-5
# dropout_rate = 0.2 # 这个应该传递给模型定义
num_layers = 1 # LSTM 层数
lstm_hidden_size_param = 256 # LSTM 隐层大小 
embedding_size = 256
vocab_size=len(vocab)
print(f"Vocab size: {vocab_size}")

# 实例化LSTM模型
model= LSTMModel(
        vocab_size,
        num_classes=2,
        emb_dim=embedding_size,
        lstm_layers=num_layers,
        lstm_hidden_size=lstm_hidden_size_param, # 使用上面定义的参数
        direction='bidirectional', # paddlepaddle 中是 'bidirectional'
        dropout_rate=0.2, # 将 dropout_rate 传递给模型
        padding_idx=vocab.token_to_idx.get('[PAD]', 0) # 使用 vocab 获取 padding_idx
)
# 指定优化策略，更新模型参数
optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, beta1=0.9, beta2=0.999, parameters= model.parameters())

paddle.seed(0)
random.seed(0)
np.random.seed(0)
# 定义训练函数
# 记录训练过程中的损失变化情况，可用于后续画图查看训练情况
losses = []
steps_log = [] # Renamed from steps to avoid conflict with step in loop

def train_model(model_to_train, opt): # Renamed train to train_model to avoid conflict
    # 开启模型训练模式
    model_to_train.train()
    
    global_step =  0
    for epoch in range(epoch_num):
        for step, (ids_batch, len_batch, labels_batch) in enumerate(train_loader):
            
            # 前向计算，将数据feed进模型，并得到预测的情感标签和损失
            logits = model_to_train(ids_batch, len_batch)

            # 计算损失
            loss = F.cross_entropy(input=logits, label=labels_batch) # labels_batch 应该是1D的
            loss = paddle.mean(loss)

            # 后向传播
            loss.backward()
            # 更新参数
            opt.step()
            # 清除梯度
            opt.clear_grad()
            
            global_step+=1

            if global_step % 100 == 0:
                # 记录当前步骤的loss变化情况
                losses.append(loss.numpy().item()) # Use .item() to get scalar
                steps_log.append(global_step) # Log global_step for better x-axis on plot
                # 打印当前loss数值
                print("epoch %d, global_step %d, loss %.3f" % (epoch,global_step, loss.numpy().item()))

#训练模型
train_model(model, optimizer)
# 保存模型，包含两部分：模型参数和优化器参数
model_name = "sentiment_classifier"
# 保存训练好的模型参数
paddle.save(model.state_dict(), "checkpoint/{}.pdparams".format(model_name))
# 保存优化器参数，方便后续模型继续训练
paddle.save(optimizer.state_dict(), "checkpoint/{}.pdopt".format(model_name))

@paddle.no_grad()
def evaluate_model(model_to_eval): 
    # 开启模型测试模式，在该模式下，网络不会进行梯度更新
    model_to_eval.eval()

    # 定义以上几个统计指标
    tp, tn, fp, fn = 0, 0, 0, 0
    
    for ids_batch, len_batch, labels_batch in dev_loader: # Use dev_loader
    
        # 获取模型对当前batch的输出结果
        logits = model_to_eval(ids_batch,len_batch)
        
        # 使用softmax进行归一化
        probs = F.softmax(logits, axis=-1) # Apply softmax here for probability interpretation

        # 把输出结果转换为numpy array数组，比较预测结果和对应label之间的关系，并更新tp，tn，fp和fn
        preds = paddle.argmax(probs, axis=1) # Get predicted class indices
        
        preds_np = preds.numpy()
        labels_np = labels_batch.numpy()

        for i in range(len(preds_np)):
            # 当样本是的真实标签是正例 (label=1)
            if labels_np[i] == 1:
                # 模型预测是正例
                if preds_np[i] == 1:
                    tp += 1
                # 模型预测是负例
                else:
                    fn += 1
            # 当样本的真实标签是负例 (label=0)
            else:
                # 模型预测是正例
                if preds_np[i] == 1:
                    fp += 1
                # 模型预测是负例
                else:
                    tn += 1

    # 整体准确率
    total_samples = tp + tn + fp + fn
    if total_samples == 0: # Avoid division by zero if dev_loader is empty
        accuracy = 0.0
    else:
        accuracy = (tp + tn) / total_samples
    
    # 输出最终评估的模型效果
    print("TP: {}\nFP: {}\nTN: {}\nFN: {}\n".format(tp, fp, tn, fn))
    print("Accuracy: %.4f" % accuracy)

# 加载训练好的模型进行预测，重新实例化一个模型，然后将训练好的模型参数加载到新模型里面
state_dict=paddle.load('checkpoint/sentiment_classifier.pdparams')
model.load_dict(state_dict) # Load into the existing model instance
# 评估模型
evaluate_model(model)

label_map = {0: 'negative', 1: 'positive'}

# 从原始的 test_ds（未处理的）中取一个样本进行预测演示
if len(test_ds) > 0:
    # 假设 test_ds.data[0] 存在并且是 {"text": "some text", "label": '', "qid": ''} 格式
    sample_test_data = test_ds.data[0]
    text_to_predict = sample_test_data['text']
    print(f"\nPredicting on new sample from test_ds: '{text_to_predict}'")

    # 文本转换成ID的形式
    encoded_input = tokenizer.encode(text_to_predict)
    input_ids_predict = encoded_input
    valid_len_predict = len(input_ids_predict)

    # 转换成Tensor的形式
    input_ids_tensor = paddle.to_tensor([input_ids_predict], dtype='int64') # Batch of 1
    valid_len_tensor = paddle.to_tensor([valid_len_predict], dtype='int64') # Batch of 1

    # 模型预测
    model.eval() 
    with paddle.no_grad():
        logits_predict = model(input_ids_tensor, valid_len_tensor)
        probs_predict = F.softmax(logits_predict, axis=-1)

    # 取概率最大值的ID
    idx_predict = paddle.argmax(probs_predict, axis=-1).numpy().item() # Get single item
    
    # 得到预测标签
    predicted_label_str = label_map[idx_predict]
    
    # 看看预测样例分类结果
    print('Data: {} \t Predicted Label: {}'.format(text_to_predict, predicted_label_str))
else:
    print("Test dataset (test_ds) is empty, skipping single prediction example.")
