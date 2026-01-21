# 导入Paddle及相关包
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
from paddle.nn import TransformerEncoderLayer, TransformerEncoder
import os
import numpy as np
from utils import Accuracy

def load_dataset(data_path):
    data_set = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            label, text = line.strip().split("\t", maxsplit=1)
            data_set.append((text, label))
    return data_set

# 加载词典
def load_dict(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        words = [word.strip() for word in f.readlines()]
        word_dict = dict(zip(words, range(len(words))))
    return word_dict

# 加载数据
path="./dataset"
train_data = load_dataset(os.path.join(path, "train.txt"))
dev_data = load_dataset(os.path.join(path, "dev.txt"))
test_data = load_dataset(os.path.join(path, "test.txt"))

# 加载词典
word2id_dict = load_dict(os.path.join(path, "vocab.txt"))
label2id_dict = load_dict(os.path.join(path, "label.txt"))

class THUCNewsDataset(Dataset):
    def __init__(self, data, word2id_dict, label2id_dict):
        # 词表
        self.word2id_dict = word2id_dict
        self.label2id_dict = label2id_dict
        # 数据
        self.examples = self.words_to_id(data)

    def __getitem__(self, idx):
        # 返回单条样本
        text, label =  self.examples[idx]
        return text, label

    def __len__(self):
        # 返回样本的个数
        return len(self.examples)

    def words_to_id(self, data):
        examples = []
        for example in data:
            text, label = example
            # text 转换成id的形式
            input_ids = [self.word2id_dict[item] if item in self.word2id_dict else self.word2id_dict['[UNK]'] for item in text]
            examples.append([input_ids, self.label2id_dict[label]])
        return examples

# 加载训练集
train_set = THUCNewsDataset(train_data, word2id_dict, label2id_dict)
# 加载验证集
dev_set = THUCNewsDataset(dev_data, word2id_dict, label2id_dict)
# 加载测试集
test_set = THUCNewsDataset(test_data, word2id_dict, label2id_dict)

def collate_fn(batch_data, pad_val=0, max_seq_len=512):
    input_ids_list, label_list = [], []
    max_len = 0
    for example in batch_data:
        input_ids, label = example
        # 对数据序列进行截断
        input_ids_list.append(input_ids[:max_seq_len])
        label_list.append(label)
        # 保存序列最大长度
        max_len = max(max_len, len(input_ids))
    # 对数据序列进行填充至最大长度
    for i in range(len(label_list)):
        input_ids_list[i] = input_ids_list[i]+[pad_val] * (max_len - len(input_ids_list[i]))
    return paddle.to_tensor(input_ids_list), paddle.to_tensor(label_list)

batch_size = 32
# 构建训练集、验证集和测试集的dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
dev_loader = DataLoader(dev_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

# 打印输出一条mini-batch的数据
for idx, item in enumerate(test_loader):
    if idx == 0:
        print(item)
        break

class WordEmbedding(nn.Layer):
    def __init__(self, vocab_size, emb_size, padding_idx=0):
        super(WordEmbedding, self).__init__()
        # Embedding的维度
        self.emb_size = emb_size
        # 使用随机正态（高斯）分布初始化 embedding
        self.word_embedding = nn.Embedding(vocab_size, emb_size,
            padding_idx=padding_idx, weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Normal(0.0, emb_size ** -0.5) ), )

    def forward(self, word):
        word_emb = self.emb_size ** 0.5 * self.word_embedding(word)
        return word_emb

# 定义三角函数，用于生成位置编码向量
def get_sinusoid_encoding(position_size, hidden_size):
    def cal_angle(pos, hidden_idx):
        return pos / np.power(10000, 2 * (hidden_idx // 2) / hidden_size)

    def get_posi_angle_vec(pos):
        return [cal_angle(pos, hidden_j) for hidden_j in range(hidden_size)]

    sinusoid = np.array([get_posi_angle_vec(pos_i) for pos_i in range(position_size)])
    # 偶数正弦
    sinusoid[:, 0::2] = np.sin(sinusoid[:, 0::2])
    # 奇数余弦
    sinusoid[:, 1::2] = np.cos(sinusoid[:, 1::2])
    return sinusoid.astype("float32")

class PositionalEmbedding(nn.Layer):
    def __init__(self, max_length,emb_size):
        super(PositionalEmbedding, self).__init__()
        self.emb_size = emb_size
        # 使用三角函数初始化Embedding
        self.pos_encoder = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=self.emb_size,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Assign(
                    get_sinusoid_encoding(max_length, self.emb_size))))

    def forward(self, pos):
        pos_emb = self.pos_encoder(pos)
        # 位置编码不需要进行梯度更新
        pos_emb.stop_gradient = True
        return pos_emb

class TransformerEmbedding(nn.Layer):
    """
    汇总token编码，位置编码
    """
    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        hidden_dropout_prob=0.1,
        position_size=512
        ):
        super(TransformerEmbedding, self).__init__()
        # token编码向量
        self.word_embeddings = WordEmbedding(vocab_size, hidden_size)
        # 位置编码向量
        self.position_embeddings = PositionalEmbedding(position_size, hidden_size)
        # 层规范化
        self.layer_norm = nn.LayerNorm(hidden_size)
        # Dropout操作
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, position_ids = None):
        if position_ids is None:
            # 初始化全1的向量，比如[1,1,1,1]
            ones = paddle.ones_like(input_ids, dtype="int64")
            # 累加输入,求出序列前K个的长度,比如[1,2,3,4]
            seq_length = paddle.cumsum(ones, axis=-1)
            # position id的形式： 比如[0,1,2,3]
            position_ids = seq_length - ones
            position_ids.stop_gradient = True
        # token编码
        input_embedings = self.word_embeddings(input_ids)
        # 位置编码
        position_embeddings = self.position_embeddings(position_ids)
        # 输入张量,位置张量进行叠加
        embeddings = input_embedings + position_embeddings
        # 层规范化
        embeddings = self.layer_norm(embeddings)
        # Dropout
        embeddings = self.dropout(embeddings)
        return embeddings

hidden_size = 768
heads_num = 12
intermediate_size = 3072

encoder_layer = TransformerEncoderLayer(hidden_size, heads_num, intermediate_size)
encoder = TransformerEncoder(encoder_layer, 1)

class ModelClassification(nn.Layer):
    def __init__(self, num_classes, vocab_size, hidden_size=768, heads_num=12, intermediate_size=3072, num_layers=1, padding_idx=0, hidden_dropout=0.1, position_size=512):
        super(ModelClassification, self).__init__()
        self.num_classes = num_classes
        self.padding_idx = padding_idx
        # 实例化Embedding层
        self.embedding = TransformerEmbedding(vocab_size, hidden_size=hidden_size, hidden_dropout_prob=hidden_dropout, position_size=position_size)
        encoder_layer = TransformerEncoderLayer(hidden_size, heads_num, intermediate_size)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        # 构建Mask矩阵，把Pad的位置即input_ids中为0的位置设置为True,非0的位置设置为False
        # input_ids: bs x seq_len
        # attention_mask: bs x 1 x 1 x seq_len
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.padding_idx).astype("float32") * -1e9, axis=[1, 2])
        # 抽取特征向量
        input_embbeding = self.embedding(input_ids=input_ids, position_ids=position_ids)
        sequence_output = self.encoder(input_embbeding, src_mask=attention_mask)

        # print(sequence_output.shape)
        pooled_output = paddle.mean(sequence_output, axis=1)
        # print(pooled_output.shape)
        logits = self.classifier(pooled_output)
        return logits

hidden_size = 768
heads_num = 12
intermediate_size = 3072
num_layers = 1
hidden_dropout = 0.1
position_size = 512

epochs = 3
learning_rate = 5E-5
weight_decay = 0.0
num_classes= len(label2id_dict)
vocab_size=len(word2id_dict)
padding_idx=word2id_dict['[PAD]']

# 实例化模型
model = ModelClassification(num_classes, vocab_size=vocab_size, hidden_size=hidden_size, heads_num=heads_num, intermediate_size=intermediate_size, num_layers=num_layers, hidden_dropout=hidden_dropout, position_size=position_size, padding_idx=padding_idx)

# 定义 Optimizer
decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
optimizer = paddle.optimizer.AdamW(
    learning_rate=learning_rate,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in decay_params)

# 评估的时候采用准确率指标
metric = Accuracy()

# 模型评估代码
def evaluate(data_loader, metric):
    model.eval()
    metric.reset()

    for batch_texts, batch_labels in data_loader:
        # 执行模型的前向计算
        logits = model(batch_texts)
        metric.update(logits, batch_labels)
    score = metric.accumulate()
    return score

def train(data_loader):
    global_step = 0
    best_score = 0.
    for epoch in range(epochs):
        model.train()
        for batch_texts, batch_labels in data_loader:
            # 执行模型的前向计算
            logits = model(batch_texts)
            # 计算损失
            losses = F.cross_entropy(input=logits, label=batch_labels, soft_label=False)
            loss = paddle.mean(losses)

            loss.backward()
            optimizer.step()
            optimizer.clear_gradients()

            if global_step % 200 == 0:
                print(f"Epoch: {epoch+1}/{epochs} - global_step: {global_step} - Loss: {loss.item()}")

            if global_step % 2000 == 0:
                dev_score = evaluate(dev_loader, metric)
                print(f"\n[Evaluate] dev score: {dev_score:.5f}")
                model.train()

                # 如果当前指标为最优指标，保存该模型
                if dev_score > best_score:
                    paddle.save(model.state_dict(), "./checkpoints/best.pdparams")
                    print(f"[Evaluate] best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                    best_score = dev_score
            global_step += 1
    print("[Train] Training done!")

train(train_loader)

model_path = "./checkpoints/best.pdparams"
best_state_dict = paddle.load(model_path)
model.set_state_dict(best_state_dict)

accuracy = evaluate(test_loader, metric)
print("Accuracy evaluated on test set is: ", accuracy)
