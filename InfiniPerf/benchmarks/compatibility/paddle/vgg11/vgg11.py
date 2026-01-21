import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from paddle.vision.transforms import Compose, Normalize, Resize, ToTensor
from paddle.io import DataLoader, Dataset
from paddle.vision import image_load

# Configuration parameters
train_parameters = {
    "data_path": "/data1/shared/Dataset/Chinese_Medicine_unzipped/Chinese_Medicine/",
    "input_size": [3, 224, 224],
    "class_dim": 5,              
    "num_epochs": 20,
    "skip_steps": 10,
    "save_steps": 100,
    "checkpoints": "/data1/shared/sunjinge/tmp/checkpoints",
    "learning_rate": 0.0001,
    "batch_size": 32,
    "num_workers": 2,
    "eval_ratio": 0.2
}

# Create directory for saving models
os.makedirs(train_parameters["checkpoints"], exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', help='train or eval mode')
args = parser.parse_args()

# Custom Dataset (instead of CIFAR-10)
class MedicineDataset(Dataset):
    def __init__(self, data_dir, mode='train',transform=None):
        super(MedicineDataset, self).__init__()
        self.data = []
        self.labels = []
        self.transform = transform
        self.class_to_idx = {}
        self.mode = mode 

        classes = sorted(os.listdir(data_dir))
        for idx, cls in enumerate(classes):
            self.class_to_idx[cls] = idx
            cls_path = os.path.join(data_dir, cls)
            img_names = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

            split_idx = int(len(img_names) * (1 - train_parameters["eval_ratio"]))
            if mode == 'train':
                selected_names = img_names[:split_idx]
            else:
                selected_names = img_names[split_idx:]

            for img_name in selected_names:
                self.data.append(os.path.join(cls_path, img_name))
                self.labels.append(idx)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = image_load(img_path)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, paddle.to_tensor(label, dtype='int64')

    def __len__(self):
        return len(self.data)


# Data loading and preprocessing
def get_dataloader(mode):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    dataset = MedicineDataset(
        train_parameters["data_path"], 
        mode=mode,
        transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=train_parameters['batch_size'],
        shuffle=(mode == 'train'),
        num_workers=train_parameters["num_workers"]
    )
    return loader


# Define convolution + pooling module
class ConvPool(nn.Layer):
    def __init__(self, num_channels, num_filters, filter_size, pool_size, pool_stride, groups,
                 conv_stride=1, conv_padding=1):
        super(ConvPool, self).__init__()
        self.block = nn.Sequential()
        for i in range(groups):
            self.block.add_sublayer(
                f'conv{i}', nn.Conv2D(
                    in_channels=num_channels,
                    out_channels=num_filters,
                    kernel_size=filter_size,
                    stride=conv_stride,
                    padding=conv_padding
                )
            )
            self.block.add_sublayer(f'relu{i}', nn.ReLU())
            num_channels = num_filters
        self.block.add_sublayer(
            'pool', nn.MaxPool2D(kernel_size=pool_size, stride=pool_stride)
        )

    def forward(self, x):
        return self.block(x)


# VGGNet architecture
class VGGNet(nn.Layer):
    def __init__(self, num_classes=5):
        super(VGGNet, self).__init__()
        self.convpool01 = ConvPool(3, 64, 3, 2, 2, 2)
        self.convpool02 = ConvPool(64, 128, 3, 2, 2, 2)
        self.convpool03 = ConvPool(128, 256, 3, 2, 2, 3)
        self.convpool04 = ConvPool(256, 512, 3, 2, 2, 3)
        self.convpool05 = ConvPool(512, 512, 3, 2, 2, 3)
        self.flatten_size = 512 * 7 * 7

        self.fc1 = nn.Linear(self.flatten_size, 4096)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x, label=None):
        x = self.convpool01(x)
        x = self.convpool02(x)
        x = self.convpool03(x)
        x = self.convpool04(x)
        x = self.convpool05(x)

        x = paddle.reshape(x, shape=[-1, self.flatten_size])
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        out = self.fc3(x)

        if label is not None:
            acc = paddle.metric.accuracy(out, paddle.unsqueeze(label, axis=1))
            return out, acc
        else:
            return out

# Training main loop
def train():
    model = VGGNet(num_classes=train_parameters['class_dim'])
    model.train()

    train_loader = get_dataloader('train')
    eval_loader = get_dataloader('eval')
    criterion = nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.Adam(learning_rate=train_parameters["learning_rate"],
                                      parameters=model.parameters())

    steps = 0
    Iters, total_loss, total_acc = [], [], []
    best_acc = 0.0

    for epoch in range(train_parameters["num_epochs"]):
        model.train()
        for batch_id, (x_data, y_data) in enumerate(train_loader):
            steps += 1
            logits, acc = model(x_data, y_data)
            loss = criterion(logits, y_data)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            if steps % train_parameters["skip_steps"] == 0:
                Iters.append(steps)
                total_loss.append(float(loss))
                total_acc.append(float(acc))
                print(f"Epoch {epoch}, Step {steps}, Loss: {float(loss):.4f}, Acc: {float(acc):.4f}")

            if steps % train_parameters["save_steps"] == 0:
                current_acc = evaluate(model, eval_loader, criterion)
                if current_acc > best_acc:
                    best_acc = current_acc
                    save_path = os.path.join(train_parameters["checkpoints"], "vgg_best.pdparams")
                    paddle.save(model.state_dict(), save_path)
                    print(f"New best model saved to: {save_path}")
                save_path = os.path.join(train_parameters["checkpoints"], f"vgg_step_{steps}.pdparams")
                paddle.save(model.state_dict(), save_path)
                print(f"Saved model to: {save_path}")
                model.train()

    # Save final model
    eval_loader = get_dataloader('eval')
    criterion = nn.CrossEntropyLoss()
    final_acc = evaluate(model, eval_loader, criterion)
    final_path = os.path.join(train_parameters["checkpoints"], "vgg_final.pdparams")
    paddle.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final accuracy: {final_acc:.4f}")
    print(f"Final model saved to: {final_path}")

# Evaluation function
def evaluate(model, valid_loader, criterion):
    model.eval()
    accs = []
    losses = []
    with paddle.no_grad():
        for batch_id, (x, y) in enumerate(valid_loader):
            logits, acc = model(x, y)
            loss = criterion(logits, y)
            accs.append(float(acc)) 
            losses.append(float(loss))
    
    avg_acc = sum(accs) / len(accs)
    avg_loss = sum(losses) / len(losses)
    print(f"[Validation] Accuracy: {avg_acc:.4f}, Loss: {avg_loss:.4f}")
    return avg_acc

if __name__ == "__main__":
    if args.mode == 'train':
        print("Start GPU training...")
        train()

    elif args.mode == 'eval':
        print("Start evaluation...")

        model = VGGNet(num_classes=train_parameters['class_dim'])
        model_path = os.path.join(train_parameters["checkpoints"], "vgg_best.pdparams")
        if os.path.exists(model_path):
            model_state_dict = paddle.load(model_path)
            model.set_state_dict(model_state_dict)
            print(f"Loaded model from: {model_path}")
        else:
            print(f"Model file not found: {model_path}")
            exit(1)

        eval_loader = get_dataloader('eval')
        criterion = nn.CrossEntropyLoss()
        evaluate(model, eval_loader, criterion)
