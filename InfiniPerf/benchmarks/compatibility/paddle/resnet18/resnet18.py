# This script assumes the dataset is located at "/data1/shared/Dataset/palm".
import os
import random
import cv2
import paddle
import argparse
import numpy as np
from paddle.optimizer import Adam
from paddle.nn import CrossEntropyLoss
from paddle.vision.models import resnet18
from paddle.vision.datasets import DatasetFolder
import paddle.nn.functional as F

# Set device to GPU
paddle.set_device('gpu')

# Image preprocessing function
def transform_img(img):
    img = cv2.resize(img, (224, 224))
    img = img.transpose((2, 0, 1))
    img = img.astype('float32')
    img = img / 255.0
    return img

# Training data loader
def data_loader(datadir, batch_size=10, mode='train'):
    filenames = [f for f in os.listdir(datadir) if f.endswith('.jpg')]

    def reader():
        if mode == 'train':
            random.shuffle(filenames)

        batch_imgs = []
        batch_labels = []

        for name in filenames:
            filepath = os.path.join(datadir, name)
            img = cv2.imread(filepath)
            if img is None:
                continue

            img = transform_img(img)

            if name[0] == 'H' or name[0] == 'N':
                label = 0  
            elif name[0] == 'P':
                label = 1  
            else:
                continue

            batch_imgs.append(img)
            batch_labels.append(label)

            if len(batch_imgs) == batch_size:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('int64')
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('int64')
            yield imgs_array, labels_array

    return reader

# Validation data loader
def valid_data_loader(datadir, csvfile, batch_size=10):
    with open(csvfile) as f:
        lines = f.readlines()[1:]  

    def reader():
        batch_imgs = []
        batch_labels = []

        for line in lines:
            parts = line.strip().split(',')
            name = parts[1]
            label = int(parts[2])

            filepath = os.path.join(datadir,"PALM-Validation400", name)
            img = cv2.imread(filepath)
            if img is None:
                continue

            img = transform_img(img)
            batch_imgs.append(img)
            batch_labels.append(label)

            if len(batch_imgs) == batch_size:
                imgs_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('int64')
                yield imgs_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            imgs_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('int64')
            yield imgs_array, labels_array

    return reader
# Define the Runner class
class GPURunner(object):
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.best_acc = 0
        
        # Move the model to GPU
        self.model.to('gpu')

    def save_model(self, save_path):
        """Save model parameters"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_path = os.path.join(save_path, 'best_model.pdparams')
        paddle.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        """Load model parameters"""
        state_dict = paddle.load(model_path)
        self.model.set_state_dict(state_dict)
        print(f"Model loaded from {model_path}")

    def train_pm(self, train_datadir, val_datadir, **kwargs):
        print('Start GPU training...')
        self.model.train()
        
        num_epochs = kwargs.get('num_epochs', 10)
        csv_file = kwargs.get('csv_file', None)
        save_path = kwargs.get("save_path", "./output/")
        
        train_loader = data_loader(train_datadir, batch_size=32, mode='train')
        
        for epoch in range(num_epochs):
            for batch_id, data in enumerate(train_loader()):
                x_data, y_data = data
                img = paddle.to_tensor(x_data).cuda()
                label = paddle.to_tensor(y_data).cuda()
                
                logits = self.model(img)
                avg_loss = self.loss_fn(logits, label)
                
                avg_loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()
                
                if batch_id % 20 == 0:
                    print(f"Epoch: {epoch}, Batch: {batch_id}, Loss: {float(avg_loss):.4f}")
            
            # Run evaluation after each epoch
            acc = self.evaluate_pm(val_datadir, csv_file)
            self.model.train()
            
            if acc > self.best_acc:
                self.save_model(save_path)
                self.best_acc = acc
    
    @paddle.no_grad()
    def evaluate_pm(self, val_datadir, csv_file):
        self.model.eval()
        accuracies = []
        losses = []
        valid_loader = valid_data_loader(val_datadir, csv_file, batch_size=32)
        
        for batch_id, data in enumerate(valid_loader()):
            x_data, y_data = data
            img = paddle.to_tensor(x_data).cuda()
            label = paddle.to_tensor(y_data).cuda()
            
            logits = self.model(img)
            pred = F.softmax(logits)
            loss = self.loss_fn(pred, label)
            label = paddle.unsqueeze(label, axis=1)
            acc = paddle.metric.accuracy(pred, label)
            
            accuracies.append(float(acc))
            losses.append(float(loss))        

        avg_acc = np.mean(accuracies)
        avg_loss = np.mean(losses)
        print(f"[Validation] Accuracy: {avg_acc:.4f}, Loss: {avg_loss:.4f}")
        return avg_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                       help='Mode: train or eval')
    args = parser.parse_args()

    if args.mode == 'train':
        # Training mode
        model = resnet18(num_classes=2)
        optimizer = Adam(learning_rate=0.001, parameters=model.parameters())
        loss_fn = CrossEntropyLoss()
        runner = GPURunner(model, optimizer, loss_fn)
    
        train_config = {
            'train_datadir': '/data1/shared/Dataset/palm',
            'val_datadir': '/data1/shared/Dataset/palm',
            'csv_file': '/data1/shared/Dataset/palm/PALM-Validation-GT/labels.csv',
            'num_epochs': 20,
            'save_path': './output/'
        }
        runner.train_pm(**train_config)
    else:
        # Evaluation mode
        evaluate_model()
def evaluate_model():
    # Initialize model
    model = resnet18(num_classes=2)

    optimizer = Adam(learning_rate=0.001, parameters=model.parameters())
    loss_fn = CrossEntropyLoss()

    runner = GPURunner(model, optimizer, loss_fn)

    model_path = './output/best_model.pdparams' 
    runner.load_model(model_path)

    eval_config = {
        'val_datadir': '/data1/shared/Dataset/palm',
        'csv_file': '/data1/shared/Dataset/palm/PALM-Validation-GT/labels.csv'
    }

    score = runner.evaluate_pm(eval_config['val_datadir'], eval_config['csv_file'])
    print(f"Final evaluation score: {score:.4f}")

if __name__ == '__main__':
    main()
