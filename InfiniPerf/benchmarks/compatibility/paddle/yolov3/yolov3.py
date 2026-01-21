# This script assumes the dataset is located at "/data1/shared/Dataset/insects".
import os
import cv2
import paddle
import numpy as np
import xml.etree.ElementTree as ET
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset

# ==================== 1. 数据加载模块 ====================
class YOLODataset(Dataset):
    def __init__(self, data_dir, input_size=416):
        assert input_size % 32 == 0, "Input size must be divisible by 32"
        self.data_dir = data_dir
        self.input_size = input_size
        self.image_dir = os.path.join(data_dir, "images")
        self.annotation_dir = os.path.join(data_dir, "annotations/xmls")
        
        # 获取所有图像文件名
        self.image_files = [f.split('.')[0] for f in os.listdir(self.image_dir) if f.endswith('.jpeg')]
        print(f"Found {len(self.image_files)} images in {self.image_dir}")

        # 初始化类别名称到ID的映射
        self.class_names = ['Boerner', 'Leconte', 'Linnaeus', 'acuminatus', 'armandi', 'coleoptera', 'linnaeus']
        self.class_dict = {name: idx for idx, name in enumerate(self.class_names)}
        print("Class mapping:", self.class_dict)

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

    def __getitem__(self, idx):
        # 图像路径
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, f"{img_name}.jpeg")
        
        # 读取图像并预处理
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.transpose((2, 0, 1)).astype('float32') / 255.0

        # 标注路径
        xml_path = os.path.join(self.annotation_dir, f"{img_name}.xml")
        
        # 解析XML标注文件
        gt_boxes = []
        gt_labels = []
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text.strip()  
                if name not in self.class_dict:
                    raise ValueError(f"Unknown class name: {name}")

                class_id = self.class_dict[name]
                gt_labels.append(class_id)

                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text) / self.input_size
                ymin = float(bndbox.find('ymin').text) / self.input_size
                xmax = float(bndbox.find('xmax').text) / self.input_size
                ymax = float(bndbox.find('ymax').text) / self.input_size
                
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin

                gt_boxes.append([x_center, y_center, width, height])

        # 转换为Tensor
        gt_boxes_np = np.array(gt_boxes, dtype='float32') if gt_boxes else np.zeros((0, 4), dtype=np.float32)
        gt_labels_np = np.array(gt_labels, dtype='int32') if gt_labels else np.zeros((0,), dtype=np.int32)
        
        return paddle.to_tensor(img), paddle.to_tensor(gt_boxes_np), paddle.to_tensor(gt_labels_np)
    
    def __len__(self):
        return len(self.image_files)

# ==================== 2. 模型定义模块 ====================
class ConvBNLayer(nn.Layer):
    """卷积+BN+激活层"""
    def __init__(self, ch_in, ch_out, filter_size=3, stride=1, padding=0, act='leaky'):
        super().__init__()
        self.conv = nn.Conv2D(
            ch_in, ch_out, filter_size, stride, padding,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Normal(0., 0.02)),
            bias_attr=False)
        self.bn = nn.BatchNorm2D(
            ch_out,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(1.)),
            bias_attr=paddle.ParamAttr(initializer=nn.initializer.Constant(0.)))
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act == 'leaky':
            x = F.leaky_relu(x, 0.1)
        return x

class DownSample(nn.Layer):
    """下采样层（stride=2的卷积）"""
    def __init__(self, ch_in, ch_out, filter_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = ConvBNLayer(ch_in, ch_out, filter_size, stride, padding)

    def forward(self, x):
        return self.conv(x)

class BasicBlock(nn.Layer):
    """DarkNet基础块"""
    def __init__(self, ch_in, ch_out):
        super().__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out//2, filter_size=1, stride=1, padding=0)
        self.conv2 = ConvBNLayer(ch_out//2, ch_out, filter_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual

class DarkNet53(nn.Layer):
    """DarkNet53骨干网络"""
    def __init__(self):
        super().__init__()
        # 初始卷积
        self.conv0 = ConvBNLayer(3, 32, 3, stride=1, padding=1)
        self.downsample0 = DownSample(32, 64)
        
        # DarkNet53的5个阶段
        self.stages = nn.LayerList([
            self._make_stage(64, 128, num_blocks=1),   # stage1
            self._make_stage(128, 256, num_blocks=2),  # stage2
            self._make_stage(256, 512, num_blocks=8),  # stage3
            self._make_stage(512, 1024, num_blocks=8), # stage4
            self._make_stage(1024, 1024, num_blocks=4) # stage5
        ])
        
        # 下采样层
        self.downsamples = nn.LayerList([
            DownSample(64, 128),
            DownSample(128, 256),
            DownSample(256, 512),
            DownSample(512, 1024)
        ])

    def _make_stage(self, ch_in, ch_out, num_blocks):
        blocks = []
        blocks.append(DownSample(ch_in, ch_out))
        for _ in range(num_blocks):
            blocks.append(BasicBlock(ch_out, ch_out))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv0(x)       
        x = self.downsample0(x) 
    
        # stage1: [64,208,208] -> [128,104,104]
        x = self.stages[0](x)
    
        # stage2: [128,104,104] -> [256,52,52] (c3)
        x = self.stages[1](x)
        c3 = x
    
        # stage3: [256,52,52] -> [512,26,26] (c4)
        x = self.stages[2](x)
        c4 = x
    
        # stage4: [512,26,26] -> [1024,13,13] (c5)
        x = self.stages[3](x)
        c5 = x
    
        return [c3, c4, c5] 

class YOLOv3(nn.Layer):
    """YOLOv3完整模型"""
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = DarkNet53()
        self.num_classes = num_classes

        # 检测头
        self.yolo_blocks = nn.LayerList()
        self.route_blocks = nn.LayerList()
        self.output_blocks = nn.LayerList()
        self.feature_transforms = nn.LayerList()
        
        # 为三个尺度构建检测头
        self.yolo_blocks.append(nn.Sequential(
            ConvBNLayer(1024, 512, 1),
            ConvBNLayer(512, 1024, 3),
            ConvBNLayer(1024, 512, 1),
            ConvBNLayer(512, 1024, 3),
            ConvBNLayer(1024, 512, 1)
        ))
        self.output_blocks.append(nn.Conv2D(
            512, 3*(5 + num_classes), 1))
        
        # 第二个尺度
        self.feature_transforms.append(ConvBNLayer(512, 256, 1))
        self.yolo_blocks.append(nn.Sequential(
            ConvBNLayer(768, 256, 1),  
            ConvBNLayer(256, 512, 3),
            ConvBNLayer(512, 256, 1),
            ConvBNLayer(256, 512, 3),
            ConvBNLayer(512, 256, 1)
        ))
        self.output_blocks.append(nn.Conv2D(256, 3*(5+num_classes), 1))
        
        # 第三个尺度
        self.feature_transforms.append(ConvBNLayer(256, 128, 1))  
        self.yolo_blocks.append(nn.Sequential(
            ConvBNLayer(384, 128, 1),
            ConvBNLayer(128, 256, 3),
            ConvBNLayer(256, 128, 1),
            ConvBNLayer(128, 256, 3),
            ConvBNLayer(256, 128, 1)
        ))
        self.output_blocks.append(nn.Conv2D(
            128, 3*(5 + num_classes), 1))
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        # 获取骨干网络特征
        c3, c4, c5 = self.backbone(x)

        # 第一个尺度检测
        x = self.yolo_blocks[0](c5)
        output1 = self.output_blocks[0](x)
        
        # 第二个尺度检测
        x = self.feature_transforms[0](x)  
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if x.shape[2:] != c4.shape[2:]:
            c4 = F.interpolate(c4, size=x.shape[2:], mode='nearest')
        x = paddle.concat([x, c4], axis=1)  
        x = self.yolo_blocks[1](x)
        output2 = self.output_blocks[1](x)
        
        # 第三个尺度检测
        x = self.feature_transforms[1](x)  
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if x.shape[2:] != c3.shape[2:]:
            c3 = F.interpolate(c3, size=x.shape[2:], mode='nearest')
        x = paddle.concat([x, c3], axis=1)  
        x = self.yolo_blocks[2](x)
        output3 = self.output_blocks[2](x)
        
        return [output1, output2, output3]

# ==================== 3. 损失函数模块 ====================
def yolo_loss(preds, gt_boxes_list, gt_labels_list, num_classes, anchors, anchor_masks, ignore_thresh=0.7):
    loss = 0
    for i, pred in enumerate(preds):
        b, c, h, w = pred.shape
        pred = paddle.transpose(pred, [0, 2, 3, 1])  
        pred = paddle.reshape(pred, [b, h, w, len(anchor_masks[i]), -1])  

        pred_cls = pred[:, :, :, :, 5:]  
        pred_cls = paddle.reshape(pred_cls, [-1, num_classes])  

        # 使用每个 batch 的 gt_labels
        labels = gt_labels_list 
        if isinstance(labels, (list, np.ndarray)):
            labels = paddle.to_tensor(labels, dtype='int32')
        batch_size = labels.shape[0]
        fake_targets = paddle.zeros([pred_cls.shape[0], num_classes], dtype='float32')

        for batch_id in range(batch_size):
            if paddle.sum(labels[batch_id] != -1) > 0:
                class_id = int(float(labels[batch_id][0]))
                fake_targets[batch_id][class_id] = 1.0

        cls_loss = F.binary_cross_entropy_with_logits(pred_cls[:batch_size], fake_targets[:batch_size])
        loss += paddle.mean(cls_loss)

    return loss

# ==================== 4. 训练模块 ====================
def get_lr(base_lr=0.0001, lr_decay=0.1):
    bd = [10000, 20000]
    lr = [base_lr, base_lr * lr_decay, base_lr * lr_decay * lr_decay]
    return paddle.optimizer.lr.PiecewiseDecay(boundaries=bd, values=lr)

def yolo_collate_fn(batch, max_boxes=20):
    imgs = [item[0] for item in batch]
    boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    imgs = paddle.stack(imgs, axis=0)
    box_tensor = paddle.full([len(batch), max_boxes, 4], fill_value=-1.0, dtype='float32')
    label_tensor = paddle.full([len(batch), max_boxes], fill_value=-1, dtype='int32')

    for i in range(len(batch)):
        num_valid = min(boxes[i].shape[0], max_boxes)
        if num_valid > 0:
            box_tensor[i, :num_valid] = paddle.to_tensor(boxes[i][:num_valid])
            label_tensor[i, :num_valid] = paddle.to_tensor(labels[i][:num_valid])

    return imgs, box_tensor, label_tensor

def train():
    # 超参数
    MAX_EPOCH = 1
    ANCHORS = [10, 13, 16, 30, 33, 23,
               30, 61, 62, 45, 59, 119,
               116, 90, 156, 198, 373, 326]
    ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    IGNORE_THRESH = 0.7
    NUM_CLASSES = 7  # 类别数

    model = YOLOv3(num_classes=NUM_CLASSES)
    train_dataset = YOLODataset('/data1/shared/Dataset/insects/train')
    train_loader = paddle.io.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=yolo_collate_fn
    )

    scheduler = get_lr()
    optimizer = paddle.optimizer.Momentum(
        learning_rate=scheduler,
        momentum=0.9,
        parameters=model.parameters()
    )

    for epoch in range(MAX_EPOCH):
        model.train()
        for batch_id, (img, gt_boxes, gt_labels) in enumerate(train_loader):
            outputs = model(img)

            # 拆分 batch 维度为 list
            gt_boxes_list = [gt_boxes[i] for i in range(gt_boxes.shape[0])]
            gt_labels_list = [gt_labels[i] for i in range(gt_labels.shape[0])]

            # 多尺度复制
            gt_boxes_list = [paddle.to_tensor(gt_boxes[i]) for i in range(gt_boxes.shape[0])]
            gt_labels_list = [paddle.to_tensor(gt_labels[i]) for i in range(gt_labels.shape[0])]

            loss = yolo_loss(
                outputs,
                gt_boxes_list,
                gt_labels_list,
                NUM_CLASSES,
                anchors=ANCHORS,
                anchor_masks=ANCHOR_MASKS,
                ignore_thresh=IGNORE_THRESH
            )

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            if batch_id % 10 == 0:
                print(f'[Epoch {epoch}] Batch {batch_id}, Loss: {float(loss):.4f}')

        paddle.save(model.state_dict(), f'yolo_epoch{epoch}.pdparams')

# ==================== 主程序入口 ====================
if __name__ == '__main__':
    paddle.set_device('gpu')  # or 'cpu'
    train()
