import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.vision.transforms as T
from paddle.io import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings # 导入 warnings 模块

# 过滤 BatchNorm 的特定 UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message="When training, we now always track global mean and variance.")

# --- 全局参数设定 ---
IMAGE_SIZE = 64  # 生成图像的尺寸 (高度和宽度)
CHANNELS_IMG = 1  # 图像通道数 (例如 MNIST 是 1, 彩色图是 3)
NOISE_DIM = 100  # 噪声向量的维度
EPOCHS = 5      # 为了快速演示，可以减少 epoch 数量，例如 5-10
BATCH_SIZE = 128
LR_G = 2e-4  # 生成器的学习率
LR_D = 2e-4  # 判别器的学习率
BETA1 = 0.5  # Adam 优化器的 beta1 参数
DEVICE = paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')

# --- 0. 自定义 make_grid 函数 ---
def make_grid_custom(tensor, nrow=8, padding=2, normalize=False, value_range=None, scale_each=False, pad_value=0.0):
    """
    将一批图像张量制作成网格图像。

    Args:
        tensor (paddle.Tensor): 4D mini-batch Tensor of shape (B x C x H x W).
        nrow (int): 每行显示的图像数量。
        padding (int): 图像之间的填充量。
        normalize (bool): 如果为True，则通过减去最小值并除以最大像素值将图像移至范围（0,1）。
        value_range (tuple): tuple (min, max)，其中min和max是数字，
                             然后将图像调整到此范围。默认情况下，从张量计算min和max。
        scale_each (bool): 如果为True，则分别缩放批次中的每个图像，
                           否则，共同缩放整个批次。
        pad_value (float): 填充区域的值。
    Returns:
        paddle.Tensor: 网格图像张量 (C x H_grid x W_grid)。
    """
    if not (isinstance(tensor, paddle.Tensor) and tensor.ndim == 4):
        raise TypeError(f"tensor is not a 4D paddle Tensor, got {type(tensor)} with ndim {tensor.ndim}")

    if normalize:
        tensor = tensor.clone()  # 避免原地修改张量
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) given"
            min_val, max_val = value_range
            tensor = paddle.clip(tensor, min_val, max_val)
            tensor = (tensor - min_val) / (max_val - min_val + 1e-5)
        else: # 自动从数据中获取范围
            if not scale_each:
                min_val = paddle.min(tensor)
                max_val = paddle.max(tensor)
            else:
                # 逐个图像缩放
                # Reshape to (B, C*H*W) to find min/max per image
                flat_tensor = tensor.reshape([tensor.shape[0], -1])
                min_val = paddle.min(flat_tensor, axis=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
                max_val = paddle.max(flat_tensor, axis=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)

            tensor = (tensor - min_val) / (max_val - min_val + 1e-5)
            tensor = paddle.clip(tensor, 0.0, 1.0) # 确保在 [0,1] 范围内

    # 计算网格的行数和列数
    num_images = tensor.shape[0]
    actual_nrow = min(nrow, num_images)
    ncol = -(-num_images // actual_nrow)  # 向上取整的列数

    channels, height, width = tensor.shape[1], tensor.shape[2], tensor.shape[3]

    # 计算网格图像的总高度和总宽度
    grid_height = ncol * height + padding * (ncol - 1)
    grid_width = actual_nrow * width + padding * (actual_nrow - 1)

    # 创建一个用 pad_value 填充的空网格
    grid = paddle.full((channels, grid_height, grid_width), fill_value=pad_value, dtype=tensor.dtype)

    # 将每个图像填充到网格中
    for i in range(num_images):
        row = i // actual_nrow
        col = i % actual_nrow

        start_h = row * (height + padding)
        end_h = start_h + height
        start_w = col * (width + padding)
        end_w = start_w + width

        grid[:, start_h:end_h, start_w:end_w] = tensor[i]

    return grid

# --- 1. 模型定义 ---
class Generator(nn.Layer):
    def __init__(self, noise_dim, channels_img, features_g=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(noise_dim, features_g * 8, 4, 1, 0),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            self._block(features_g * 2, features_g, 4, 2, 1),
            nn.Conv2DTranspose(
                features_g, channels_img, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2DTranspose(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias_attr=False,
            ),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Layer):
    def __init__(self, channels_img, features_d=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2D(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2D(features_d * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias_attr=False,
            ),
            nn.BatchNorm2D(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.net(x)

def initialize_weights(model):
    for m in model.sublayers():
        if isinstance(m, (nn.Conv2D, nn.Conv2DTranspose)):
            paddle.nn.initializer.Normal(mean=0.0, std=0.02)(m.weight)
        elif isinstance(m, nn.BatchNorm2D):
            paddle.nn.initializer.Normal(mean=1.0, std=0.02)(m.weight)
            paddle.nn.initializer.Constant(value=0.0)(m.bias)

# --- 2. 数据准备 ---
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5 for _ in range(CHANNELS_IMG)],
                std=[0.5 for _ in range(CHANNELS_IMG)])
])

# 使用 MNIST 数据集示例
# 注意: 如果 MNIST 数据集下载慢或失败，你可能需要手动下载并放到 `~/.cache/paddle/dataset/mnist/`
try:
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform, download=True)
except Exception as e:
    print(f"下载 MNIST 失败: {e}. 请确保网络连接或手动下载。")
    print("将尝试使用一个非常小的随机数据集进行演示。")
    # 创建一个非常小的虚拟数据集以允许脚本运行
    class DummyDataset(Dataset):
        def __init__(self, num_samples, channels, height, width, transform):
            self.num_samples = num_samples
            self.channels = channels
            self.height = height
            self.width = width
            self.transform = transform

        def __getitem__(self, idx):
            # 生成随机图像数据，模拟加载的图像 (未归一化的)
            # PIL Image.open().convert('L') or 'RGB'
            # ToTensor会将其从 HWC [0-255] uint8 -> CHW [0,1] float32
            # 这里我们直接生成类似 ToTensor 之后但 Normalize 之前的数据
            if self.channels == 1:
                img = np.random.rand(self.height, self.width, 1).astype('float32') * 255
                img = paddle.to_tensor(img).transpose([2,0,1]) / 255.0 # CHW, [0,1]
            else: # channels == 3
                img = np.random.rand(self.height, self.width, self.channels).astype('float32') * 255
                img = paddle.to_tensor(img).transpose([2,0,1]) / 255.0 # CHW, [0,1]

            # 应用 Normalize (因为 transform 通常包含 ToTensor 和 Normalize)
            # 为了简化，我们直接应用Normalize部分的效果
            img = (img - 0.5) / 0.5
            return img

        def __len__(self):
            return self.num_samples
    train_dataset = DummyDataset(BATCH_SIZE * 2, CHANNELS_IMG, IMAGE_SIZE, IMAGE_SIZE, transform)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# --- 3. 初始化模型、优化器和损失函数 ---
gen = Generator(NOISE_DIM, CHANNELS_IMG)
disc = Discriminator(CHANNELS_IMG)

initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(parameters=gen.parameters(), learning_rate=LR_G, beta1=BETA1, beta2=0.999)
opt_disc = optim.Adam(parameters=disc.parameters(), learning_rate=LR_D, beta1=BETA1, beta2=0.999)

criterion = nn.BCELoss()

fixed_noise = paddle.randn([64, NOISE_DIM, 1, 1])

if not os.path.exists("generated_images"):
    os.makedirs("generated_images")
if not os.path.exists("models"):
    os.makedirs("models")

# --- 4. 训练模型 ---
print(f"开始训练... 设备: {DEVICE}")
gen.train()
disc.train()

for epoch in range(EPOCHS):
    for batch_idx, real_images in enumerate(train_loader):
        if isinstance(real_images, list):
            real_images = real_images[0]
        batch_size_current = real_images.shape[0]

        # --- 训练判别器 ---
        disc.clear_gradients()
        label_real = paddle.full(shape=[batch_size_current, 1, 1, 1], fill_value=1.0, dtype='float32')
        output_real = disc(real_images)
        loss_d_real = criterion(output_real, label_real)
        loss_d_real.backward()

        noise = paddle.randn([batch_size_current, NOISE_DIM, 1, 1])
        fake_images_detached = gen(noise).detach() # 先生成并分离，判别器不应训练生成器
        label_fake = paddle.full(shape=[batch_size_current, 1, 1, 1], fill_value=0.0, dtype='float32')
        output_fake_d = disc(fake_images_detached)
        loss_d_fake = criterion(output_fake_d, label_fake)
        loss_d_fake.backward()

        loss_d = loss_d_real + loss_d_fake
        opt_disc.step()

        # --- 训练生成器 ---
        gen.clear_gradients()
        # 为了训练生成器，我们重新通过判别器传递生成的图像（不 detach）
        fake_images_for_g = gen(noise)
        output_fake_g = disc(fake_images_for_g)
        loss_g = criterion(output_fake_g, label_real) # 生成器希望判别器将其误判为真实图像
        loss_g.backward()
        opt_gen.step()

        if batch_idx % 100 == 0 or (len(train_loader) <= 100 and batch_idx % 10 == 0) : # 对于小数据集，更频繁地打印
            print(
                f"Epoch [{epoch+1}/{EPOCHS}] Batch {batch_idx}/{len(train_loader)} "
                f"Loss D: {loss_d.item():.4f} (Real: {loss_d_real.item():.4f}, Fake: {loss_d_fake.item():.4f}) "
                f"Loss G: {loss_g.item():.4f}"
            )

    # --- 每个 epoch 结束后进行一些操作 ---
    gen.eval()
    with paddle.no_grad():
        fake_samples = gen(fixed_noise).detach().cpu()
        # 将图像从 [-1, 1] 转换回 [0, 1] 以便显示/保存
        # 这个操作现在由 make_grid_custom 内部的 normalize=True 和自动value_range处理
        # fake_samples = (fake_samples * 0.5) + 0.5 # 如果make_grid不进行归一化，则需要此行

        # 使用自定义的 make_grid_custom
        # 注意：如果你的 fake_samples 输出已经是 [0,1]，则 normalize 可以设为 False
        # 但由于我们的 Generator 最后是 Tanh，输出是 [-1,1]，所以 normalize=True 是合适的
        grid = make_grid_custom(fake_samples, nrow=8, normalize=True, padding=2)

        grid_np = grid.transpose([1, 2, 0]).numpy()
        if CHANNELS_IMG == 1:
            plt.imsave(f"generated_images/epoch_{epoch+1}.png", grid_np.squeeze(), cmap='gray')
        else:
            plt.imsave(f"generated_images/epoch_{epoch+1}.png", grid_np)
    gen.train()

    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
        paddle.save(gen.state_dict(), f"models/generator_epoch_{epoch+1}.pdparams")
        paddle.save(disc.state_dict(), f"models/discriminator_epoch_{epoch+1}.pdparams")
        print(f"模型已保存到 models/ 文件夹 (Epoch {epoch+1})")

print("训练完成!")

# --- 5. "验证" (定性评估) ---
# 已在训练循环中通过 fixed_noise 生成并保存了图像。

# --- 6. 测试模型 (生成新图像) ---
print("\n开始测试 (生成图像)...")
gen.eval()
num_test_samples = 16
test_noise = paddle.randn([num_test_samples, NOISE_DIM, 1, 1])

with paddle.no_grad():
    generated_images_test = gen(test_noise).detach().cpu()
    # 使用 make_grid_custom 进行可视化
    grid_test = make_grid_custom(generated_images_test, nrow=4, normalize=True, padding=2)
    grid_test_np = grid_test.transpose([1,2,0]).numpy()


if not os.path.exists("test_results"):
    os.makedirs("test_results")

plt.figure(figsize=(8,8))
if CHANNELS_IMG == 1:
    plt.imshow(grid_test_np.squeeze(), cmap='gray')
else:
    plt.imshow(grid_test_np)
plt.axis('off')
plt.title("Generated test images")
plt.savefig("test_results/generated_test_images_grid.png")
print(f"测试生成的图像网格已保存到 test_results/generated_test_images_grid.png")
# plt.show() # 在某些环境可能会阻塞，按需取消注释

# 单独保存和显示几张图片 (可选)
# fig, axes = plt.subplots(4, 4, figsize=(8, 8))
# plt.suptitle("测试生成的单张图像", fontsize=16)
# generated_images_test_norm = (generated_images_test * 0.5) + 0.5 # 手动归一化到[0,1]如果需要
# for i, ax in enumerate(axes.flatten()):
#     if i >= num_test_samples:
#         break
#     img_tensor = generated_images_test_norm[i] # 使用上面手动归一化的
#     img_np = img_tensor.transpose([1, 2, 0]).numpy()
#     if CHANNELS_IMG == 1:
#         ax.imshow(img_np.squeeze(), cmap='gray')
#     else:
#         ax.imshow(img_np)
#     ax.axis("off")
# plt.savefig("test_results/generated_test_images_individual.png")
# plt.show()

print("脚本执行完毕。")

