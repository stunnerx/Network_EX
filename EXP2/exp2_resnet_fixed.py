# svhn_resnet.py
# -*- coding: utf-8 -*-

import os
import csv
import json
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models


# =========================
# 1. 固定随机种子
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 2. 日志输出函数
# =========================
def log_and_print(message, log_file=None):
    print(message)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


# =========================
# 3. 创建实验保存目录
# =========================
def make_run_dir(base_dir, run_name=None, batch_size=128, lr=1e-3, num_epochs=50, seed=42):
    os.makedirs(base_dir, exist_ok=True)

    if run_name is None or run_name.strip() == "":
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ResNet18_AdamW_bs{batch_size}_lr{lr}_ep{num_epochs}_seed{seed}_{time_str}"

    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    return run_dir, run_name


# =========================
# 4. 保存每轮指标到 CSV
# =========================
def save_metrics_csv(csv_path, epoch, train_loss, train_acc, test_loss, test_acc):
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
        writer.writerow([
            epoch,
            f"{train_loss:.6f}",
            f"{train_acc:.6f}",
            f"{test_loss:.6f}",
            f"{test_acc:.6f}"
        ])


# =========================
# 5. 保存参数配置
# =========================
def save_config(config_path, config_dict):
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)


# =========================
# 6. 读取 SVHN .mat 数据集
# =========================
class SVHNDataset(Dataset):
    def __init__(self, mat_path, transform=None):
        """
        SVHN 原始 .mat 中:
        X.shape = (32, 32, 3, N)
        y.shape = (N, 1)
        其中标签 10 表示数字 0
        """
        data = loadmat(mat_path)

        self.images = data["X"]    # (32, 32, 3, N)
        self.labels = data["y"].astype(np.int64).squeeze()

        # SVHN 中标签 10 表示数字 0
        self.labels[self.labels == 10] = 0

        # 转成 (N, 32, 32, 3)
        self.images = np.transpose(self.images, (3, 0, 1, 2))

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]   # (32, 32, 3)
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, label


# =========================
# 7. 定义 ResNet18 模型（适配 32x32 小图）
# =========================
class ResNet18SVHN(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18SVHN, self).__init__()

        # 不使用预训练权重
        self.backbone = models.resnet18(weights=None)

        # 适配 32x32 输入
        self.backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # 去掉最开始的 maxpool，避免过早下采样
        self.backbone.maxpool = nn.Identity()

        # 修改最终分类层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# =========================
# 8. 训练函数
# =========================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# =========================
# 9. 测试函数
# =========================
@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# =========================
# 10. 画图函数
# =========================
def plot_curves(train_losses, test_losses, train_accs, test_accs, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Test Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, test_accs, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train/Test Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()


# =========================
# 11. 主函数
# =========================
def main():
    # ===== 可改参数 =====
    seed = 42
    batch_size = 128
    lr = 1e-3
    num_epochs = 50
    weight_decay = 1e-3
    label_smoothing = 0.1

    train_mat_path = "./train_32x32.mat"
    test_mat_path = "./test_32x32.mat"

    # 根目录
    base_save_dir = "./Results_ResNet"

    # 手动命名；设为 None 则自动命名
    run_name = None
    # run_name = "SVHN_ResNet18_AdamW_LabelSmoothing_Aug"

    set_seed(seed)

    # ===== 创建设备 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    # ===== 创建实验文件夹 =====
    run_dir, final_run_name = make_run_dir(
        base_dir=base_save_dir,
        run_name=run_name,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        seed=seed
    )

    log_txt_path = os.path.join(run_dir, "training_log.txt")
    metrics_csv_path = os.path.join(run_dir, "metrics.csv")
    config_json_path = os.path.join(run_dir, "config.json")

    config_dict = {
        "model_name": "ResNet18SVHN",
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_epochs": num_epochs,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "optimizer": "AdamW",
        "train_mat_path": train_mat_path,
        "test_mat_path": test_mat_path,
        "device": str(device),
        "run_name": final_run_name,
        "run_dir": run_dir,
        "train_augmentation": {
            "RandomCrop": {"size": 32, "padding": 4},
            "RandomRotation": 10,
            "RandomAffine": {
                "degrees": 0,
                "translate": [0.1, 0.1],
                "scale": [0.9, 1.1]
            }
        }
    }
    save_config(config_json_path, config_dict)

    with open(log_txt_path, "w", encoding="utf-8") as log_file:
        log_and_print(f"Using device: {device}", log_file)
        log_and_print(f"Model: ResNet18SVHN", log_file)
        log_and_print(f"Optimizer: AdamW", log_file)
        log_and_print(f"Label smoothing: {label_smoothing}", log_file)
        log_and_print(f"Weight decay: {weight_decay}", log_file)
        log_and_print(f"Run name: {final_run_name}", log_file)
        log_and_print(f"Run directory: {run_dir}", log_file)

        # ===== 数据增强与预处理 =====
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # ===== 构建数据集 =====
        train_dataset = SVHNDataset(train_mat_path, transform=train_transform)
        test_dataset = SVHNDataset(test_mat_path, transform=test_transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory
        )

        log_and_print(f"Train samples: {len(train_dataset)}", log_file)
        log_and_print(f"Test samples: {len(test_dataset)}", log_file)

        # ===== 模型、损失函数、优化器 =====
        model = ResNet18SVHN(num_classes=10).to(device)

        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5
        )

        # ===== 记录指标 =====
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []

        best_test_acc = 0.0

        # ===== 开始训练 =====
        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            test_loss, test_acc = evaluate(
                model, test_loader, criterion, device
            )

            scheduler.step()

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            epoch_msg = (
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
            )
            log_and_print(epoch_msg, log_file)

            save_metrics_csv(
                metrics_csv_path,
                epoch + 1,
                train_loss,
                train_acc,
                test_loss,
                test_acc
            )

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))

        # ===== 保存最终模型 =====
        torch.save(model.state_dict(), os.path.join(run_dir, "final_model.pth"))

        # ===== 绘图 =====
        plot_curves(train_losses, test_losses, train_accs, test_accs, run_dir)

        log_and_print("", log_file)
        log_and_print(f"Best Test Accuracy: {best_test_acc:.4f}", log_file)
        log_and_print(f"模型和曲线已保存到: {run_dir}", log_file)


if __name__ == "__main__":
    main()