import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# 1. 读取数据
data = pd.read_csv("E:/Code/Pycharm_Project/network_exp1/Concrete_Data_Yeh.csv")

X = data.iloc[:, 0:8].values
y = data.iloc[:, 8].values.reshape(-1, 1)


# 2. 划分训练集 / 测试集
# 随机80%训练，20%测试
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)


# 3. 标准化
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# 转为 tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)


# 4. 构建神经网络
class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

model = RegressionNet()


# 5. 损失函数与优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)


# 6. 模型训练
epochs = 1000
train_loss_list = []
test_loss_list = []

for epoch in range(epochs):

    model.train()

    y_pred_train = model(X_train_tensor)
    train_loss = criterion(y_pred_train, y_train_tensor)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    train_loss_list.append(train_loss.item())

    # 计算测试集loss
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor)
        test_loss = criterion(y_pred_test, y_test_tensor)

    test_loss_list.append(test_loss.item())

    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Train Loss={train_loss.item():.4f}, Test Loss={test_loss.item():.4f}")


# 7. 测试集预测
model.eval()
with torch.no_grad():
    y_pred_test_scaled = model(X_test_tensor).numpy()

# 反标准化，恢复到原始MPa单位
y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled)
y_test_original = y_test


# 8. 评价指标
mse = mean_squared_error(y_test_original, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred_test)

print("\nNeural Network Results:")
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R^2  = {r2:.4f}")


# 9. Loss曲线
plt.figure(figsize=(8,5))

plt.plot(train_loss_list, label="Train Loss")
plt.plot(test_loss_list, label="Test Loss")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

plt.title("Training vs Test Loss")
plt.legend()

plt.tight_layout()

plt.savefig("E:/Code/Pycharm_Project/network_exp1/nn_loss_curve_v4.png", dpi=300)

plt.close()

print("Loss曲线已保存到: E:/Code/Pycharm_Project/network_exp1/nn_loss_curve_v4.png")


# 10. 真实值 vs 预测值
plt.figure(figsize=(7, 7))

# 散点图
plt.scatter(y_test_original, y_pred_test, alpha=0.7, label="Test Samples")

# 参考线 y = x
min_val = min(y_test_original.min(), y_pred_test.min())
max_val = max(y_test_original.max(), y_pred_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal: y = x")

#  拟合直线（测试集预测结果）
x_true = y_test_original.flatten()
y_pred = y_pred_test.flatten()

slope, intercept = np.polyfit(x_true, y_pred, 1)
fit_line = slope * x_true + intercept

# 为了画线更平滑，对 x 排序
sorted_idx = np.argsort(x_true)
plt.plot(
    x_true[sorted_idx],
    fit_line[sorted_idx],
    linestyle='-',
    linewidth=2,
    label=f"Fit: y={slope:.2f}x+{intercept:.2f}"
)

# 坐标轴与标题
plt.xlabel("True Strength (MPa)")
plt.ylabel("Predicted Strength (MPa)")
plt.title("Neural Network: True vs Predicted")


metrics_text = f"MSE = {mse:.4f}\nRMSE = {rmse:.4f}\nR² = {r2:.4f}"

plt.text(
    0.05, 0.95, metrics_text,
    transform=plt.gca().transAxes,   # 使用坐标轴比例坐标
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

plt.legend()
plt.tight_layout()
plt.savefig("E:/Code/Pycharm_Project/network_exp1/nn_true_vs_pred_v4.png", dpi=300)
plt.close()

print("真实值-预测值散点图已保存到: E:/Code/Pycharm_Project/network_exp1/nn_true_vs_pred_v4.png")
