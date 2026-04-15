import pandas as pd
import matplotlib
matplotlib.use('Agg')   # 保存图片，不弹窗

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 读取数据
data = pd.read_csv("E:/Code/Pycharm_Project/network_exp1/Concrete_Data_Yeh.csv")

# 前8列是输入特征，最后1列是输出
X = data.iloc[:, 0:8]
y = data.iloc[:, 8]

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 输出每个主成分的方差贡献率
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = explained_variance_ratio.cumsum()

print("各主成分方差贡献率：")
for i, ratio in enumerate(explained_variance_ratio, start=1):
    print(f"PC{i}: {ratio:.4f}")

print("\n累计方差贡献率：")
for i, ratio in enumerate(cumulative_variance_ratio, start=1):
    print(f"前{i}个主成分: {ratio:.4f}")

# 画累计解释方差图
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance Ratio")
plt.title("PCA Cumulative Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.savefig("E:/Code/Pycharm_Project/network_exp1/pca_cumulative_variance.png", dpi=300)
plt.close()

print("累计方差贡献率图已保存到: E:/Code/Pycharm_Project/network_exp1/pca_cumulative_variance.png")

# 画前两个主成分散点图
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=25)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Scatter Plot (PC1 vs PC2)")
plt.colorbar(scatter, label="Concrete Compressive Strength")
plt.tight_layout()
plt.savefig("E:/Code/Pycharm_Project/network_exp1/pca_scatter.png", dpi=300)
plt.close()

print("PCA散点图已保存到: E:/Code/Pycharm_Project/network_exp1/pca_scatter.png")

# 查看主成分载荷（每个主成分由哪些原始特征组成）
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i}" for i in range(1, 9)],
    index=X.columns
)

print("\n主成分载荷矩阵：")
print(loadings)