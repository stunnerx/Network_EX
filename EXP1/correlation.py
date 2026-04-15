import matplotlib

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("E:/Code/Pycharm_Project/network_exp1/Concrete_Data_Yeh.csv")

# 计算相关性矩阵
corr = data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")

plt.tight_layout()
plt.savefig("vcorrelation_heatmap.png", dpi=300)
plt.close()

print("图像已保存到: E:/Code/Pycharm_Project/network_exp1/correlation_heatmap.png")