import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 读取数据
data = pd.read_csv("E:/Code/Pycharm_Project/network_exp1/Concrete_Data_Yeh.csv")

# 1 检查缺失值
print("Missing values:")
print(data.isnull().sum())

# 2 分离特征和标签
X = data.iloc[:,0:8]
y = data.iloc[:,8]

# 3 划分训练集和测试集
split_index = int(len(data)*0.8)

X_train = X.iloc[:split_index,:]
X_test = X.iloc[split_index:,:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# 4 特征标准化
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 线性回归模型
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 预测
y_pred = model.predict(X_test_scaled)

# 评价指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Results:")
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R^2  = {r2:.4f}")

# 输出回归系数
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print("\nRegression Coefficients:")
print(coef_df)

# 真实值 vs 预测值
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("True Strength (MPa)")
plt.ylabel("Predicted Strength (MPa)")
plt.title("Linear Regression: True vs Predicted")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.tight_layout()
plt.savefig("E:/Code/Pycharm_Project/network_exp1/linear_true_vs_pred.png", dpi=300)
plt.close()

print("真实值-预测值散点图已保存到: E:/Code/Pycharm_Project/network_exp1/linear_true_vs_pred.png")

# 残差图
residuals = y_test - y_pred
plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Strength (MPa)")
plt.ylabel("Residual")
plt.title("Linear Regression Residual Plot")
plt.tight_layout()
plt.savefig("E:/Code/Pycharm_Project/network_exp1/linear_residual_plot.png", dpi=300)
plt.close()

print("残差图已保存到: E:/Code/Pycharm_Project/network_exp1/linear_residual_plot.png")