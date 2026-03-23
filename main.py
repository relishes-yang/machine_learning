# main.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from model_utils import generate_data, train_models

# === 修复中文乱码的关键代码 ===
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端，避免乱码
import matplotlib.font_manager as fm

# 尝试加载中文字体（支持Windows/Mac/Linux）
try:
    # Windows系统常用中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    # 如果系统没有中文字体，使用英文
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = True

# === 以下为原有代码 ===

st.title("岭回归 vs 稀疏回归：系数差异对比")

st.sidebar.header("参数调整")
n_samples = st.sidebar.slider("样本数量", 50, 500, 200)
n_features = st.sidebar.slider("总特征数", 5, 20, 10)
n_informative = st.sidebar.slider("重要特征数", 1, 5, 2)
noise = st.sidebar.slider("噪声强度", 5, 50, 10)
alpha = st.sidebar.slider("正则化强度", 0.1, 10.0, 1.0, 0.1)

X, y, true_coef = generate_data(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    noise=noise
)
results = train_models(X, y, alpha=alpha)

st.header("关键发现：系数差异")
st.write("""
- **岭回归（Ridge）**：所有系数都变小了，但**不会变成0**
- **稀疏回归（Lasso）**：很多系数**直接变成0**，只保留重要特征
""")

# === 修复后的图表代码 ===
fig, ax = plt.subplots(figsize=(10, 4))

# 真实系数（用黑色显示）
ax.stem(true_coef, linefmt='k-', markerfmt='ko', basefmt=' ', label='真实系数')

# Ridge系数（蓝色）
ax.stem(
    results["Ridge (L2)"]["coef"],
    linefmt='b-', markerfmt='bo', basefmt=' ',
    label='岭回归系数',

)

# Lasso系数（红色）
ax.stem(
    results["Lasso (Sparse/L1)"]["coef"],
    linefmt='r-', markerfmt='ro', basefmt=' ',
    label='稀疏回归系数',

)

# === 修复标签重叠问题（使用相对偏移量）===
# 1. 添加真实系数的数值标签（向上偏移0.3个单位）
for i, coef in enumerate(true_coef):
    ax.text(i, coef + 5, f'{coef:.2f}',
            ha='center', va='bottom',  # 保持底部对齐
            color='black', fontsize=8,
            alpha=0.9)

# 2. 添加岭回归系数的数值标签（向上偏移0.15个单位）
for i, coef in enumerate(results["Ridge (L2)"]["coef"]):
    ax.text(i, coef + 3, f'{coef:.2f}',
            ha='center', va='bottom',
            color='blue', fontsize=8,
            alpha=0.9)

# 3. 添加稀疏回归系数的数值标签（向上偏移0.25个单位）
for i, coef in enumerate(results["Lasso (Sparse/L1)"]["coef"]):
    ax.text(i, coef + 1, f'{coef:.2f}',
            ha='center', va='bottom',
            color='red', fontsize=8,
            alpha=0.9)



# 设置中文标签（已修复乱码）
ax.set_xlabel('特征索引')
ax.set_ylabel('系数值')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.header("模型性能对比")
st.write("""
| 模型                | MSE (均方误差) |
|---------------------|---------------|
| 岭回归 (L2)         | {:.4f}        |
| 稀疏回归 (L1)       | {:.4f}        |
""".format(
    results["Ridge (L2)"]["mse"],
    results["Lasso (Sparse/L1)"]["mse"]
))

st.header("为什么会有这个差异？")
st.markdown("""
1. **岭回归（L2）**：给所有系数加"小惩罚" → 系数变小但**不为0**
   - 适合：特征间有相关性的情况
   - 例子：所有特征都对结果有轻微影响

2. **稀疏回归（L1）**：给不重要的系数加"大惩罚" → 系数**直接变0**
   - 适合：特征很多但只有少数重要
   - 例子：只保留2个关键特征，其他特征直接忽略
""")