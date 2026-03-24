import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from model_utils import generate_data, train_models
import matplotlib.font_manager as fm

# === 1. 确保使用 Agg 后端（必须放在最前面） ===
matplotlib.use('Agg')

# === 2. 关键修复：使用开源中文字体（无需系统字体） ===
try:
    # 1. 尝试加载开源中文字体（Source Han Sans，支持中文）
    font_path = 'SourceHanSansSC-Regular.otf'
    font_prop = fm.FontProperties(fname=font_path)

    # 2. 设置全局字体
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 3. 验证字体是否加载成功
    print(f"成功加载字体: {font_prop.get_name()}")
except Exception as e:
    print(f"字体加载失败: {str(e)}")
    # 4. 备用方案：使用系统默认中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

# === 3. 以下为项目核心逻辑 ===

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

# 创建图表
fig, ax = plt.subplots(figsize=(10, 4))

# 绘制真实系数
ax.stem(true_coef, linefmt='k-', markerfmt='ko', basefmt=' ', label='真实系数')

# 绘制岭回归系数
ax.stem(
    results["Ridge (L2)"]["coef"],
    linefmt='b-', markerfmt='bo', basefmt=' ',
    label='岭回归系数',
)

# 绘制稀疏回归系数
ax.stem(
    results["Lasso (Sparse/L1)"]["coef"],
    linefmt='r-', markerfmt='ro', basefmt=' ',
    label='稀疏回归系数',
)

# 添加系数数值标签
for i, coef in enumerate(true_coef):
    ax.text(i, coef + 8, f'{coef:.2f}',
            ha='center', va='bottom',
            color='black', fontsize=8,
            alpha=0.9)

for i, coef in enumerate(results["Ridge (L2)"]["coef"]):
    ax.text(i, coef + 4, f'{coef:.2f}',
            ha='center', va='bottom',
            color='blue', fontsize=8,
            alpha=0.9)

for i, coef in enumerate(results["Lasso (Sparse/L1)"]["coef"]):
    ax.text(i, coef + 1, f'{coef:.2f}',
            ha='center', va='bottom',
            color='red', fontsize=8,
            alpha=0.9)

# === 重点修复：使用字体属性设置中文标签 ===
ax.set_xlabel('特征索引', fontproperties=plt.rcParams['font.sans-serif'][0])
ax.set_ylabel('系数值', fontproperties=plt.rcParams['font.sans-serif'][0])
ax.legend(prop={'family': plt.rcParams['font.sans-serif'][0]})
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
   - 例子：所有特征都对结果有轻微影响（如身高和体重）

2. **稀疏回归（L1）**：给不重要的系数加"大惩罚" → 系数**直接变0**
   - 适合：特征很多但只有少数重要
   - 例子：只保留2个关键特征（如房价中的"面积"和"位置"），其他特征忽略
""")