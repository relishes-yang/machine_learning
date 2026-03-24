import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from model_utils import generate_data, train_models
import matplotlib.font_manager as fm

# === 1. 确保使用 Agg 后端（必须放在最前面） ===
matplotlib.use('Agg')

# === 2. 关键修复：使用 Streamlit Cloud 上已预装的中文字体 ===
try:
    # Streamlit Cloud 已预装 Source Han Sans
    font_path = 'SourceHanSansSC-Regular.otf'
    font_prop = fm.FontProperties(fname=font_path)

    # 设置全局字体
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    print(f"成功加载字体: {font_prop.get_name()}")
except Exception as e:
    print(f"字体加载失败: {str(e)}")
    # 备用方案：使用系统默认中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

# === 3. 以下为项目核心逻辑 ===

# === 1. 创建Streamlit页面标题（最顶部的标题） ===
st.title("岭回归 vs 稀疏回归：系数差异对比")

# === 2. 创建侧边栏参数调整区（用户可以拖动滑块改变参数） ===
st.sidebar.header("参数调整")  # 侧边栏标题

# 用户可以调整的参数（滑块控件）
n_samples = st.sidebar.slider("样本数量", 50, 500, 200)  # 默认200个样本
n_features = st.sidebar.slider("总特征数", 5, 20, 10)  # 默认10个特征
n_informative = st.sidebar.slider("重要特征数", 1, 5, 2)  # 默认只有2个特征重要
noise = st.sidebar.slider("噪声强度", 5, 50, 10)  # 噪声强度（越大越乱）
alpha = st.sidebar.slider("正则化强度", 0.1, 10.0, 1.0, 0.1)  # L1/L2正则化强度

# === 3. 生成数据并训练模型（核心计算部分） ===
# 1. 生成模拟数据（制造稀疏性：只有n_informative个特征重要）
X, y, true_coef = generate_data(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    noise=noise
)

# 2. 训练两个模型（岭回归和稀疏回归）
results = train_models(X, y, alpha=alpha)

# === 4. 可视化展示（关键：用图表展示系数差异） ===
st.header("关键发现：系数差异")
st.write("""
- **岭回归（Ridge）**：所有系数都变小了，但**不会变成0**
- **稀疏回归（Lasso）**：很多系数**直接变成0**，只保留重要特征
""")

# 创建图表（10宽 x 4高英寸）
fig, ax = plt.subplots(figsize=(10, 4))

# 绘制真实系数（黑色，代表数据真实情况）
ax.stem(true_coef, linefmt='k-', markerfmt='ko', basefmt=' ', label='真实系数')

# 绘制岭回归系数（蓝色，L2正则化）
ax.stem(
    results["Ridge (L2)"]["coef"],
    linefmt='b-', markerfmt='bo', basefmt=' ',
    label='岭回归系数',
)

# 绘制稀疏回归系数（红色，L1正则化）
ax.stem(
    results["Lasso (Sparse/L1)"]["coef"],
    linefmt='r-', markerfmt='ro', basefmt=' ',
    label='稀疏回归系数',
)

# === 修复图表标签重叠问题（关键！）===
# 1. 在真实系数上方添加数值标签（向上偏移5个单位）
for i, coef in enumerate(true_coef):
    ax.text(i, coef + 8, f'{coef:.2f}',  # 显示系数值（保留2位小数）
            ha='center', va='bottom',  # 水平居中，垂直底部对齐
            color='black', fontsize=8,  # 字体颜色和大小
            alpha=0.9)  # 透明度

# 2. 在岭回归系数上方添加数值标签（向上偏移3个单位）
for i, coef in enumerate(results["Ridge (L2)"]["coef"]):
    ax.text(i, coef + 4, f'{coef:.2f}',
            ha='center', va='bottom',
            color='blue', fontsize=8,
            alpha=0.9)

# 3. 在稀疏回归系数上方添加数值标签（向上偏移1个单位）
for i, coef in enumerate(results["Lasso (Sparse/L1)"]["coef"]):
    ax.text(i, coef + 1, f'{coef:.2f}',
            ha='center', va='bottom',
            color='red', fontsize=8,
            alpha=0.9)

# === 重点修复：显式指定字体属性 ===
# 使用之前设置的字体
font_prop = fm.FontProperties(fname=font_path) if 'font_path' in locals() else plt.rcParams['font.sans-serif'][0]

# 设置图表标签（中文已修复）
ax.set_xlabel('特征索引', fontproperties=font_prop)
ax.set_ylabel('系数值', fontproperties=font_prop)
ax.legend(prop={'family': font_prop.get_name()})  # 确保图例也使用正确字体
ax.grid(True, alpha=0.3)  # 显示网格线
st.pyplot(fig)  # 将图表显示在Streamlit页面

# === 5. 模型性能对比表格 ===
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

# === 6. 关键结论解释（用大白话解释差异） ===
st.header("为什么会有这个差异？")
st.markdown("""
1. **岭回归（L2）**：给所有系数加"小惩罚" → 系数变小但**不为0**
   - 适合：特征间有相关性的情况
   - 例子：所有特征都对结果有轻微影响（如身高和体重）

2. **稀疏回归（L1）**：给不重要的系数加"大惩罚" → 系数**直接变0**
   - 适合：特征很多但只有少数重要
   - 例子：只保留2个关键特征（如房价中的"面积"和"位置"），其他特征忽略
""")