"""
midterm_project/
├── main.py               # Streamlit主程序（运行这个文件！）
├── model_utils.py        # 核心计算逻辑（数据生成/模型训练）
└── requirements.txt      # 依赖库
"""

import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler


def generate_data(n_samples=200, n_features=10, n_informative=2, noise=10):
    """
    生成模拟数据（制造稀疏性：只有2个特征真正重要）

    参数:
    n_samples (int): 样本数量（默认200）
    n_features (int): 总特征数（默认10）
    n_informative (int): 真实重要的特征数（默认2，制造稀疏性）
    noise (float): 噪声强度（默认10，值越大噪声越强）

    返回:
    X (numpy array): 特征矩阵 (n_samples, n_features)
    y (numpy array): 目标值向量 (n_samples,)
    true_coef (numpy array): 真实系数向量 (n_features,)

    说明:
    - 使用make_regression生成数据，其中只有n_informative个特征有非零系数
    - 通过设置n_informative=2，制造了稀疏性（其他特征系数为0）
    - random_state=42确保结果可复现
    """
    # 生成带真实系数的模拟数据（制造稀疏性）
    X, y, true_coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,  # 关键：只让2个特征有真实影响
        noise=noise,  # 噪声强度
        coef=True,  # 返回真实系数
        random_state=42  # 固定随机种子确保可复现
    )
    return X, y, true_coef


def train_models(X, y, alpha=1.0):
    """
    训练岭回归和稀疏回归模型

    参数:
    X (numpy array): 特征矩阵 (n_samples, n_features)
    y (numpy array): 目标值向量 (n_samples,)
    alpha (float): 正则化强度（值越大惩罚越强，默认1.0）

    返回:
    models (dict): 模型结果字典，包含：
        "Ridge (L2)": {
            "coef": 岭回归系数,
            "mse": 训练集均方误差
        },
        "Lasso (Sparse/L1)": {
            "coef": Lasso回归系数,
            "mse": 训练集均方误差
        }

    说明:
    1. 岭回归 (Ridge): 使用L2正则化（所有系数被平滑但不为零）
    2. Lasso回归 (Lasso): 使用L1正则化（能产生稀疏解，部分系数为零）
    3. max_iter=10000: 确保Lasso收敛（避免因迭代不足导致不收敛）
    """
    # 1. 岭回归 (L2正则化) - 保留所有特征但系数被平滑
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X, y)  # 训练模型

    # 2. 稀疏回归 (L1正则化，Lasso实现) - 产生稀疏解
    lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)  # 增加迭代次数确保收敛
    lasso.fit(X, y)  # 训练模型

    # 3. 计算模型性能（训练集MSE）
    ridge_mse = np.mean((ridge.predict(X) - y) ** 2)  # 岭回归MSE
    lasso_mse = np.mean((lasso.predict(X) - y) ** 2)  # Lasso MSE

    # 4. 返回模型结果
    return {
        "Ridge (L2)": {
            "coef": ridge.coef_,  # 岭回归系数
            "mse": ridge_mse  # 岭回归训练集MSE
        },
        "Lasso (Sparse/L1)": {
            "coef": lasso.coef_,  # Lasso系数（稀疏解）
            "mse": lasso_mse  # Lasso训练集MSE
        }
    }