# model_utils.py
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler


def generate_data(n_samples=200, n_features=10, n_informative=2, noise=10):
    """
    生成模拟数据（制造稀疏性：只有2个特征真正重要）
    :param n_samples: 样本数量
    :param n_features: 总特征数
    :param n_informative: 真实重要的特征数（制造稀疏性）
    :param noise: 噪声强度
    :return: X（特征矩阵）, y（目标值）, true_coef（真实系数）
    """
    # 生成数据（make_regression会返回真实系数）
    X, y, true_coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,  # 只有2个特征真正重要！
        noise=noise,
        coef=True,
        random_state=42
    )
    return X, y, true_coef


def train_models(X, y, alpha=1.0):
    """
    训练岭回归和稀疏回归模型
    :param X: 特征矩阵
    :param y: 目标值
    :param alpha: 正则化强度（值越大惩罚越强）
    :return: 模型结果字典（包含系数、MSE等）
    """
    # 1. 岭回归 (L2正则化)
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X, y)

    # 2. 稀疏回归 (L1正则化，用Lasso实现)
    lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    lasso.fit(X, y)

    # 3. 返回结果（包含系数和性能指标）
    return {
        "Ridge (L2)": {
            "coef": ridge.coef_,
            "mse": np.mean((ridge.predict(X) - y) ** 2)  # 训练集MSE
        },
        "Lasso (Sparse/L1)": {
            "coef": lasso.coef_,
            "mse": np.mean((lasso.predict(X) - y) ** 2)  # 训练集MSE
        }
    }