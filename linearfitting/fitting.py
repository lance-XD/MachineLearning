# -*- coding: utf-8 -*-
"""
递推参数估计项目 - 完整实现（仅供教学演示）
============================================
本程序用于从有限项整数序列 a_n 中反推线性递推关系：
    a_{n+3} = p * a_{n+1} - q * a_n
通过最小二乘法（正规方程）求解最优参数 p, q。
同时，对每个数据集输出拟合误差（MSE 和 MAE），帮助判断解的质量。

环境依赖：仅需 pandas 和 os（均为标准库）
数据文件：需放在项目根目录下的 data/ 文件夹内
          包括 a_seq_train.csv, a_seq_testA.csv, a_seq_testB.csv
输出文件：submission.csv（三行两列：训练集、验证集A、测试集B）
"""

import pandas as pd
import os


# ================== 核心求解函数（纯 pandas 实现最小二乘） ==================
def fit_pq_from_series(a_series):
    """
    根据一维序列（pandas.Series）通过最小二乘法估计 p 和 q。

    数学原理：
        递推式：a_{n+3} = p * a_{n+1} - q * a_n
        对于每个 n，可写成线性方程：
            a_{n+3} = [a_{n+1}, -a_n] · [p, q]^T
        将所有 n 堆叠成矩阵形式：Y = X · β，其中 β = [p, q]^T
        超定方程组（方程数 > 未知数），用最小二乘法求解：
            β = (X^T X)^{-1} X^T Y
        这里手动实现 2×2 矩阵的求逆。

    参数：
        a_series : pandas.Series，数列值（长度至少4）
    返回：
        p, q : float，估计出的递推参数
    """
    # ---- 预处理：重置索引并转为浮点数，确保运算正确 ----
    # reset_index(drop=True) 使索引从0连续排列，便于切片对齐
    a = a_series.reset_index(drop=True).astype(float)
    n = len(a)

    # 至少需要4个点才能构造一个方程（n=0时用到 a0,a1,a3）
    if n < 4:
        return 0.0, 0.0

    # ---- 构造特征矩阵 X 和标签向量 Y ----
    # 递推式涉及 a_{n}, a_{n+1}, a_{n+3}
    # 可用的 n 范围：0 到 n-4（因为 n+3 最大为 n-1）
    # 切片含义：
    #   a.iloc[1:-2]  → 取第2个到倒数第3个元素，即 a_1 到 a_{n-3}（作为 a_{n+1}）
    #   -a.iloc[0:-3] → 取第1个到倒数第4个元素的相反数，即 -a_0 到 -a_{n-4}（作为 -a_n）
    #   a.iloc[3:]    → 取第4个到最后一个，即 a_3 到 a_{n-1}（作为 a_{n+3}）
    # 三者的长度均为 n-3，一一对应
    x1 = a.iloc[1:-2].reset_index(drop=True)   # 对应 a_{n+1}
    x2 = -a.iloc[0:-3].reset_index(drop=True)  # 对应 -a_n
    y = a.iloc[3:].reset_index(drop=True)      # 对应 a_{n+3}

    # ---- 计算正规方程 (X^T X) · β = X^T Y 的各个元素 ----
    # 设 X 为 (n-3)×2 矩阵，两列分别为 x1 和 x2
    # X^T X = [[Σ(x1^2), Σ(x1*x2)], [Σ(x1*x2), Σ(x2^2)]]
    # X^T Y = [Σ(x1*y), Σ(x2*y)]^T
    S11 = (x1 * x1).sum()   # Σ (a_{n+1})^2
    S12 = (x1 * x2).sum()   # Σ (a_{n+1} * (-a_n))
    S22 = (x2 * x2).sum()   # Σ (-a_n)^2 = Σ (a_n)^2
    E = (x1 * y).sum()      # Σ (a_{n+1} * a_{n+3})
    F = (x2 * y).sum()      # Σ (-a_n * a_{n+3})

    # ---- 手动求解 2×2 线性方程组 ----
    # 矩阵 A = [[S11, S12], [S12, S22]], 向量 b = [E, F]
    # 解 β = A^{-1} b，其中 A^{-1} = (1/det) * [[S22, -S12], [-S12, S11]]
    det = S11 * S22 - S12 * S12   # 行列式
    # 若行列式接近 0，说明数据退化（例如所有 a_n 全为零），此时无法求解
    if abs(det) < 1e-15:
        return 0.0, 0.0

    p = (S22 * E - S12 * F) / det   # β 的第一个分量
    q = (-S12 * E + S11 * F) / det  # β 的第二个分量
    return p, q


def pqregression(data):
    """
    与原始 baseline 接口完全兼容：从 DataFrame 中提取 'a_seq' 列并求解。

    参数：
        data : pandas.DataFrame，必须包含 'a_seq' 列
    返回：
        p, q : float
    """
    return fit_pq_from_series(data['a_seq'])


# ================== 评估指标函数 ==================
def evaluate_fit(p, q, data):
    """
    使用给定的 p, q 在 data 上计算预测误差，用于衡量拟合质量。

    递推预测：a_hat_{n+3} = p * a_{n+1} - q * a_n
    比较预测值与真实 a_{n+3}，计算：
        - 均方误差 (MSE) ：对误差平方取平均，对大误差敏感
        - 平均绝对误差 (MAE)：对误差绝对值取平均，更直观（单位与 a 相同）

    参数：
        p, q : float，待评估的参数
        data : pandas.DataFrame，包含 'a_seq' 列
    返回：
        mse, mae : float，若数据不足则返回 (nan, nan)
    """
    a = data['a_seq'].reset_index(drop=True).astype(float)
    # 至少需要4个点才能构造一个预测
    if len(a) < 4:
        return float('nan'), float('nan')

    # 构造与拟合时完全一致的特征和真实值
    x1 = a.iloc[1:-2].reset_index(drop=True)   # a_{n+1}
    x2 = -a.iloc[0:-3].reset_index(drop=True)  # -a_n
    y_true = a.iloc[3:].reset_index(drop=True) # a_{n+3}

    # 利用当前的 p, q 进行预测
    y_pred = p * x1 + q * x2
    errors = y_true - y_pred
    mse = (errors ** 2).mean()   # 均方误差
    mae = errors.abs().mean()    # 平均绝对误差
    return mse, mae


# ================== 主程序 ==================
if __name__ == '__main__':
    """
    整个流程分为四步：
      1. 读取训练集（优先本地 ./data/，否则使用线上路径）
      2. 读取测试集 A 和 B（使用环境变量或本地 ./data/）
      3. 对三个数据集分别求解参数，并输出评估指标
      4. 生成 submission.csv 提交文件
    """

    # ---------- 1. 读取训练集（兼容本地和线上） ----------
    # 本地路径：项目根目录下的 data 文件夹
    local_train = "./data/a_seq_train.csv"
    if os.path.exists(local_train):
        train_path = local_train
        print("本地模式：使用 ./data/a_seq_train.csv 作为训练集")
    else:
        # 线上平台（如 Bohr）使用的绝对路径
        train_path = "/bohr/train-ma3m/v1/a_seq_train.csv"
        print("线上模式：使用默认训练集路径")

    try:
        data_train = pd.read_csv(train_path)
        print(f"训练集加载成功，共 {len(data_train)} 项")
    except FileNotFoundError:
        print(f"错误：训练集文件 {train_path} 未找到，请检查路径。")
        exit(1)   # 终止程序

    # ---------- 2. 读取测试集 A 和 B ----------
    # 优先使用环境变量 DATA_PATH（线上评分时会设置），否则指向本地 ./data/
    if os.environ.get('DATA_PATH'):
        DATA_PATH = os.environ.get("DATA_PATH") + "/"
        print("线上模式：使用环境变量 DATA_PATH")
    else:
        DATA_PATH = "./data/"
        print("本地模式：使用 ./data/ 作为测试集路径")

    testA_path = DATA_PATH + "a_seq_testA.csv"
    testB_path = DATA_PATH + "a_seq_testB.csv"

    # 初始化为空 DataFrame，以便后续判断
    data_testA = pd.DataFrame()
    data_testB = pd.DataFrame()

    # 读取验证集 A（若文件不存在则给出警告，继续执行）
    try:
        data_testA = pd.read_csv(testA_path)
        print(f"验证集 A 加载成功，共 {len(data_testA)} 项")
    except FileNotFoundError:
        print(f"警告：验证集 A 文件 {testA_path} 未找到，将跳过该部分。")

    # 读取测试集 B
    try:
        data_testB = pd.read_csv(testB_path)
        print(f"测试集 B 加载成功，共 {len(data_testB)} 项")
    except FileNotFoundError:
        print(f"警告：测试集 B 文件 {testB_path} 未找到，将跳过该部分。")

    # ---------- 3. 分别求解三个数据集，并展示评估指标 ----------
    # 将三个数据集打包成列表，统一处理
    datasets = [
        ("训练集", data_train),
        ("验证集A", data_testA),
        ("测试集B", data_testB)
    ]

    results = []  # 用于存储每组的 (p, q)，顺序与 datasets 一致
    print("\n========== 求解结果与评估指标 ==========")

    for name, df in datasets:
        # 若数据为空，则记录 NaN 并跳过
        if df.empty:
            print(f"{name}：数据为空，跳过")
            results.append((float('nan'), float('nan')))
            continue

        # 求解参数
        p, q = pqregression(df)
        # 计算在当前数据集上的预测误差
        mse, mae = evaluate_fit(p, q, df)
        results.append((p, q))

        # 打印结果（保留6位小数）
        print(f"\n{name}：")
        print(f"  拟合参数：p = {p:.6f}, q = {q:.6f}")
        print(f"  均方误差 (MSE) = {mse:.6f}")
        print(f"  平均绝对误差 (MAE) = {mae:.6f}")

        # 根据 MAE 给出定性评价，便于快速判断
        if mae < 0.01:
            print("  ★ 拟合极佳，几乎完全还原递推关系")
        elif mae < 0.5:
            print("  ★ 拟合良好，预测误差在半个单位以内")
        else:
            print("  ※ 拟合偏差较大，建议检查数据或算法")

    # ---------- 4. 生成提交文件 submission.csv ----------
    # 提取所有 (p, q)，按行填充
    p_values = [r[0] for r in results]
    q_values = [r[1] for r in results]
    df_params = pd.DataFrame({'p': p_values, 'q': q_values})

    print("\n========== 提交文件内容 ==========")
    print(df_params)

    csv_file_path = 'submission.csv'
    df_params.to_csv(csv_file_path, index=False)
    print(f"\n结果已保存至 {csv_file_path}")
    print("程序运行完毕。")

