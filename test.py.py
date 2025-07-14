import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 自定义训练集和测试集划分函数
def manual_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)  # 如果设置了随机种子，确保结果可重现
    indices = np.arange(X.shape[0])  # 获取所有数据点的索引
    np.random.shuffle(indices)  # 打乱数据集
    test_size = int(len(indices) * test_size)  # 根据测试集比例计算测试集大小
    train_indices, test_indices = indices[test_size:], indices[:test_size]  # 分割数据为训练集和测试集
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]  # 返回训练集和测试集

# 自定义Logistic回归模型类
class LogisticRegressionCustom:
    def __init__(self, lr=0.01, max_iter=1000, reg_lambda=0.01):
        self.lr = lr  # 学习率
        self.max_iter = max_iter  # 最大迭代次数
        self.reg_lambda = reg_lambda  # 正则化参数
        self.weights = None  # 权重初始化为空
        self.bias = None  # 偏置初始化为空

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))  # Sigmoid 激活函数

    def fit(self, X, y):
        n_samples, n_features = X.shape  # 获取样本数和特征数
        self.weights = np.zeros(n_features)  # 初始化权重为零
        self.bias = 0  # 初始化偏置为零
        for _ in range(self.max_iter):  # 进行最大迭代次数
            model = np.dot(X, self.weights) + self.bias  # 计算模型预测值
            y_pred = self.sigmoid(model)  # 应用sigmoid函数得到预测结果
            # 计算梯度（包括L2正则化项）
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + self.reg_lambda * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw  # 更新权重
            self.bias -= self.lr * db  # 更新偏置

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias  # 计算模型预测值
        y_pred = self.sigmoid(model)  # 应用sigmoid函数得到预测结果
        return np.where(y_pred >= 0.5, 1, 0)  # 预测结果大于0.5为1，否则为0

# 特征选择函数：根据特征的方差进行筛选
def variance_feature_selection(X, threshold=0.01):
    variances = np.var(X, axis=0)  # 计算每个特征的方差
    return np.where(variances > threshold)[0]  # 返回方差大于阈值的特征索引

# 自定义交叉验证函数
def cross_val_score_custom(model, X, y, cv=5):
    n_samples = X.shape[0]  # 获取样本数
    fold_size = n_samples // cv  # 计算每个折叠的大小
    scores = []  # 用于存储每个折叠的得分
    for i in range(cv):  # 进行交叉验证
        start, end = i * fold_size, (i + 1) * fold_size  # 划分验证集
        X_val, y_val = X[start:end], y[start:end]  # 获取验证集
        X_train = np.concatenate([X[:start], X[end:]], axis=0)  # 合并成训练集
        y_train = np.concatenate([y[:start], y[end:]], axis=0)  # 合并成训练标签
        model.fit(X_train, y_train)  # 训练模型
        y_pred = model.predict(X_val)  # 对验证集进行预测
        accuracy = np.mean(y_pred == y_val)  # 计算准确率
        scores.append(accuracy)  # 保存准确率
    return scores  # 返回交叉验证的得分

# 自定义准确率计算函数
def accuracy_score_custom(y_true, y_pred):
    return np.mean(y_true == y_pred)  # 计算预测准确率

# 自定义分类报告函数：返回每个类别的精确度、召回率、F1值等
def classification_report_custom(y_true, y_pred):
    unique_classes = np.unique(y_true)  # 获取所有类别
    results = []  # 用于存储每个类别的评估结果
    for cls in unique_classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))  # 计算真正例
        fp = np.sum((y_pred == cls) & (y_true != cls))  # 计算假正例
        fn = np.sum((y_pred != cls) & (y_true == cls))  # 计算假负例
        support = np.sum(y_true == cls)  # 计算每个类别的样本数
        precision = tp / (tp + fp) if tp + fp > 0 else 0  # 精确度
        recall = tp / (tp + fn) if tp + fn > 0 else 0  # 召回率
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0  # F1值
        results.append([cls, precision, recall, f1_score, support])  # 存储每个类别的结果
    df = pd.DataFrame(results, columns=["Class", "Precision", "Recall", "F1-Score", "Support"])  # 转换为DataFrame
    return df  # 返回分类报告

# 数据读取和预处理
data_train = pd.read_excel(r'C:\Users\Lenovo\Desktop\数据集\离婚率预测/divorce.xlsx')  # 读取数据
print("数据预览：", data_train.head())  # 打印数据预览

# 检查缺失值
missing_values = data_train.isnull().sum()  # 计算每一列缺失值的数量
print("\n缺失值情况：\n", missing_values)  # 打印缺失值情况

# 数据分析和可视化
# 相关性热力图
correlation_matrix = data_train.corr()  # 计算特征之间的相关性矩阵
plt.figure(figsize=(12, 8))  # 设置图像大小
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')  # 绘制热力图
plt.title("特征与目标变量的相关性热力图")  # 设置图像标题
plt.show()  # 显示图像

# 绘制特征方差分布
all_X = data_train.drop('Class', axis=1).values  # 获取所有特征数据
variances = np.var(all_X, axis=0)  # 计算特征的方差
plt.figure(figsize=(10, 6))  # 设置图像大小
plt.hist(variances, bins=30, color='blue', alpha=0.7)  # 绘制方差的直方图
plt.xlabel("Feature Variance")  # 设置x轴标签
plt.ylabel("Frequency")  # 设置y轴标签
plt.title("Feature Variance Distribution")  # 设置图像标题
plt.show()  # 显示图像

# 特征选择
selected_indices = variance_feature_selection(all_X, threshold=0.02)  # 选择方差大于0.02的特征
X_top = all_X[:, selected_indices]  # 根据选择的特征索引获取新的特征矩阵
print("选择后的特征数量：", X_top.shape[1])  # 打印选择后的特征数量

# 数据分割
all_y = data_train['Class'].values  # 获取目标变量
X_train, X_test, y_train, y_test = manual_train_test_split(X_top, all_y, test_size=0.2, random_state=42)  # 划分训练集和测试集

# 初始化Logistic回归模型
model = LogisticRegressionCustom(lr=0.1, max_iter=1000, reg_lambda=0.01)

# 初始化LogisticRegressionCustom模型
model = LogisticRegressionCustom(lr=0.1, max_iter=1000, reg_lambda=0.01)

# 交叉验证得分绘制
def plot_cv_scores(cv_scores):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linestyle='-', color='b')
    plt.title("Cross Validation Scores per Fold", fontsize=16)
    plt.xlabel("Fold", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(range(1, len(cv_scores) + 1))
    plt.grid(True)
    plt.show()

# 使用自定义交叉验证得分函数
cv_scores = cross_val_score_custom(model, X_train, y_train, cv=5)

# 打印交叉验证得分
print("交叉验证得分：", cv_scores)
print("平均交叉验证得分：", np.mean(cv_scores))

# 绘制交叉验证得分图
plot_cv_scores(cv_scores)


# 特征选择
selected_indices = variance_feature_selection(all_X, threshold=0.02)
X_top = all_X[:, selected_indices]
print("选择后的特征数量：", X_top.shape[1])

# 数据分割
all_y = data_train['Class'].values
X_train, X_test, y_train, y_test = manual_train_test_split(X_top, all_y, test_size=0.2, random_state=42)

# 模型训练与评估
model = LogisticRegressionCustom(lr=0.1, max_iter=1000, reg_lambda=0.01)
cv_scores = cross_val_score_custom(model, X_train, y_train, cv=5)
print("交叉验证得分：", cv_scores)
print("平均交叉验证得分：", np.mean(cv_scores))

#特征重要性可视化
def plot_feature_importance(weights, feature_names):
    importance = np.abs(weights)  # 取权重的绝对值作为特征的重要性
    sorted_idx = np.argsort(importance)[::-1]  # 按重要性排序
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance[sorted_idx], align='center')
    plt.yticks(range(len(importance)), np.array(feature_names)[sorted_idx])
    plt.title("Feature Importance", fontsize=16)
    plt.xlabel("Absolute Weight", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.show()

# 假设特征名称为 feature_names
plot_feature_importance(model.weights, data_train.columns[:-1])

# 测试集预测
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 性能评估
accuracy = accuracy_score_custom(y_test, y_pred)
print("测试集准确率：", accuracy)
print("分类报告：\n", classification_report_custom(y_test, y_pred))