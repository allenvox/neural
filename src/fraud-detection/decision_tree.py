import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample
import time
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('./data/creditcard.csv')

# Балансировка классов
data_class_0 = data[data['Class'] == 0]
data_class_1 = data[data['Class'] == 1]
data_class_1_oversampled = resample(data_class_1, 
                                    replace=True,     
                                    n_samples=len(data_class_0), 
                                    random_state=42) 
data_balanced = pd.concat([data_class_0, data_class_1_oversampled])

# Масштабирование данных
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_balanced['Amount'] = scaler.fit_transform(data_balanced['Amount'].values.reshape(-1, 1))

# Разделение на тренировочную и тестовую выборки
X = data_balanced.drop('Class', axis=1).values
y = data_balanced['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Узел дерева решений
class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Критерий разбиения
def gini(y):
    classes = np.unique(y)
    return 1.0 - sum((np.sum(y == c) / len(y)) ** 2 for c in classes)

# Разбиение
def split_dataset(X, y, feature_index, threshold):
    left_indices = np.where(X[:, feature_index] <= threshold)
    right_indices = np.where(X[:, feature_index] > threshold)
    return X[left_indices], X[right_indices], y[left_indices], y[right_indices]

# Измерение времени выполнения для лучшего разбиения
def best_split(X, y, criterion='gini', num_thresholds=10, num_features=None):
    best_feature, best_threshold, best_gain = None, None, -1
    n_samples, n_features = X.shape
    parent_impurity = gini(y)
    
    features = np.random.choice(n_features, num_features or n_features, replace=False)

    for feature_index in features:
        thresholds = np.unique(X[:, feature_index])
        if len(thresholds) > num_thresholds:
            thresholds = np.random.choice(thresholds, num_thresholds, replace=False)
        
        for threshold in thresholds:
            X_left, X_right, y_left, y_right = split_dataset(X, y, feature_index, threshold)
            if len(y_left) > 0 and len(y_right) > 0:
                p_left, p_right = len(y_left) / len(y), len(y_right) / len(y)
                impurity = p_left * gini(y_left) + p_right * gini(y_right)
                gain = parent_impurity - impurity
                if gain > best_gain:
                    best_feature, best_threshold, best_gain = feature_index, threshold, gain

    return best_feature, best_threshold

# Построение дерева

def build_tree_with_timing(X, y, depth=0, max_depth=5, min_samples_split=10, criterion='gini', timing_data=None):
    start_time = time.time()

    if len(np.unique(y)) == 1:
        elapsed_time = time.time() - start_time
        if timing_data is not None:
            timing_data.append((depth, elapsed_time))
        return DecisionNode(value=np.unique(y)[0])

    if depth >= max_depth or len(y) < min_samples_split:
        elapsed_time = time.time() - start_time
        if timing_data is not None:
            timing_data.append((depth, elapsed_time))
        return DecisionNode(value=np.bincount(y).argmax())

    feature_index, threshold = best_split(X, y, criterion, num_thresholds=10, num_features=5)
    if feature_index is None:
        elapsed_time = time.time() - start_time
        if timing_data is not None:
            timing_data.append((depth, elapsed_time))
        return DecisionNode(value=np.bincount(y).argmax())

    X_left, X_right, y_left, y_right = split_dataset(X, y, feature_index, threshold)
    
    left_subtree = build_tree_with_timing(X_left, y_left, depth + 1, max_depth, min_samples_split, criterion, timing_data)
    right_subtree = build_tree_with_timing(X_right, y_right, depth + 1, max_depth, min_samples_split, criterion, timing_data)
    
    elapsed_time = time.time() - start_time
    if timing_data is not None:
        timing_data.append((depth, elapsed_time))
    
    return DecisionNode(feature_index, threshold, left_subtree, right_subtree)

# Предсказание
def predict_tree(node, sample):
    if node.value is not None:
        return node.value
    if sample[node.feature_index] <= node.threshold:
        return predict_tree(node.left, sample)
    else:
        return predict_tree(node.right, sample)

def predict(X, tree):
    return np.array([predict_tree(tree, sample) for sample in X])

# Сбор метрик и времени для разных глубин дерева
max_depth_values = range(1, 11)
metrics_data = []
cumulative_timing_data = []

for max_depth in max_depth_values:
    timing_data = []
    tree = build_tree_with_timing(X_train, y_train, max_depth=max_depth, min_samples_split=20, timing_data=timing_data)

    y_pred = predict(X_test, tree)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    metrics_data.append((max_depth, accuracy, precision, recall, f1, roc_auc))
    cumulative_timing_data.append((max_depth, sum([t[1] for t in timing_data])))

# Построение графиков
metrics_df = pd.DataFrame(metrics_data, columns=['max_depth', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
cumulative_timing_df = pd.DataFrame(cumulative_timing_data, columns=['max_depth', 'cumulative_time'])

plt.figure(figsize=(12, 6))
plt.plot(metrics_df['max_depth'], metrics_df['accuracy'], marker='o', label='Accuracy')
plt.plot(metrics_df['max_depth'], metrics_df['roc_auc'], marker='o', label='ROC-AUC')
plt.xlabel('Max Depth')
plt.ylabel('Metrics')
plt.title('Metrics vs Tree Depth')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(cumulative_timing_df['max_depth'], cumulative_timing_df['cumulative_time'], marker='o', label='Cumulative Time')
plt.xlabel('Max Depth')
plt.ylabel('Cumulative Time (seconds)')
plt.title('Cumulative Time vs Tree Depth')
plt.legend()
plt.grid()
plt.show()
