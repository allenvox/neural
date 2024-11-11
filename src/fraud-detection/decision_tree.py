import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample

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

# Выбор лучшего разбиения с ограниченным числом порогов и признаков
def best_split(X, y, criterion='gini', num_thresholds=10, num_features=None):
    best_feature, best_threshold, best_gain = None, None, -1
    n_samples, n_features = X.shape
    parent_impurity = gini(y)
    
    # Ограничение на количество признаков
    features = np.random.choice(n_features, num_features or n_features, replace=False)

    for feature_index in features:
        thresholds = np.unique(X[:, feature_index])
        
        # Случайные пороги для разбиения
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

    # Отладочный вывод для лучшего разбиения
    print(f"Best split: Feature {best_feature}, Threshold {best_threshold}, Gain {best_gain:.4f}")
    return best_feature, best_threshold

# Построение дерева с ограничением на минимальное число образцов в узле
def build_tree(X, y, depth=0, max_depth=5, min_samples_split=10, criterion='gini'):
    print(f"Building tree at depth {depth} with {len(y)} samples")

    if len(np.unique(y)) == 1:
        return DecisionNode(value=np.unique(y)[0])

    if depth >= max_depth or len(y) < min_samples_split:
        return DecisionNode(value=np.bincount(y).argmax())

    feature_index, threshold = best_split(X, y, criterion, num_thresholds=10, num_features=5)
    if feature_index is None:
        return DecisionNode(value=np.bincount(y).argmax())

    X_left, X_right, y_left, y_right = split_dataset(X, y, feature_index, threshold)
    
    left_subtree = build_tree(X_left, y_left, depth + 1, max_depth, min_samples_split, criterion)
    right_subtree = build_tree(X_right, y_right, depth + 1, max_depth, min_samples_split, criterion)
    
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

# Построение и оценка дерева решений
tree = build_tree(X_train, y_train, max_depth=5, min_samples_split=20)

y_pred = predict(X_test, tree)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
