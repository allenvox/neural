import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        # Условия остановки
        if depth >= self.max_depth or len(set(y)) == 1 or len(y) < self.min_samples_split:
            self.tree = Counter(y).most_common(1)[0][0]  # Присваиваем наиболее частый класс
            return

        # Находим наилучший разбиение по критерию Джини
        best_gini, best_idx, best_thr = 1, None, None
        for i in range(X.shape[1]):
            thresholds = np.linspace(X[:, i].min(), X[:, i].max(), num=10)
            for thr in thresholds:
                left_y = y[X[:, i] <= thr]
                right_y = y[X[:, i] > thr]
                gini = self._gini_impurity(left_y, right_y)
                if gini < best_gini:
                    best_gini, best_idx, best_thr = gini, i, thr

        if best_idx is None:
            self.tree = Counter(y).most_common(1)[0][0]
            return

        self.tree = {
            'index': best_idx,
            'threshold': best_thr,
            'left': DecisionTree(self.max_depth, self.min_samples_split),
            'right': DecisionTree(self.max_depth, self.min_samples_split)
        }
        left_indices = X[:, best_idx] <= best_thr
        right_indices = X[:, best_idx] > best_thr
        self.tree['left'].fit(X[left_indices], y[left_indices], depth + 1)
        self.tree['right'].fit(X[right_indices], y[right_indices], depth + 1)

    def _gini_impurity(self, left_y, right_y):
        total_size = len(left_y) + len(right_y)
        left_gini = 1 - sum((np.sum(left_y == c) / len(left_y)) ** 2 for c in np.unique(left_y))
        right_gini = 1 - sum((np.sum(right_y == c) / len(right_y)) ** 2 for c in np.unique(right_y))
        return (len(left_y) / total_size) * left_gini + (len(right_y) / total_size) * right_gini

    def predict_single(self, x):
        if not isinstance(self.tree, dict):
            return self.tree
        if x[self.tree['index']] <= self.tree['threshold']:
            return self.tree['left'].predict_single(x)
        return self.tree['right'].predict_single(x)

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.timing_data = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        for i in range(self.n_estimators):
            start_time = time.time()
            indices = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)
            elapsed_time = time.time() - start_time
            self.timing_data.append((i + 1, elapsed_time))

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.round(np.mean(tree_preds, axis=0))

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
features = data_balanced.drop('Class', axis=1).columns
scaler = StandardScaler()
data_balanced[features] = scaler.fit_transform(data_balanced[features])

# Разделение на тренировочную и тестовую выборки
X = data_balanced.drop('Class', axis=1).values
y = data_balanced['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Проведение эксперимента с различным числом деревьев и глубиной
n_estimators_list = [5, 10, 20]
max_depth_list = [3, 5, 7]
metrics_data = []

for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        model = RandomForest(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=10)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        metrics_data.append({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        })

# Построение графиков времени и метрик
metrics_df = pd.DataFrame(metrics_data)

# График времени обучения
plt.figure(figsize=(10, 6))
for n_estimators in n_estimators_list:
    subset = metrics_df[metrics_df['n_estimators'] == n_estimators]
    plt.plot(subset['max_depth'], subset['roc_auc'], label=f'n_estimators={n_estimators}', marker='o')

plt.xlabel('Max Depth')
plt.ylabel('ROC-AUC')
plt.title('Random Forest Performance by Max Depth and n_estimators')
plt.legend()
plt.grid()
plt.show()

# Кумулятивное время построения деревьев
cumulative_time = pd.DataFrame(model.timing_data, columns=['Tree', 'Time']).cumsum()
plt.figure(figsize=(10, 6))
plt.plot(cumulative_time['Tree'], cumulative_time['Time'], marker='o')
plt.xlabel('Number of Trees')
plt.ylabel('Cumulative Time (seconds)')
plt.title('Cumulative Training Time of Random Forest')
plt.grid()
plt.show()
