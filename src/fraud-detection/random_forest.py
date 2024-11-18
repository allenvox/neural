import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import time

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        print(f"Depth: {depth}, Samples: {len(y)}")  # Отладочное сообщение

        # Условия остановки
        if depth >= self.max_depth or len(set(y)) == 1 or len(y) < self.min_samples_split:
            self.tree = Counter(y).most_common(1)[0][0]  # Присваиваем наиболее частый класс
            return

        # Находим наилучший разбиение по критерию Джини
        best_gini, best_idx, best_thr = 1, None, None
        for i in range(X.shape[1]):
            # Уменьшаем количество порогов для ускорения
            thresholds = np.linspace(X[:, i].min(), X[:, i].max(), num=10)
            for thr in thresholds:
                left_y = y[X[:, i] <= thr]
                right_y = y[X[:, i] > thr]
                gini = self._gini_impurity(left_y, right_y)
                if gini < best_gini:
                    best_gini, best_idx, best_thr = gini, i, thr

        # Если не удалось найти подходящее разбиение
        if best_idx is None:
            print("No split found, assigning majority class.")
            self.tree = Counter(y).most_common(1)[0][0]
            return

        print(f"Best split: Feature {best_idx}, Threshold {best_thr}, Gini {best_gini}")

        # Рекурсивно строим дерево
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

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]
        print("Training Random Forest...")
        for i in range(self.n_estimators):
            print(f"Training tree {i+1}/{self.n_estimators}...")
            start_time = time.time()
            indices = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)
            print(f"Tree {i+1} trained in {time.time() - start_time:.2f} seconds.")

    def predict(self, X):
        print("Predicting with Random Forest...")
        start_time = time.time()
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        print(f"Prediction completed in {time.time() - start_time:.2f} seconds.")
        return np.round(np.mean(tree_preds, axis=0))


# Загрузка данных
print("Loading data...")
data = pd.read_csv('./data/creditcard.csv')

# Проверяем, что данные загружены корректно
if data.isnull().values.any():
    print("Data contains missing values. Please check the data source.")
    exit()
print("Data loaded successfully.")

# Балансировка классов
print("Balancing classes...")
data_class_0 = data[data['Class'] == 0]
data_class_1 = data[data['Class'] == 1]
data_class_1_oversampled = resample(data_class_1,
                                    replace=True,
                                    n_samples=len(data_class_0),
                                    random_state=42)
data_balanced = pd.concat([data_class_0, data_class_1_oversampled])
print("Classes balanced.")

# Масштабируем все числовые признаки, кроме `Class`
print("Scaling data...")
features = data_balanced.drop('Class', axis=1).columns
scaler = StandardScaler()
data_balanced[features] = scaler.fit_transform(data_balanced[features])
print("Data scaled.")

# Разделение на тренировочную и тестовую выборки
print("Splitting data...")
X = data_balanced.drop('Class', axis=1).values
y = data_balanced['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Data split into training and testing sets.")

# Создаем и обучаем модель
print("Initializing Random Forest model...")
model = RandomForest(n_estimators=10, max_depth=5, min_samples_split=10)
model.fit(X_train, y_train)

# Предсказания и метрики
print("Calculating metrics...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("Metrics calculated:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)
