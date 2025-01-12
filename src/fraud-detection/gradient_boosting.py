import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time


class SimpleDecisionTree:
    def __init__(self, max_depth=3, min_samples_split=10, max_thresholds=50):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_thresholds = max_thresholds  # Максимальное количество порогов для разбиения
        self.tree = None

    def fit(self, X, y, depth=0):
        print(f"Tree depth {depth}: fitting {len(y)} samples.")

        # Условия остановки
        if depth >= self.max_depth or len(set(y)) == 1 or len(y) < self.min_samples_split:
            self.tree = np.mean(y)
            print(f"Stopping condition met at depth {depth}. Node prediction: {self.tree}")
            return

        # Поиск лучшего разбиения
        best_mse, best_idx, best_thr = float('inf'), None, None
        for i in range(X.shape[1]):
            unique_thresholds = np.unique(X[:, i])
            if len(unique_thresholds) > self.max_thresholds:
                # Ограничиваем количество порогов, выбирая равномерно распределённые
                unique_thresholds = np.linspace(min(unique_thresholds), max(unique_thresholds), self.max_thresholds)

            print(f"Feature {i}: testing {len(unique_thresholds)} thresholds.")

            for thr in unique_thresholds:
                left_mask = X[:, i] <= thr
                right_mask = X[:, i] > thr
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                mse = (
                    self._mse(y[left_mask]) * np.sum(left_mask)
                    + self._mse(y[right_mask]) * np.sum(right_mask)
                )
                if mse < best_mse:
                    best_mse, best_idx, best_thr = mse, i, thr

        if best_idx is None:  # Если подходящего разбиения не найдено
            self.tree = np.mean(y)
            print(f"No valid split found at depth {depth}. Node prediction: {self.tree}")
            return

        print(f"Best split: Feature {best_idx}, Threshold {best_thr}, MSE {best_mse:.4f}")

        # Сохраняем разбиение
        self.tree = {
            'index': best_idx,
            'threshold': best_thr,
            'left': SimpleDecisionTree(self.max_depth, self.min_samples_split, self.max_thresholds),
            'right': SimpleDecisionTree(self.max_depth, self.min_samples_split, self.max_thresholds),
        }

        left_mask = X[:, best_idx] <= best_thr
        right_mask = X[:, best_idx] > best_thr
        print(f"Splitting: {np.sum(left_mask)} samples go to the left, {np.sum(right_mask)} to the right.")

        self.tree['left'].fit(X[left_mask], y[left_mask], depth + 1)
        self.tree['right'].fit(X[right_mask], y[right_mask], depth + 1)

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def predict_single(self, x):
        if not isinstance(self.tree, dict):
            return self.tree
        if x[self.tree['index']] <= self.tree['threshold']:
            return self.tree['left'].predict_single(x)
        return self.tree['right'].predict_single(x)

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])


class GradientBoosting:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=10, min_samples_split=10, max_thresholds=50):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_thresholds = max_thresholds
        self.trees = []
        self.init_prediction = None
        self.training_times = []  # Время на обучение каждого дерева
        self.depth_metrics = []  # Метрики в зависимости от глубины
        self.depth_times = []  # Суммарное время в зависимости от глубины

    def fit(self, X, y):
        self.trees = []
        self.init_prediction = np.mean(y)
        residuals = y - self.init_prediction
        cumulative_time = 0  # Накопительное время обучения

        for i in range(self.n_estimators):
            start_time = time.time()
            tree = SimpleDecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                      max_thresholds=self.max_thresholds)
            tree.fit(X, residuals)
            predictions = tree.predict(X)
            residuals -= self.learning_rate * predictions
            elapsed_time = time.time() - start_time
            cumulative_time += elapsed_time
            
            # Логируем данные
            self.training_times.append(elapsed_time)
            self.trees.append(tree)

        # Оценка метрик для всех глубин
        for depth in range(1, self.max_depth + 1):
            temp_tree = SimpleDecisionTree(max_depth=depth, min_samples_split=self.min_samples_split,
                                           max_thresholds=self.max_thresholds)
            start_time = time.time()
            temp_tree.fit(X, residuals)
            elapsed_time = time.time() - start_time
            predictions = temp_tree.predict(X)
            roc_auc = roc_auc_score(y, np.round(self.init_prediction + self.learning_rate * predictions))
            f1 = f1_score(y, np.round(self.init_prediction + self.learning_rate * predictions))
            
            self.depth_metrics.append((depth, roc_auc, f1))
            self.depth_times.append((depth, elapsed_time))

    def predict(self, X):
        predictions = np.full(X.shape[0], self.init_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return np.round(predictions).astype(int)

    def plot_depth_vs_metrics(self):
        # График зависимости метрик от глубины
        depths, roc_aucs, f1_scores = zip(*self.depth_metrics)
        plt.figure(figsize=(12, 6))
        plt.plot(depths, roc_aucs, marker='o')
        plt.xlabel("Max Tree Depth")
        plt.ylabel("ROC-AUC")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_time_vs_depth(self):
        # График времени обучения в зависимости от глубины
        depths, times = zip(*self.depth_times)
        plt.figure(figsize=(12, 6))
        plt.plot(depths, times, marker='o')
        plt.xlabel("Max Tree Depth")
        plt.ylabel("Learning Time (seconds)")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_tree_training_time(self):
        # Время обучения для каждого дерева
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(self.training_times) + 1), self.training_times, marker='o')
        plt.xlabel("Tree Number")
        plt.ylabel("Learning Time (seconds)")
        plt.legend()
        plt.grid()
        plt.show()


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

# Разделение на тренировочные и тестовые выборки
X = data_balanced.drop('Class', axis=1).values
y = data_balanced['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели
model = GradientBoosting(n_estimators=10, learning_rate=0.1, max_depth=10, min_samples_split=10)
model.fit(X_train, y_train)

# Оценка модели
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

model.plot_depth_vs_metrics()  # Зависимость метрик от глубины
model.plot_time_vs_depth()  # Время обучения от глубины
model.plot_tree_training_time()  # Время обучения каждого дерева
