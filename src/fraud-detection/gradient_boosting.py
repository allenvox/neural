import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

class SimpleDecisionTree:
    def __init__(self, max_depth=3, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        print(f"Fitting tree at depth {depth}, number of samples: {len(y)}")
        
        # Условия остановки
        if depth >= self.max_depth or len(set(y)) == 1 or len(y) < self.min_samples_split:
            self.tree = np.mean(y)  # Простой предсказатель - среднее значение
            print(f"Stopping condition reached at depth {depth}. Node prediction: {self.tree}")
            return

        # Ищем лучшее разбиение
        best_mse, best_idx, best_thr = float('inf'), None, None
        for i in range(X.shape[1]):
            thresholds = np.unique(X[:, i])
            print(f"Feature {i}: Testing {len(thresholds)} thresholds...")
            for thr in thresholds:
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

        # Если не найдено разбиение
        if best_idx is None:
            self.tree = np.mean(y)
            print(f"No valid split found at depth {depth}. Node prediction: {self.tree}")
            return

        print(f"Best split: Feature {best_idx}, Threshold {best_thr}, MSE {best_mse:.4f}")

        # Сохраняем разбиение
        self.tree = {
            'index': best_idx,
            'threshold': best_thr,
            'left': SimpleDecisionTree(self.max_depth, self.min_samples_split),
            'right': SimpleDecisionTree(self.max_depth, self.min_samples_split),
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



import time

class GradientBoosting:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, min_samples_split=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.init_prediction = None

    def fit(self, X, y):
        self.trees = []
        self.init_prediction = np.mean(y)  # Начальное предсказание - среднее по целевой переменной
        residuals = y - self.init_prediction

        print(f"Initial prediction (mean of target): {self.init_prediction}")

        for i in range(self.n_estimators):
            print(f"\nTraining tree {i + 1}/{self.n_estimators}...")
            start_time = time.time()

            tree = SimpleDecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            predictions = tree.predict(X)

            print(f"Tree {i + 1} predictions (mean): {np.mean(predictions):.4f}, std: {np.std(predictions):.4f}")
            residuals -= self.learning_rate * predictions  # Обновляем ошибки

            elapsed_time = time.time() - start_time
            print(f"Tree {i + 1} trained in {elapsed_time:.2f} seconds.")
            print(f"Residuals after tree {i + 1}: Mean={np.mean(residuals):.4f}, Std={np.std(residuals):.4f}")

            self.trees.append(tree)

    def predict(self, X):
        predictions = np.full(X.shape[0], self.init_prediction)
        for i, tree in enumerate(self.trees):
            pred = tree.predict(X)
            print(f"Tree {i + 1} contribution to prediction: Mean={np.mean(pred):.4f}, Std={np.std(pred):.4f}")
            predictions += self.learning_rate * pred
        return np.round(predictions).astype(int)


# Загрузка и подготовка данных
print("Loading data...")
data = pd.read_csv('./data/creditcard.csv')

# Балансировка классов
print("Balancing classes...")
data_class_0 = data[data['Class'] == 0]
data_class_1 = data[data['Class'] == 1]
data_class_1_oversampled = resample(data_class_1,
                                    replace=True,
                                    n_samples=len(data_class_0),
                                    random_state=42)
data_balanced = pd.concat([data_class_0, data_class_1_oversampled])

# Масштабирование данных
print("Scaling data...")
features = data_balanced.drop('Class', axis=1).columns
scaler = StandardScaler()
data_balanced[features] = scaler.fit_transform(data_balanced[features])

# Разделение на тренировочные и тестовые выборки
print("Splitting data...")
X = data_balanced.drop('Class', axis=1).values
y = data_balanced['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели
print("Initializing Gradient Boosting model...")
model = GradientBoosting(n_estimators=10, learning_rate=0.1, max_depth=3, min_samples_split=10)
model.fit(X_train, y_train)

# Оценка модели
print("Evaluating the model...")
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
