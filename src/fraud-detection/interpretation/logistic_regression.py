import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt
import time

# Загрузка данных
data = pd.read_csv('./data/creditcard.csv')

# Проверка на пропущенные значения
if data.isnull().values.any():
    data.fillna(data.mean(), inplace=True)

# Балансировка классов с помощью oversampling для класса 1
data_class_0 = data[data['Class'] == 0]
data_class_1 = data[data['Class'] == 1]
data_class_1_oversampled = resample(data_class_1, replace=True, n_samples=len(data_class_0), random_state=42)
data_balanced = pd.concat([data_class_0, data_class_1_oversampled])

# Масштабируем все признаки
scaler = StandardScaler()
X = scaler.fit_transform(data_balanced.drop('Class', axis=1))
y = data_balanced['Class'].values

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Инициализация весов и параметров модели
np.random.seed(0)
num_features = X_train.shape[1]
weights = np.random.uniform(-0.01, 0.01, num_features)
bias = 0.0
learning_rate = 0.1

# Сигмоидная функция
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Функция потерь
def compute_loss(y, y_hat, class_weights, epsilon=1e-9):
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    weight = np.where(y == 1, class_weights[1], class_weights[0])
    return -np.mean(weight * (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

# Прямой проход
def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z)

# Оценка модели
def evaluate_model(X, y, weights, bias):
    y_pred = predict(X, weights, bias) >= 0.5
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=1)
    recall = recall_score(y, y_pred, zero_division=1)
    f1 = f1_score(y, y_pred, zero_division=1)
    roc_auc = roc_auc_score(y, predict(X, weights, bias))

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

# Установка весов классов
class_weights = {0: 1, 1: len(data_class_0) / len(data_class_1_oversampled)}

# Обучение модели
def train_logistic_regression(X, y, weights, bias, learning_rate, num_epochs, class_weights, tol=0.01, patience=10):
    no_improve_count = 0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # Прогноз и вычисление ошибки
        y_hat = predict(X, weights, bias)
        error = y_hat - y

        # Градиенты
        dW = np.dot(X.T, error) / len(y)
        dB = np.sum(error) / len(y)

        # Обновление весов и смещения
        weights -= learning_rate * dW
        bias -= learning_rate * dB

        # Вычисление метрики каждые 10 эпох
        if epoch % 10 == 0:
            loss = compute_loss(y, y_hat, class_weights)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

            # Раняя остановка
            if loss < best_loss - tol:
                best_loss = loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch} due to minimal loss improvement.")
                break
    
    return weights, bias

# Обучение модели
weights, bias = train_logistic_regression(
    X_train, y_train, weights, bias, learning_rate, num_epochs=2000, class_weights=class_weights
)

# Оценка на тестовой выборке
evaluate_model(X_test, y_test, weights, bias)

# --- SHAP Анализ ---
# Определение функции предсказания для SHAP
def model_predict(X):
    return predict(X, weights, bias)

# Использование KernelExplainer для объяснения модели
explainer = shap.KernelExplainer(model_predict, shap.sample(X_train, 100))  # Используем подвыборку из обучающей выборки

# Расчёт SHAP значений для тестовой выборки
shap_values = explainer.shap_values(X_test, nsamples=100)  # Количество сэмплов для оценки

# Визуализация SHAP значений
feature_names = data_balanced.drop('Class', axis=1).columns

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Зависимость признака V14 от предсказаний
shap.dependence_plot("V14", shap_values, X_test, feature_names=feature_names)

# Force plot для отдельной транзакции
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0, :], feature_names=feature_names)
