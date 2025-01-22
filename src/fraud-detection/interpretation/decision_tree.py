import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import shap
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
scaler = StandardScaler()
data_balanced['Amount'] = scaler.fit_transform(data_balanced['Amount'].values.reshape(-1, 1))

# Разделение на тренировочную и тестовую выборки
X = data_balanced.drop('Class', axis=1).values
y = data_balanced['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение дерева решений с использованием sklearn
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)

# Оценка метрик
y_pred = tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, tree_model.predict_proba(X_test)[:, 1])

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Использование SHAP для анализа важности признаков
explainer = shap.TreeExplainer(tree_model)
shap_values = explainer.shap_values(X_test)

shap_values_for_class_1 = shap_values[:, :, 1]  # Извлечение значений для класса 1

# Проверка формы
print(f"SHAP values for class 1 shape: {shap_values_for_class_1.shape}")
print(f"X_test shape: {X_test.shape}")

# Визуализация SHAP summary plot
shap.summary_plot(
    shap_values_for_class_1,
    X_test,
    feature_names=data_balanced.drop('Class', axis=1).columns
)

# Визуализация для конкретного примера
sample_index = 0  # Замените на нужный индекс
shap.force_plot(
    explainer.expected_value[1],  # Ожидаемое значение модели для класса 1
    shap_values_for_class_1[sample_index],
    X_test[sample_index],
    feature_names=data_balanced.drop('Class', axis=1).columns
)