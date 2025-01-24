import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Загрузка данных
data = pd.read_csv('./data/creditcard.csv')

# Разделение данных на классы
data_class_0 = data[data['Class'] == 0]
data_class_1 = data[data['Class'] == 1]

# Масштабирование данных
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

X = data.drop('Class', axis=1).values
y = data['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Функция для оценки метрик
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return accuracy, precision, recall, f1, roc_auc

# Метрики для всех подходов
metrics = {
    "Approach": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1": [],
    "ROC-AUC": []
}

# 1. Oversampling через SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model_smote = LogisticRegression(random_state=42, max_iter=1000)
model_smote.fit(X_train_smote, y_train_smote)
metrics_smote = evaluate_model(model_smote, X_test, y_test)

metrics["Approach"].append("SMOTE Oversampling")
metrics["Accuracy"].append(metrics_smote[0])
metrics["Precision"].append(metrics_smote[1])
metrics["Recall"].append(metrics_smote[2])
metrics["F1"].append(metrics_smote[3])
metrics["ROC-AUC"].append(metrics_smote[4])

# 2. Undersampling
data_class_0_undersampled = resample(data_class_0, 
                                     replace=False, 
                                     n_samples=len(data_class_1), 
                                     random_state=42)
data_undersampled = pd.concat([data_class_0_undersampled, data_class_1])
X_undersampled = data_undersampled.drop('Class', axis=1).values
y_undersampled = data_undersampled['Class'].values
X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(
    X_undersampled, y_undersampled, test_size=0.3, random_state=42)

model_undersample = LogisticRegression(random_state=42, max_iter=1000)
model_undersample.fit(X_train_under, y_train_under)
metrics_undersample = evaluate_model(model_undersample, X_test, y_test)

metrics["Approach"].append("Undersampling")
metrics["Accuracy"].append(metrics_undersample[0])
metrics["Precision"].append(metrics_undersample[1])
metrics["Recall"].append(metrics_undersample[2])
metrics["F1"].append(metrics_undersample[3])
metrics["ROC-AUC"].append(metrics_undersample[4])

# 3. Перевзвешивание классов
class_weights = {0: 1, 1: len(data_class_0) / len(data_class_1)}
model_weighted = LogisticRegression(random_state=42, max_iter=1000, class_weight=class_weights)
model_weighted.fit(X_train, y_train)
metrics_weighted = evaluate_model(model_weighted, X_test, y_test)

metrics["Approach"].append("Class Weighting")
metrics["Accuracy"].append(metrics_weighted[0])
metrics["Precision"].append(metrics_weighted[1])
metrics["Recall"].append(metrics_weighted[2])
metrics["F1"].append(metrics_weighted[3])
metrics["ROC-AUC"].append(metrics_weighted[4])

# Построение графиков
metrics_df = pd.DataFrame(metrics)

plt.figure(figsize=(12, 6))
for metric in ["Precision"]:
    plt.plot(metrics_df["Approach"], metrics_df[metric], marker='o')

plt.xlabel("\nApproach")
plt.ylabel("Precision value")
plt.title("Comparison of data balancing techniques")
plt.legend()
plt.grid()
plt.show()
