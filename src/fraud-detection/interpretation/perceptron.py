import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from lime.lime_tabular import LimeTabularExplainer  # Импорт LIME

# Загрузка данных
data = pd.read_csv('./data/creditcard.csv')

# Балансировка классов с помощью SMOTE
X = data.drop(columns=['Class']).values
y = data['Class'].values
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Масштабирование данных
scaler = StandardScaler()
X_smote = scaler.fit_transform(X_smote)

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=42)

# Преобразование данных в тензоры
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# DataLoader для батчей
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Определение многослойного перцептрона
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),  # Нормализация для стабилизации
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()  # Для предсказания вероятностей
        )

    def forward(self, x):
        return self.model(x)

# Инициализация модели
input_size = X_train.shape[1]
model = MLP(input_size)

# Функция потерь и оптимизатор
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # AdamW для лучшей регуляризации

# Сборка результатов
num_epochs = 100
accumulation_steps = 4  # Накопление градиентов для уменьшения шума

train_losses = []
val_roc_auc = []
train_time = []
best_roc_auc = 0
early_stop_counter = 0

# Цикл обучения
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_start = time.time()

    optimizer.zero_grad()
    for step, (X_batch, y_batch) in enumerate(train_loader):
        y_pred = model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        loss = loss / accumulation_steps  # Делим loss на количество шагов для накопления
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))
    epoch_end = time.time()
    train_time.append(epoch_end - epoch_start)

    # Оценка на валидации
    model.eval()
    y_test_pred = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            y_batch_pred = model(X_batch).squeeze()
            y_test_pred.extend(y_batch_pred.tolist())

    # Расчет ROC-AUC
    roc_auc = roc_auc_score(y_test, y_test_pred)
    val_roc_auc.append(roc_auc)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_losses[-1]:.4f}, ROC-AUC: {roc_auc:.4f}")

    # Early Stopping
    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        early_stop_counter = 0
        best_model = model.state_dict()
    else:
        early_stop_counter += 1

    if early_stop_counter >= 30:
        print("Early stopping triggered!")
        break

# Загрузка лучших весов
model.load_state_dict(best_model)

# Финальная оценка метрик
model.eval()
final_y_pred = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        y_batch_pred = model(X_batch).squeeze()
        final_y_pred.extend((y_batch_pred > 0.5).float().tolist())

accuracy = accuracy_score(y_test, final_y_pred)
precision = precision_score(y_test, final_y_pred)
recall = recall_score(y_test, final_y_pred)
f1 = f1_score(y_test, final_y_pred)
roc_auc = roc_auc_score(y_test, y_test_pred)

print(f"\nFinal Test Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Интерпретация с помощью LIME
def predict_proba(X):
    """Функция для получения вероятностей из модели (для LIME)"""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        probs = model(X_tensor).squeeze().numpy()
    return np.vstack([1 - probs, probs]).T

# Создание LIME объяснителя
explainer = LimeTabularExplainer(
    X_train,
    feature_names=data.drop(columns=['Class']).columns,
    class_names=['Non-Fraud', 'Fraud'],
    discretize_continuous=True
)

# Выбор экземпляра для объяснения
sample_index = 42  # Замените на нужный индекс
sample = X_test[sample_index]
explanation = explainer.explain_instance(sample, predict_proba, num_features=10)

# Визуализация результатов LIME
explanation.save_to_file('lime_explanation.html')
print("LIME explanation saved to lime_explanation.html")
