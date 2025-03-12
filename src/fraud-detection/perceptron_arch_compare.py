import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Загрузка данных
data = pd.read_csv('./data/creditcard.csv')
X = data.drop(columns=['Class']).values
y = data['Class'].values

# Балансировка с помощью SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Масштабирование
scaler = StandardScaler()
X_smote = scaler.fit_transform(X_smote)

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=42)

# Преобразование в тензоры
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Класс MLP с изменяемой архитектурой
class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.2):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:  # Не добавляем активацию и dropout в последний слой
                layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Sigmoid())  # Выходной слой
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Список архитектур для тестирования
architectures = [
    [30, 64, 1],              # Простая: 1 скрытый слой
    [30, 128, 64, 1],         # Средняя: 2 скрытых слоя
    [30, 128, 64, 32, 1],     # Базовая (ваша): 3 скрытых слоя
    [30, 256, 128, 64, 32, 1] # Глубокая: 4 скрытых слоя
]
architecture_names = ["1 скрытый (64)", "2 скрытых (128-64)", "3 скрытых (128-64-32)", "4 скрытых (256-128-64-32)"]

# Параметры обучения
num_epochs = 50  # Уменьшено для ускорения экспериментов
criterion = nn.BCELoss()
results = {"loss": [], "roc_auc": [], "train_time": []}

# Тестирование каждой архитектуры
for i, layer_sizes in enumerate(architectures):
    print(f"\nTesting architecture: {architecture_names[i]}")
    model = MLP(layer_sizes)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    train_losses = []
    val_roc_auc = []
    train_time = []
    best_roc_auc = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))
        epoch_end = time.time()
        train_time.append(epoch_end - epoch_start)

        # Оценка на тесте
        model.eval()
        y_test_pred = []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                y_batch_pred = model(X_batch).squeeze()
                y_test_pred.extend(y_batch_pred.tolist())
        roc_auc = roc_auc_score(y_test, y_test_pred)
        val_roc_auc.append(roc_auc)

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc

    results["loss"].append(train_losses[-1])  # Итоговая потеря
    results["roc_auc"].append(best_roc_auc)  # Лучший ROC-AUC
    results["train_time"].append(np.sum(train_time))  # Общее время

# Графики сравнения архитектур
plt.figure(figsize=(12, 6))
plt.bar(architecture_names, results["roc_auc"], color='skyblue')
plt.xlabel('Архитектура MLP')
plt.ylabel('ROC-AUC')
plt.title('Зависимость ROC-AUC от архитектуры MLP')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('mlp_roc_auc_comparison.png')
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(architecture_names, results["loss"], color='lightcoral')
plt.xlabel('Архитектура MLP')
plt.ylabel('Итоговая функция потерь')
plt.title('Зависимость функции потерь от архитектуры MLP')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('mlp_loss_comparison.png')
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(architecture_names, results["train_time"], color='lightgreen')
plt.xlabel('Архитектура MLP')
plt.ylabel('Время обучения (сек)')
plt.title('Зависимость времени обучения от архитектуры MLP')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('mlp_time_comparison.png')
plt.show()

# Вывод лучших результатов
best_idx = np.argmax(results["roc_auc"])
print(f"\nЛучшая архитектура: {architecture_names[best_idx]}")
print(f"ROC-AUC: {results['roc_auc'][best_idx]:.4f}")
print(f"Loss: {results['loss'][best_idx]:.4f}")
print(f"Время обучения: {results['train_time'][best_idx]:.2f} сек")
