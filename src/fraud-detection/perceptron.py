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
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Инициализация модели
input_size = X_train.shape[1]
model = MLP(input_size)

# Определение функции потерь и оптимизатора
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 20
train_losses = []
val_roc_auc = []
train_time = []

for epoch in range(num_epochs):
    model.train()
    epoch_start = time.time()
    epoch_loss = 0.0
    
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

    # Оценка на тестовом наборе
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

# Построение графиков
plt.figure(figsize=(12, 6))

# График ошибки
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# График ROC-AUC
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), val_roc_auc, label='Validation ROC-AUC')
plt.title('Validation ROC-AUC Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('ROC-AUC')
plt.legend()

plt.tight_layout()
plt.show()

# График кумулятивного времени
cumulative_time = np.cumsum(train_time)
plt.figure(figsize=(6, 4))
plt.plot(range(1, num_epochs + 1), cumulative_time, label='Cumulative Train Time (s)')
plt.title('Cumulative Training Time')
plt.xlabel('Epoch')
plt.ylabel('Time (s)')
plt.legend()
plt.show()
