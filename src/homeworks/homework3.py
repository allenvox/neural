import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Подготовка данных
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# Параметры модели
input_size = 28 * 28   # Размер входного слоя (размер изображения 28x28)
hidden_size = 128      # Количество нейронов в скрытом слое
output_size = 10       # Количество классов (0-9)

# Определение нейронной сети с использованием nn.Module
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   # Первый полносвязный слой
        self.fc2 = nn.Linear(hidden_size, output_size)  # Второй полносвязный слой
        self.dropout = nn.Dropout(p=0.5)                # Dropout для регуляризации
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01) # Leaky ReLU

    def forward(self, x):
        x = self.fc1(x)                # Проход через первый слой
        x = self.leaky_relu(x)         # Активация Leaky ReLU
        x = self.dropout(x)            # Dropout
        x = self.fc2(x)                # Проход через второй слой
        return x                       # Без softmax, так как используется CrossEntropy

# Инициализация модели, функции потерь и оптимизатора
model = NeuralNet(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()               # Функция потерь (встроенная cross-entropy)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Оптимизатор Adam

# Функция для вычисления точности
def calculate_accuracy(output, labels):
    _, predicted = torch.max(output, 1)
    correct = (predicted == labels).float().sum()
    accuracy = correct / labels.size(0)
    return accuracy

# Цикл обучения
num_epochs = 50
for epoch in range(num_epochs):
    correct_predictions = 0
    total_predictions = 0
    for batch_images, batch_labels in train_loader:
        batch_images = batch_images.view(-1, 28*28).to(device)  # Векторизация изображений
        batch_labels = batch_labels.to(device)
        
        # Forward pass
        output = model(batch_images)
        loss = criterion(output, batch_labels)  # Вычисление ошибки
        
        # Backward pass и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Вычисление точности
        accuracy = calculate_accuracy(output, batch_labels)
        correct_predictions += accuracy.item() * batch_labels.size(0)
        total_predictions += batch_labels.size(0)

    avg_accuracy = correct_predictions / total_predictions
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {avg_accuracy:.4f}')

print("Training complete.")
