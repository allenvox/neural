# поиграться со свертками 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

batch_size = 64
learning_rate = 0.001
num_epochs = 20

transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразование в тензор
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Нормализация
]) # [-1;1]

# Canadian Institute For Advanced Research dataset 10
# 60000 32x32 colour images in 10 classes, 6000 images per class, train/test = 50000/10000
# classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Первый сверточный слой
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Второй сверточный слой
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPooling reduces image with window
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Полносвязный слой
        self.fc2 = nn.Linear(128, 10)  # Выходной слой (10 классов)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Свертка + ReLU + MaxPooling
        x = self.pool(torch.relu(self.conv2(x)))  # Свертка + ReLU + MaxPooling
        x = x.view(-1, 64 * 8 * 8)  # Разворачиваем вектор перед подачей на полносвязный слой
        x = torch.relu(self.fc1(x))  # Полносвязный слой + ReLU
        x = self.fc2(x)  # Выходной слой
        return x

model = SimpleCNN()

criterion = nn.CrossEntropyLoss()  # Функция потерь
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def compute_accuracy(loader, model):
    model.eval()  # Режим оценки
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()  # Возвращаемся в режим обучения
    return 100 * correct / total

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Прямой проход
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Обратное распространение и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Вывод промежуточного лосса
        running_loss += loss.item()
        if (i + 1) % 100 == 0:  # Каждые 100 батчей
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

    # Вычисление точности на тренировочной выборке
    train_accuracy = compute_accuracy(train_loader, model)
    print(f"Epoch [{epoch+1}/{num_epochs}] completed. Training Accuracy: {train_accuracy:.2f}%")

print("Training finished.")
