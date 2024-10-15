# Импорт необходимых библиотек
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

# Установка устройства для вычислений (GPU, если доступно)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка и предобработка MNIST датасета
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

# Выбор небольшого батча для обучения
data_iter = iter(train_loader)
images, labels = next(data_iter)
images = images.view(-1, 28*28).to(device)  # Преобразование изображений в вектор (batch_size, 28*28)
labels = labels.to(device)

# Размеры нейросети
input_size = 28 * 28  # Размер входного слоя (размер изображения 28x28)
hidden_size = 128     # Количество нейронов в скрытом слое
output_size = 10      # Количество классов (0-9)

# Инициализация весов (случайные)
W1 = torch.randn(input_size, hidden_size, device=device, requires_grad=True) * 0.01
b1 = torch.zeros(hidden_size, device=device, requires_grad=True)

W2 = torch.randn(hidden_size, output_size, device=device, requires_grad=True) * 0.01
b2 = torch.zeros(output_size, device=device, requires_grad=True)

# Прямой проход (Forward Propagation)
def forward(images):
    # Первый слой (вход -> скрытый слой)
    z1 = torch.matmul(images, W1) + b1  # Взвешенная сумма входных данных
    a1 = torch.relu(z1)  # Применение функции активации ReLU
    
    # Второй слой (скрытый слой -> выходной слой)
    z2 = torch.matmul(a1, W2) + b2  # Взвешенная сумма скрытого слоя
    output = torch.softmax(z2, dim=1)  # Применение softmax для получения вероятностей классов
    return output, a1

# Прямой проход на примере батча
output, hidden_activation = forward(images)

# Функция потерь (Cross-Entropy Loss)
def compute_loss(output, labels):
    labels_one_hot = F.one_hot(labels, num_classes=output_size).float()
    loss = -torch.sum(labels_one_hot * torch.log(output)) / output.size(0)
    return loss

# Расчет потерь
loss = compute_loss(output, labels)
print(f"Loss: {loss.item()}")

def one_hot_encode(labels, num_classes=10):
    # Создаем матрицу нулей размером (количество меток, количество классов)
    one_hot = torch.zeros((labels.size(0), num_classes))
    one_hot[torch.arange(labels.size(0)), labels.long()] = 1
    return one_hot

def backpropagation(images, hidden_activation, output, labels, W1, W2, b1, b2, learning_rate=0.00001):
    # Преобразуем метки в one-hot код
    labels_one_hot = one_hot_encode(labels)
    #print("Output size:", output.size())
    #print("One-hot labels size:", labels_one_hot.size())
    
    # Рассчитаем ошибку (разницу между предсказанным выходом и реальными метками)
    dL_dz2 = output - labels_one_hot  # Градиент ошибки по выходу сети
    
    # Градиент по весам и смещениям для второго (выходного) слоя
    dL_dW2 = torch.matmul(hidden_activation.T, dL_dz2)
    dL_db2 = torch.sum(dL_dz2, axis=0)
    
    # Обновляем веса и смещения второго слоя
    W2 = W2 - learning_rate * dL_dW2
    b2 = b2 - learning_rate * dL_db2
    
    # Градиент по скрытому слою
    dL_da1 = torch.matmul(dL_dz2, W2.T)  # Градиент потерь по активациям скрытого слоя
    dL_dz1 = dL_da1 * (hidden_activation > 0)  # Учитываем производную ReLU
    
    # Градиент по весам и смещениям первого слоя
    dL_dW1 = torch.matmul(images.T, dL_dz1)
    dL_db1 = torch.sum(dL_dz1, axis=0)
    
    # Обновляем веса и смещения первого слоя
    W1 = W1 - learning_rate * dL_dW1
    b1 = b1 - learning_rate * dL_db1
    
    return W1, W2, b1, b2

# Пример вызова функции обратного распространения
W1, W2, b1, b2 = backpropagation(images, hidden_activation, output, labels, W1, W2, b1, b2)

# Выполнение нескольких итераций обучения
num_epochs = 50

for epoch in range(num_epochs):
    for batch_images, batch_labels in train_loader:
        batch_images = batch_images.view(-1, 28*28).to(device)
        batch_labels = batch_labels.to(device)

        # Прямой проход
        output, hidden_activation = forward(batch_images)
        
        # Потери
        loss = compute_loss(output, batch_labels)
        
        # Обратный проход
        W1, W2, b1, b2 = backpropagation(images, hidden_activation, output, labels, W1, W2, b1, b2)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    if torch.isnan(loss).any():
        print("Loss is NaN!")
        break  # Прерываем цикл, если находим NaN

print("Training complete.")
