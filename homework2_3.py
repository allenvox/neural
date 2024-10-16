import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

data_iter = iter(train_loader)
images, labels = next(data_iter)
images = images.view(-1, 28*28).to(device)  # Преобразование изображений в вектор (batch_size, 28*28)
labels = labels.to(device)

input_size = 28 * 28  # Размер входного слоя (размер изображения 28x28)
hidden_size = 128     # Количество нейронов в скрытом слое
output_size = 10      # Количество классов (0-9)

# Инициализация весов с использованием Xavier (Glorot) инициализации
W1 = torch.randn(input_size, hidden_size, device=device, requires_grad=True) * torch.sqrt(torch.tensor(2. / input_size))
b1 = torch.zeros(hidden_size, device=device, requires_grad=True)

W2 = torch.randn(hidden_size, output_size, device=device, requires_grad=True) * torch.sqrt(torch.tensor(2. / hidden_size))
b2 = torch.zeros(output_size, device=device, requires_grad=True)

def forward(images):
    # Первый слой (вход -> скрытый слой)
    z1 = torch.matmul(images, W1) + b1  # Взвешенная сумма входных данных
    a1 = torch.nn.functional.leaky_relu(z1, negative_slope=0.01)
    #a1 = torch.relu(z1)  # Применение функции активации ReLU
    
    # Второй слой (скрытый слой -> выходной слой)
    z2 = torch.matmul(a1, W2) + b2  # Взвешенная сумма скрытого слоя
    output = torch.softmax(z2, dim=1)  # Применение softmax для получения вероятностей классов
    return output, a1

def compute_loss(output, labels):
    loss = F.cross_entropy(output, labels)
    return loss

# Обратное распространение ошибки (Backpropagation)
def backpropagation(images, hidden_activation, output, labels, W1, W2, b1, b2, learning_rate=0.001):
    # Рассчитаем ошибку (разницу между предсказанным выходом и реальными метками)
    labels_one_hot = F.one_hot(labels, num_classes=output_size).float()
    
    # Градиент ошибки для выходного слоя
    dL_dz2 = output - labels_one_hot

    # Градиент по весам и смещениям для второго (выходного) слоя
    dL_dW2 = torch.matmul(hidden_activation.T, dL_dz2)
    dL_db2 = torch.sum(dL_dz2, axis=0)

    # Обновляем веса и смещения второго слоя
    W2 = W2.detach() - learning_rate * dL_dW2  # Отделяем от графа вычислений
    b2 = b2.detach() - learning_rate * dL_db2

    # Градиент по скрытому слою
    dL_da1 = torch.matmul(dL_dz2, W2.T)  # Градиент по активациям скрытого слоя
    dL_dz1 = dL_da1 * (hidden_activation > 0).float()  # Учитываем производную ReLU

    # Градиент по весам и смещениям первого слоя
    dL_dW1 = torch.matmul(images.T, dL_dz1)
    dL_db1 = torch.sum(dL_dz1, axis=0)

    # Обновляем веса и смещения первого слоя
    W1 = W1.detach() - learning_rate * dL_dW1  # Отделяем от графа вычислений
    b1 = b1.detach() - learning_rate * dL_db1

    return W1, W2, b1, b2

num_epochs = 50
learning_rate = 0.001

for epoch in range(num_epochs):
    for batch_images, batch_labels in train_loader:
        batch_images = batch_images.view(-1, 28*28).to(device)
        batch_labels = batch_labels.to(device)
        # Forward
        output, hidden_activation = forward(batch_images)
        # Loss
        loss = compute_loss(output, batch_labels)
        # Backprop
        W1, W2, b1, b2 = backpropagation(batch_images, hidden_activation, output, batch_labels, W1, W2, b1, b2, learning_rate)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    if torch.isnan(loss).any():
        print("Loss is NaN!")
        break

print("Training complete.")
