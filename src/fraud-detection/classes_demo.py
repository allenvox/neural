import pandas as pd
import matplotlib.pyplot as plt

# Загрузка датасета
data = pd.read_csv('./data/creditcard.csv')

# Разделение данных по классам
data_class_0 = data[data['Class'] == 0]
data_class_1 = data[data['Class'] == 1]

# Подсчёт количества записей в каждом классе
class_counts = [len(data_class_0), len(data_class_1)]
class_labels = ['Нормальные (0)', 'Мошеннические (1)']

# Создание столбчатой диаграммы
plt.figure(figsize=(8, 6))  # Установка размера графика
plt.bar(class_labels, class_counts, color=['blue', 'red'], width=0.5)

# Настройка осей и заголовка
plt.xlabel('Классы', fontsize=12)
plt.ylabel('Количество транзакций', fontsize=12)
plt.title('Распределение классов в наборе данных', fontsize=14)

# Добавление значений над столбцами
for i, count in enumerate(class_counts):
    plt.text(i, count + 5000, str(count), ha='center', va='bottom', fontsize=10)

# Настройка сетки и отображение графика
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Отображение графика
plt.show()
