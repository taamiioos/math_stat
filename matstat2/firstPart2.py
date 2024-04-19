import numpy as np
import matplotlib.pyplot as plt
from firstPart1 import estimate_p

# Параметры биномиального распределения
m = 4
theta_true = 1/5

# Задаем массив объемов выборки
n_array = np.array([10, 50, 100, 500, 1000, 2000, 4000])

# Задаем порог для определения отличающихся оценок
threshold = 0.1

# Список для хранения смещения
bias_list = []

# Список для хранения дисперсии оценок
variance_list = []

# Список для хранения среднеквадратической ошибки
rmse_list = []

# Список для хранения количества отличающихся оценок
diff_count_list = []

# Список для хранения оценок параметра theta
theta_estimates = []

# Список для хранения разницы между оценкой и реальным параметром
diff_list = []

# Для каждого объема выборки
for n in n_array:
    # Генерируем выборку из биномиального распределения
    x = np.random.binomial(m, theta_true, n)

    # Находим оценку параметра методом максимального правдоподобия
    p = estimate_p(x)

    # Добавляем оценку параметра в список
    theta_estimates.append(p)

    # Добавляем разницу между оценкой и реальным параметром в список
    diff = p - theta_true
    diff_list.append(diff)

    # Преобразуем список diff_list в массив NumPy
    diff_array = np.array(diff_list)

    # Вычисляем выборочные характеристики для разницы между оценкой и реальным параметром
    bias = np.mean(diff_list)
    variance = np.var(diff_list)
    rmse = np.sqrt(np.mean(diff_array**2))

    # Вычисляем количество выборок, для которых оценка отличается от реального параметра более чем на заданный порог
    diff_count = np.sum(np.abs(diff_list) > threshold)
    # Добавляем вычисленные характеристики в списки
    bias_list.append(bias)
    variance_list.append(variance)
    rmse_list.append(rmse)
    diff_count_list.append(diff_count)


# Выводим вычисленные характеристики
print(f'Смещение: {bias:.4f}')
print(f'Дисперсия: {variance:.4f}')
print(f'Среднеквадратическая ошибка: {rmse:.4f}')
print(f'Количество отличающихся оценок: {diff_count}')
print()



# Строим графики зависимости выборочных характеристик от объема выборки
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(n_array, bias_list, marker='o')
plt.xlabel('Объем выборки')
plt.ylabel('Смещение')
plt.subplot(1, 2, 2)
plt.plot(n_array, variance_list, marker='o')
plt.xlabel('Объем выборки')
plt.ylabel('Дисперсия')
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(n_array, rmse_list, marker='o')
plt.xlabel('Объем выборки')
plt.ylabel('Среднеквадратическая ошибка')
plt.subplot(1, 2, 2)
plt.plot(n_array, diff_count_list, marker='o')
plt.xlabel('Объем выборки')
plt.ylabel('Количество отличающихся оценок')
plt.show()

print("Свойства оценок")
print("Смещение должно стремиться к 0 при увеличении объема выборки, чтобы оценка была ассимптотически несмещенной")
print("Несмещенная")
print("Дисперсия должна стремиться к 0 при увеличении объема выборки, чтобы оценка была состоятельной")
print("Состоятельная")
print("Среднеквадратическая ошибка должна к 0 при увеличении объема выборки, чтобы оценка была эффективной")
print("Эффективная")

