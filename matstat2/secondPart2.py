import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import beta

# Параметры биномиального распределения
a=b=1
theta_true = 0.5

# Задаем массив объемов выборки
n_array = np.array([10, 50, 100, 500, 1000, 2000])

# Задаем порог для определения отличающихся оценок
threshold = 0.001

# Задаем количество повторений эксперимента для каждого объема выборки
num_experiments = 100

# Списки для хранения результатов
bias_list = [] #смещение
variance_list = [] #дисперсия
rmse_list = [] #ско
diff_count_list = [] #кол-во отличающихся оценок
theta_estimates = [] # массив наших значений тета
diff_list = [] # все оценки которые отличаются на более чем заданный порог от реального

# Для каждого объема выборки
for n in n_array:
    # Для каждого повторения эксперимента
    for _ in range(num_experiments):
        # Генерируем выборку из биномиального распределения
        x = np.random.geometric(theta_true, n)


        def geometric_sample(n, theta):
            return np.random.geometric(theta, n) + 1
        def geometric_likelihood(x, theta):
            n = len(x)
            return theta ** n * (1 - theta) ** (sum(x) - n)
        def aprior(theta, alpha, b):
            return beta.pdf(theta, alpha, b)
        def normalizing_constant(x, alpha, beta):
            # Вычисляем интеграл по всей области определения параметра theta
            # от произведения априорного распределения и функции правдоподобия
            integrand = lambda theta: aprior(theta, alpha, beta) * geometric_likelihood(x, theta)
            integral = quad(integrand, 0, 1)[0]
            return integral
        def expected_value(posterior):
            # Вычисляем интеграл от произведения апостериорного распределения и значения параметра theta
            integrand = lambda theta: theta * posterior(theta)
            integral = quad(integrand, 0, 1)[0]
            return integral


        a = np.random.uniform(0.1, 10)
        b = np.random.uniform(0.1, 10)
        samples = geometric_sample(50, theta_true)

        aprior_value = aprior(theta_true, a, b)
        normalizing_constant_value = normalizing_constant(samples, a, b)
        aposterior_value = lambda theta: (aprior(theta, a, b) * geometric_likelihood(samples,
                                                                                     theta)) / normalizing_constant_value

        expected_theta = expected_value(aposterior_value)

        # Добавляем оценку параметра в список
        theta_estimates.append(expected_theta)

        # Добавляем разницу между оценкой и реальным параметром в список
        diff = abs(expected_theta - theta_true)
        diff_list.append(diff)

    # Преобразуем список diff_list в массив NumPy
    diff_array = np.array(diff_list)

    # Вычисляем выборочные характеристики для разницы между оценкой и реальным параметром
    bias = np.mean(diff_list) #смещение
    variance = np.var(diff_list) # диспресия
    rmse = np.sqrt(np.mean(diff_array**2)) #ско

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