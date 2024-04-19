import numpy as np
from scipy.stats import uniform

# Заданные параметры

# Параметр распределения
theta = 5
# Уровень значимости
alpha = 0.05

# Размеры выборок
sample_sizes = [25, 10000]

# Количество экспериментов
experiments = 1000


# Функция для вычисления асимптотического доверительного интервала
def asymptotic_confidence_interval(sample, alpha):
    n = len(sample)
    sorted_sample = np.sort(sample)
    # Минимальное значение выборки
    min_val = sorted_sample[0]
    # Максимальное значение выборки
    max_val = sorted_sample[-1]
    # Нижняя и верхняя граница доверительного интервала
    lower_bound = min_val - theta / np.sqrt(2 * n * np.log(2 / alpha))
    upper_bound = max_val + theta / np.sqrt(2 * n * np.log(2 / alpha))
    return lower_bound, upper_bound


# Функция для генерации выборок и проверки покрытия интервалов
def experiment(sample_size, theta, alpha, experiments):
    covered = 0
    for _ in range(experiments):
        # Генерация выборки из равномерного распределения
        sample = uniform.rvs(loc=-theta, scale=2 * theta, size=sample_size)
        # Вычисление интервалае
        lower_bound, upper_bound = asymptotic_confidence_interval(sample, alpha)
        # Проверка покрытия интервала
        if theta >= lower_bound and theta <= upper_bound:
            covered += 1
    coverage_probability = covered / experiments
    return coverage_probability


# Проведение экспериментов для разных объемов выборки
for sample_size in sample_sizes:
    coverage_probability = experiment(sample_size, theta, alpha, experiments)
    print(f"Для выборки объемом {sample_size:}")
    print(f"Вероятность покрытия реального значения параметра: {coverage_probability}")


