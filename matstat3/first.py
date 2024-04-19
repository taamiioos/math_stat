import numpy as np
from scipy.stats import f


# мат.ожидания
mu1 = 0
mu2 = 0

# Дисперсии
sigma1_sq = 2
sigma2_sq = 1

# Уровень значимости
alpha = 0.05

# Размеры выборок
sample_sizes = [25, 10000]

# Количество экспериментов
experiments = 1000

# Реальное значение параметра
real_tau = sigma1_sq / sigma2_sq

# Функция для вычисления доверительного интервала
def confidence_interval(X, Y, alpha):
    n1 = len(X)
    n2 = len(Y)
    # Выборочная дисперсии
    s1_sq = np.var(X, ddof=1)
    s2_sq = np.var(Y, ddof=1)
    # Степени свободы
    df1 = n1-1
    df2 = n2-1
    # Квантили рапсределения Фишера
    quantil1 = f.ppf(1 - alpha/2, df1, df2)
    quantil2 = f.ppf(alpha/2, df1, df2)
    # Нижняя и верхняя граница доверительного интервала
    lower_bound = (s1_sq / s2_sq) * (1 / quantil1)
    upper_bound = (s1_sq / s2_sq) * (1 / quantil2)
    return lower_bound, upper_bound

# Функция для проведения экспериментов
def experiment(mu1, mu2, real_tau, alpha, sample_size, experiments):
    covered = 0
    for _ in range(experiments):
        # Генерация выборок
        X = np.random.normal(mu1, np.sqrt(sigma1_sq), sample_size)
        Y = np.random.normal(mu2, np.sqrt(sigma2_sq), sample_size)
        # Вычисление доверительного интервала
        interval = confidence_interval(X, Y, alpha)
        # Проверка покрытия реального значения параметра
        if interval[0] <= real_tau <= interval[1]:
            covered += 1
    # Вычисление вероятности покрытия реального значения параметра
    coverage_probability = covered / experiments
    return coverage_probability

# Проведение экспериментов для разных объемов выборки
for sample_size in sample_sizes:
    coverage_probability = experiment(mu1, mu2, real_tau, alpha, sample_size, experiments)
    print(f"Для выборки объемом {sample_size:}")
    print(f"Вероятность покрытия реального значения параметра: {coverage_probability}")
