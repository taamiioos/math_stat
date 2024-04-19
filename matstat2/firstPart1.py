import numpy as np
from scipy.optimize import  minimize_scalar

# Фиксируем количество испытаний
m = 1000

# Генерируем выборку результатов испытаний (0 или 1)
x = np.random.choice([0, 1], size=m)


# Определяем функцию логарифмической правдоподобия
def log_likelihood(p, x):
    log_likelihood = 0
    for i in x:
        if 0 < p < 1:
            # Суммируем логарифмы вероятностей для каждого наблюдения
            log_likelihood += i * np.log(p) + (1 - i) * np.log(1 - p)
        else:
            log_likelihood += 0
    # Возвращаем отрицательное значение, так как minimize ищет минимум функции
    return -log_likelihood


def estimate_p(x):
    # Минимизация отрицательного логарифма правдоподобия
    result = minimize_scalar(log_likelihood, args=x, bounds=(0, 1), method='bounded')
    return result.x


p = estimate_p(x)

print("Оценка параметра p методом максимального правдоподобия:", p)
