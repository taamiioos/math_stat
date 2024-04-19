import numpy as np

# размер выборки
size_selections = 1000
# количество выборок
num_selections = 10000
# параметр пуассона
lam = 5
# генерируем выборки по распределию Пуассона
selections = np.random.poisson(lam=lam, size=(num_selections, size_selections))

# выборочное среднее(axis=1, тк по строкам находим, то есть по выборкам)
selections_mean = np.mean(selections, axis=1)

# выборочная дисперсия
selections_var = np.var(selections, axis=1)

# выборочный квантиль порядка 0,5
selections_quantile = np.quantile(selections, 0.5)

# математическое ожидание для каждой статистики
math_expectation_mean = np.mean(selections_mean)
math_expectation_var = np.mean(selections_var)
math_expectation_quantile = np.mean(selections_quantile)

# дисперсия для каждой статистики
dispersion_mean = np.var(selections_mean)
dispersion_var = np.var(selections_var)
dispersion_quantile = np.var(selections_quantile)

# медиана для каждой статистики
median_mean = np.median(selections_mean)
median_var = np.median(selections_var)
median_quantile = np.median(selections_quantile)

# коэффициент асимметрии(он показывает насколько ассиметрично расспеределение относительно мат ожидание и дисперсия,
# если оно равно 0 значит распределение нормальное(не факт))
skewness_mean = np.mean((selections_mean - math_expectation_mean) ** 3) / dispersion_mean ** 1.5
skewness_var = np.mean((selections_var - math_expectation_var) ** 3) / dispersion_var ** 1.5

# коэффициент эксцесса(он показывает насколько острый или тупой пик, если +, то острее, если - тупее, если 0, то близко к нормальному)
kurtosis_mean = np.mean((selections_mean - math_expectation_mean) ** 4) / dispersion_mean ** 2 - 3
kurtosis_var = np.mean((selections_var - math_expectation_var) ** 4) / dispersion_var ** 2 - 3



print(f"Математическое ожидание выборочного среднего: {math_expectation_mean}")
print(f"Дисперсия выборочного среднего: {dispersion_mean}")
print(f"Медиана выборочного среднего: {median_mean}")
print(f"Математическое ожидание выборочной дисперсии: {math_expectation_var}")
print(f"Дисперсия выборочной дисперсии: {dispersion_var}")
print(f"Медиана выборочной дисперсии: {median_var}")
print(f"Математическое ожидание выборочного квантиля порядка 0,5: {math_expectation_quantile}")
print(f"Дисперсия выборочного квантиля порядка 0,5: {dispersion_quantile}")
print(f"Медиана выборочного квантиля порядка 0,5: {median_quantile}")
print(f"Коэффициент асимметрии выборочного среднего: {skewness_mean}")
print(f"Коэффициент эксцесса выборочного среднего: {kurtosis_mean}")
print(f"Коэффициент асимметрии выборочной дисперсии: {skewness_var}")
print(f"Коэффициент эксцесса выборочной дисперсии: {kurtosis_var}")
