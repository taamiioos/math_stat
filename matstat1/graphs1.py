import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns
import main

fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# гистограмма выборочного среднего
axs[0].hist(main.selections_mean, bins=100, density=True, rwidth=1, alpha=0.5, color='red', label='Гистограмма', edgecolor='black', linewidth=1.2)
axs[0].set_xlabel('Значение выборочного среднего')
axs[0].set_ylabel('Плотность')
sns.kdeplot(main.selections_mean, color='black', lw=2, ax=axs[0], label='Плотность вероятности')
x = np.linspace(main.selections_mean.min(), main.selections_mean.max(), 100)
axs[0].plot(x, norm.pdf(x, main.selections_mean.mean(), main.selections_mean.std()), label='Нормальное распределение')
axs[0].set_title('Выборочное среднее')
axs[0].legend()

# гистограмма выборочной дисперсии
axs[1].hist(main.selections_var, bins=100, density=True, rwidth=1, alpha=0.5, color='red', label='Гистограмма', edgecolor='black', linewidth=1.2)
axs[1].set_xlabel('Значение выборочной дисперсии')
axs[1].set_ylabel('Плотность')
sns.kdeplot(main.selections_var, color='black', lw=2, ax=axs[1], label='Плотность вероятности')
x = np.linspace(main.selections_var.min(), main.selections_var.max(), 100)
axs[1].plot(x, norm.pdf(x, main.selections_var.mean(), main.selections_var.std()), label='Нормальное распределение')
axs[1].set_title('Выборочная дисперсия')
axs[1].legend()

plt.show()