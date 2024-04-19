import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, expon

# выборка из экспоненциального распределения(тк нам нужно убедиться в сходимости эмпирических распредеелений к теоритическим с помощью гамма распределения и экспотенциального
samples = np.random.exponential(scale=1, size=(100000, 10))

# X(2) вариационный ряд(возрастающий). берем 2 член ряда
X2 = np.sort(samples, axis=1)[:, 1]

# nF(X(2)) вычисляем значение случайной величины(формула эмпирической функции)
n = 10
FX2 = n * (1 - np.exp(-X2))
# cтроим гистограмму эмпирической функции и накладываем на нее плотность гамма-распредления
plt.hist(FX2, bins=50, density=True, alpha=0.5, label='Гистограмма nF(X(2))')
x = np.linspace(0, 20, 100)
plt.plot(x, gamma.pdf(x, 2, scale=1), label='Γ(2,1)')
plt.legend()
plt.show()



# nF(X(2)) -> U1 ~ Г(2,1) n(1 - F(X(n))) -> U2 ~ Г (1,1) = Exp(1).
# U- случ.вел, которая аппркосимирует значение из выборки с помощью как раз таки гамма функции


# n(1 - F(X(n))) последний член ряда
Xn = np.max(samples, axis=1)
U2 = n * (np.exp(-Xn))

# cтроим гистограмму эмпирической функции и накладываем на нее плотность гамма-распредления(экспотенциального)
plt.hist(U2, bins=50, density=True, alpha=0.5, label='Гистограмма n(1 - F(X(n)))')
x = np.linspace(0, 10, 100)
plt.plot(x, expon.pdf(x, scale=1), label='Г(1,1) ')
plt.legend()
plt.show()
