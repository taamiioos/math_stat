import numpy as np
from scipy.stats import beta
from scipy.integrate import quad


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


theta = 0.5
a = np.random.uniform(0.1, 10)
b = np.random.uniform(0.1, 10)
samples = geometric_sample(50, theta)

aprior_value = aprior(theta, a, b)
normalizing_constant_value = normalizing_constant(samples, a, b)
aposterior_value = lambda theta: (aprior(theta, a, b) * geometric_likelihood(samples,
                                                                             theta)) / normalizing_constant_value

expected_theta = expected_value(aposterior_value)
print("Байесовская оценка параметра: ", expected_theta)
