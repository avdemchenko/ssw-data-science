import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import math as mt
import scipy as sci


# Normal distribution generator
def normal_d(size=1000, mean=0, std_dev=1):
    return np.random.normal(loc=mean, scale=std_dev, size=size)


# Histogram
def histogram(data, bins=50, label="", xlabel="", ylabel="", title=""):
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='b', label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def plot(data, label="", xlabel="", ylabel=""):
    plt.plot(data, color="gray", alpha=0.6, marker="x", markersize=2, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def linear_law(size=1000, slope=0.1, intercept=2):
    time = np.arange(size)
    data = slope * time + intercept
    plt.plot(data, label='Модель з лінійним законом зміни')
    plt.xlabel('Час/Параметр')
    plt.ylabel('Значення')
    plt.title('Модель з лінійним законом зміни')
    plt.legend()
    plt.show()
    return data


def quadratic_law(size=1000):
    data = np.linspace(0, 10, size) ** 2
    plt.plot(data, label='Модель з квадратичним законом зміни')
    plt.xlabel('Час/Параметр')
    plt.ylabel('Значення')
    plt.title('Модель з квадратичним законом зміни')
    plt.legend()
    plt.show()
    return data


def additive_model(data):
    size = len(data)
    stochastic_component = normal_d(size, 0, 1)
    deterministic_component = data
    experimental_data = stochastic_component + deterministic_component
    plt.plot(experimental_data, label='Адитивна модель', alpha=0.5)
    plt.plot(stochastic_component, label='Стохастична складова', alpha=0.5)
    plt.plot(deterministic_component, label='Невипадкова складова', alpha=0.5)
    plt.ylabel('Значення')
    plt.title('Адитивна модель експериментальних даних')
    plt.legend()
    plt.show()
    return experimental_data


# Dispersion
def variance(data):
    return np.var(data)


# Mathematical expectation
def mean(data):
    return np.mean(data)


# Deviation
def deviation(data):
    return (data - np.mean(data))


# Mean square deviation
def mean_squared_deviations(data):
    return np.sqrt(np.mean(deviation(data) ** 2))


def analysis(data, label="", xlabel="", ylabel="", title=""):
    histogram(data, label=label, xlabel=xlabel, ylabel=ylabel, title=title)


if __name__ == '__main__':
    size = 10000
    lambda_ = 2

    normal_distribution_model = normal_d(size, lambda_)

    """Random variable generation model"""
    histogram(
        data=normal_distribution_model,
        bins=100,
        label='Закон зміни похибки - нормальний',
        xlabel='Значення випадкової величини',
        ylabel='Ймовірність'
    )

    """Model of change of the researched process"""
    plot(
        data=normal_distribution_model,
        label='Модель зміни досліджуваного процесу нормального розподілу',
        xlabel='Час',
        ylabel='Значення'
    )

    """Additive model"""
    linear_model = additive_model(linear_law(size))
    quadratic_model = additive_model(quadratic_law(size))

    """Analysis"""
    analysis(linear_model, 'Адитивна лінійна модель', 'Значення', 'Ймовірність', 'Адитивна модель')
    analysis(quadratic_model, 'Адитивна квадратична модель', 'Значення', 'Ймовірність', 'Адитивна модель')
