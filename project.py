from IPython import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Вариант 16

data = pd.read_excel("LAB_DATA_MS (1).xlsx", sheet_name="Котировки")[
    15
]  # Считывание данных

df = pd.DataFrame(data)
df

# Интервальный статистический ряд
data_new = []
for i in range(2, len(data)):
    data_new.append((data[i] - data[i - 1]) / data[i - 1])
s = pd.Series(data_new, name="Относительные приращения котировок", dtype="float64")
df = pd.DataFrame(s)
df

# Количество интервалов по формуле Стерджесса

m = 1 + int(np.log2(len(data_new)))
print("Количество интервалов:", m)
mx = max(data_new)
mn = min(data_new)
width = (mx - mn) / m
print("Ширина интервала:", width)

intervals = []  # массив значений интервалов
average = []
cur = mn
for _ in range(m):
    intervals.append([cur, cur + width])
    average.append((cur + cur + width) / 2)
    cur += width

display(pd.DataFrame(intervals, columns=["Левая граница", "Правая граница"]))
display(pd.DataFrame(average, columns=["Среднее значение интервала"]))

data_in_intervals = [[] for i in range(m)]
freq = []
for i in data_new:
    for num, l in enumerate(intervals, 0):
        if l[0] <= i < l[1]:
            data_in_intervals[num].append(i)
for i in range(m):
    freq.append(len(data_in_intervals[i]))

display(pd.DataFrame(freq, columns=["Частота"]))

freq = np.array(freq)
rel_freq = freq / len(data_new)
density_freq = rel_freq / width
display(
    pd.DataFrame(
        np.column_stack((rel_freq, density_freq)),
        columns=["Относительная частота", "Плотность частоты"],
    )
)

# Итоговый интервальный статистический ряд
# Интервальный статистический ряд является оценкой закона  распределения случайной величины, так как с помощью его можно оценить вероятности попадания в заданные интервалы

display(
    pd.DataFrame(
        np.column_stack((intervals, average, freq, rel_freq, density_freq)),
        index=range(1, 9),
        columns=[
            "Левая граница",
            "Правая граница",
            "Среднее значение интервала",
            "Частота",
            "Относительная частота",
            "Плотность частоты",
        ],
    )
)

# Полигон частот является оценкой многоугольника распределения, для непрерывной случайной величины полигон частот есть оценка кривой плотности распределения

plt.plot(
    average,
    density_freq,
    color="black",
    marker="o",
    markeredgecolor="red",
    markersize=10,
    markeredgewidth=4,
)
plt.title("Полигон частот")
plt.xlabel("x")
plt.ylabel("density", rotation=0)
plt.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
plt.show()

# Гистограмма частот - статистический аналог кривой плотности распределения

plt.hist(data_new, bins=8, edgecolor="black", color="#B1B176")
plt.title("Гистограмма частот", fontsize=14)
plt.show()

# Выборочное среднее и дисперсия
import scipy.stats

X_aver = sum(data_new) / len(data_new)
D = sum((np.array(data_new) - X_aver) ** 2) / len(data_new)

X = np.linspace(mn, mx, 1000)
Y = scipy.stats.norm.pdf(X, X_aver, np.sqrt(D))
plt.plot(X, Y)
plt.title(f"Плотность нормального распределения N({X_aver:.4f},{D:.5f})", fontsize=10)
plt.xlabel("x", labelpad=10)
plt.ylabel("y", rotation=0, labelpad=20)
plt.show()

print(
    "Вывод: Видим, что график плотности распределения частот имеет визуальные отличия от графика плотности нормального распределения. Но, в целом, сохраняет общие тренды."
)
