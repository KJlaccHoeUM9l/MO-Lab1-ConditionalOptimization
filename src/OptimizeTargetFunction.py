import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt
#from sympy import as sp
import sympy as sp

leftBoundary = -3.5
rightBoundary = 3.5
density = 500

x1, x2 = np.meshgrid(np.linspace(leftBoundary, rightBoundary, density),
                     np.linspace(leftBoundary, rightBoundary, density))

# Инициализация заданных функций
f = 3 * (x1**2 + x2)**2 + (x1**2 - 1)**2
g1 = 0.5 * x1 + x2 + 0.5
g2 = -10 * (x1 + 1)**2 - (x2 - 2)**2 + 12
g3 = x2 + 0.8

# Отрисовка допустимого множества и линий уровня целевой функции
plt.figure("Допустимое множество")
plt.pcolormesh(x1, x2, (g1 <= 0) & (g2 <= 0) & (g3 <= 0), alpha=0.1) # Окрашивание допустимого множества D (желтый цвет)

contours = plt.contour(x1, x2, f, 28)                # Линии уровня целевой функции (последний аргумент - количество линий)
plt.clabel(contours, inline=True, fontsize=8)        # Значение уровня на линиях (последний аргумент - размер шрифта)
contours = plt.contour(x1, x2, g1, (0,), colors='g') # Ограничение g1 (предпоследний аргумент - активность ограничения, т.е. g1(x) = 0)
contours = plt.contour(x1, x2, g2, (0,), colors='r')
contours = plt.contour(x1, x2, g3, (0,), colors='y')
plt.show("Допустимое множество")

# Численная оптимизация целевой функции
x1_ = 2.4
x2_ = -3.4
initialGuess = np.array([x1_, x2_]) # Начальное приближение для численного метода оптимизации


# Активные ограничения (constraint)
def constraintG2(x):
    return -10 * (x[0] + 1)**2 - (x[1] - 2)**2 + 12


cons = {'type':'eq', 'fun': constraintG2}


# Целевая функция
def targetFunction(x):
    return 3 * (x[0]**2 + x[1])**2 + (x[0]**2 - 1)**2


print(opt.minimize(targetFunction, initialGuess, constraints=cons))
#print("Optimum: ", initialGuess)

# Условия Каруша-Куна-Таккера


def f(x):
    return 3 * (x[0]**2 + x[1])**2 + (x[0]**2 - 1)**2


def g1(x):
    return 0.5 * x[0] + x[1] + 0.5


def g2(x):
    return -10 * (x[0] + 1)**2 - (x[1] - 2)**2 + 12


def g3(x):
    return x[1] + 0.8


x = sp.symbols('x1 x2')
lambda_ = sp.symbols('lambda')

df_dx1 = sp.diff(f(x), x[0])
df_dx2 = sp.diff(f(x), x[1])

dg1_dx1 = sp.diff(g1(x), x[0])
dg1_dx2 = sp.diff(g1(x), x[1])

dL_dx1 = sp.Add(df_dx1, sp.Mul(lambda_, dg1_dx1))
dL_dx2 = sp.Add(df_dx2, sp.Mul(lambda_, dg1_dx2))

print("Karush - Kuhn - Tucker")
print ("dL_dx1: ", dL_dx1)
print("dL_dx2: ",dL_dx2)
print("g1(x): ", g1(x))

# print(sp.solve([dL_dx1, dL_dx2, g1(x)], dict=True))
# d=1.4
# sp.plot3d(f(x),(x[0],-d,d),(x[1],-d,d))
