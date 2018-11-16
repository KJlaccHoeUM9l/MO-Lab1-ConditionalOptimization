import numpy as np
import matplotlib.pylab as plt
import scipy.optimize as opt
import sympy as sp

lambda1 = sp.symbols('lambda1')
lambda2 = sp.symbols('lambda2')
lambda3 = sp.symbols('lambda3')

# --------------------------------------------------- ВАРИАНТ 3 --------------------------------------------------------
def f(x):
    return 3 * (x[0]**2 + x[1])**2 + (x[0]**2 - 1)**2


def g1(x):
    return 0.5 * x[0] + x[1] + 0.5


def g2(x):
    return -10 * (x[0] + 1)**2 - (x[1] - 2)**2 + 12


def g3(x):
    return x[1] + 0.8
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------- HELPER FUNCTIONS -----------------------------------------------------
def initPlot(plotName, x1, x2):
    plt.figure(plotName)
    # Окрашивание допустимого множества D (желтый цвет)
    plt.pcolormesh(x1, x2, (g1([x1, x2]) <= 0) & (g2([x1, x2]) <= 0) & (g3([x1, x2]) <= 0), alpha=0.1)

    # Линии уровня целевой функции (последний аргумент - количество линий)
    contours = plt.contour(x1, x2, f([x1, x2]), 30)

    # Значение уровня на линиях (последний аргумент - размер шрифта)
    plt.clabel(contours, inline=True, fontsize=8)

    # Ограничения(предпоследний аргумент - активность ограничения, т.е. g1(x) = 0)
    plt.contour(x1, x2, g1([x1, x2]), (0,), colors='g')
    plt.contour(x1, x2, g2([x1, x2]), (0,), colors='r')
    plt.contour(x1, x2, g3([x1, x2]), (0,), colors='y')

    return plt


def findMin(f, initialGuess, cons):
    result = opt.minimize(f, initialGuess, constraints=cons)

    print("--->Начально приближение: ", initialGuess)
    print("--->f* = ", result.fun)
    print("--->x* = ", result.x)

    return result


def KKT(L):
    x = sp.symbols('x1 x2')

    dL_dx1 = sp.diff(L, x[0])
    dL_dx2 = sp.diff(L, x[1])
    dL_dlambda1 = sp.diff(L, lambda1)
    dL_dlambda2 = sp.diff(L, lambda2)
    dL_dlambda3 = sp.diff(L, lambda3)

    print("--->L = ", L)
    print("--->dL_dx1: ", dL_dx1)
    print("--->dL_dx2: ", dL_dx2)
    print("--->dL_dlambda1 = ", dL_dlambda1)
    print("--->dL_dlambda2 = ", dL_dlambda2)
    print("--->dL_dlambda3 = ", dL_dlambda3)

    return sp.solve([dL_dx1, dL_dx2, dL_dlambda1, dL_dlambda2, dL_dlambda3], dict=True)
# ----------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------------- MAIN ---------------------------------------------------------
if __name__ == "__main__":
    leftBoundary = -3.5
    rightBoundary = 3.5
    density = 500
    x1, x2 = np.meshgrid(np.linspace(leftBoundary, rightBoundary, density),
                         np.linspace(leftBoundary, rightBoundary, density))
    x = sp.symbols('x1 x2')

    plotName = "Допустимое множество"
    initPlot(plotName, x1, x2).show(plotName)


    # НАЧАЛЬНЫЕ УСЛОВИЯ #1
    # При таком начальном приближении минимум будет находится на ограничении g2
    # но при проверке условий ККТ решений нет в принципе, а оно должно быть (ОЧЕНЬ СТРАННО)
    startApproximation = [-1.0, -2.5]
    print("Поиск минимума[1]:");
    opt1 = findMin(f, startApproximation, {'type':'eq', 'fun': g2})

    print("Проверка условий Каруша-Куна-Таккера[1]");
    kkt = KKT(f(x) + lambda2 * g2(x))
    print("--->Решение системы: ", kkt)

    plotName = "Minimum1"
    plot1 = initPlot(plotName, x1, x2)

    # Рисование линии уровня, соответствующей найденному min (предпоследний аргумент - значение уровня)
    plot1.contour(x1, x2, f([x1, x2]), (opt1.fun,), colors='b')
    plot1.contour(x1, x2, (x1 - opt1.x[0]) ** 2 + (x2 - opt1.x[1]) ** 2, (0.005,), colors='black')  # Точка min
    plot1.show(plotName)


    # НАЧАЛЬНЫЕ УСЛОВИЯ #2
    startApproximation = [0.0, -2.5]
    print("Поиск минимума[2]:");
    opt2 = findMin(f, startApproximation, ({'type': 'eq', 'fun': g2}, {'type': 'eq', 'fun': g3}))

    print("Проверка условий Каруша-Куна-Таккера");
    kkt = KKT(f(x) + lambda2 * g2(x) + lambda3 * g3(x))
    print("--->Решение системы: ", kkt[0])

    plotName = "Min2"
    plot2 = initPlot(plotName, x1, x2)

    # Рисование линии уровня, соответствующей найденному min (предпоследний аргумент - значение уровня)
    plot2.contour(x1, x2, f([x1, x2]), (opt2.fun,), colors='b')
    plot2.contour(x1, x2, (x1 - opt2.x[0]) ** 2 + (x2 - opt2.x[1]) ** 2, (0.005,), colors='black')  # Точка min
    plot2.show("Min2")
# ----------------------------------------------------------------------------------------------------------------------
