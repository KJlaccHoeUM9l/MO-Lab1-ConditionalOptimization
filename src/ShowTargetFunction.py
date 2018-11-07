from sympy import *
from sympy.plotting import plot3d

# Инициализация заданных функций
x1, x2 = symbols('x1 x2')
f = 3 * (x1**2 + x2)**2 + (x1**2 - 1)**2
g1 = 0.5 * x1 + x2 + 0.5
g2 = -10 * (x1 + 1)**2 - (x2 - 2)**2 + 12
g3 = x2 + 0.8

# Отрисовка графика функции
leftBoundary = -3.5
rightBoundary = 3.5
plot3d(f, (x1, leftBoundary, rightBoundary), (x2, leftBoundary, rightBoundary))


# Для notebook
def gradient(scalarFunction, variables):
    matrixScalarFunction = Matrix([scalarFunction])
    return matrixScalarFunction.jacobian(variables)


print(latex(gradient(f, [x1, x2])))
