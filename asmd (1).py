import numpy as np

print('Введите количество переменных функции ')
arguments_number = int(input())  # количество переменных ==m

print('Введите сколько точек известно')
points_number = int(input())  # количество известных точек ==n

one_row = np.ones(points_number).transpose()
args = np.zeros((points_number, arguments_number))
values = np.zeros(points_number)
quested = np.zeros(arguments_number)

print('Введите значения аргументов в известных точках')
for i in range(points_number):
    args[i] = [float(n) for n in input().split()]

print('Введите значения функции в известных точках')
for i in range(points_number):
    values[i] = float(input())
values = values.transpose()

print('Введите значения аргументов в искомой точке')
quested = [float(n) for n in input().split()]

W_waved = np.zeros((points_number, points_number))
W_waved_out_l = np.zeros((points_number, points_number))
weights = np.zeros(arguments_number)
weights_waved = np.zeros(arguments_number)


for l in range(arguments_number):  # поиск весов
    for i in range(points_number):
        for j in range(points_number):
            for k in range(arguments_number):
                W_waved[i][j] += (args[i][k] - quested[k]) * (args[j][k] - quested[k])
                if k != l:
                    W_waved_out_l[i][j] += (args[i][k] - quested[k]) * (args[j][k] - quested[k])

Y_star_waved = np.dot(np.dot(np.linalg.inv(W_waved) * one_row), values) \
               / np.dot(np.dot(np.linalg.inv(W_waved) * one_row), one_row)
Y_star_out_l = np.dot(np.dot(np.linalg.inv(W_waved_out_l) * one_row), values) \
               / np.dot(np.dot(np.linalg.inv(W_waved_out_l) * one_row), one_row)

for l in range(arguments_number):
    weights_waved[l] = (Y_star_waved - Y_star_out_l) ** 2

sum_of_waved_weights = 0
for i in range(arguments_number):
    sum_of_waved_weights += weights_waved[i]

for l in range(arguments_number):
    weights[l] = arguments_number * weights_waved[l] / sum_of_waved_weights

W = np.zeros(points_number, points_number)

for i in range(points_number):
    for j in range(points_number):
        for k in range(arguments_number):
            W[i][j] += weights[k] * (args[i][k] - quested[k]) * (args[j][k] - quested[k])

ans = np.dot(np.dot(np.linalg.inv(W), values), one_row) / np.dot(np.dot(np.linalg.inv(W), one_row), one_row)

print('Предполагаемое значение:', ans)
