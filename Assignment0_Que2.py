import numpy as np

def arr_dot(array1, array2):
    if array1.shape != array2.shape:
        print("The array must have same shape")
        return
    else :
        return np.sum(array1 * array2)
        

def arr_det(matrix):
    for i in range(len(matrix)):
        if len(matrix) != len(matrix[i]):
            print("The Matrix Must be square matrix")
            return
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
    det = 0
    for ele in range(len(matrix)):
        minor = [matrix[i][:ele]+matrix[i][(ele+1):] for i in range(1,len(matrix))]
        det = det + ((-1)**ele)*matrix[0][ele]*arr_det(minor)
    return det


def arr_mul(matrix1, matrix2):
    for i in matrix1:
        if len(i) != len(matrix2):
            print("Give a valid input")
            return
    mult = [[0 for i in range(len(matrix2[0]))] for i in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                mult[i][j]=mult[i][j]+matrix1[i][k]*matrix2[k][j]
    return mult

def matrix_operation(array1,array2,operation):
    if operation=="dot":
        return arr_dot(array1,array2)
    elif operation=="matrix":
        return arr_mul(array1,array2)
    elif operation=="determinant":
        return (arr_det(array1),arr_det(array2))


