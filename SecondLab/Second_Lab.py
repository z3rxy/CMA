import random
import numpy as np

def generate_matrix(n):
    matrix = [[random.randint(-100, 100) for j in range(n)] for i in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            matrix[j][i] = matrix[i][j]
    
    return matrix

def is_symmetric(matrix):
    if not all(len(row) == len(matrix) for row in matrix):
        return False

    n = len(matrix)
    for i in range(n):
        for j in range(i+1, n):
            if matrix[i][j] != matrix[j][i]:
                return False
    
    return True

def method_rotations(A, tol=1e-2, max_iter=1000):
    n = len(A)
    V = np.identity(n)

    for _ in range(max_iter):
        max_off_diag = 0.0
        p, q = 0, 0
        for i in range(n):
            for j in range(i+1, n):
                if abs(A[i, j]) > max_off_diag:
                    max_off_diag = abs(A[i, j])
                    p, q = i, j

        if max_off_diag < tol:
            break

        if A[p, p] == A[q, q]:
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan(2 * A[p, q] / (A[p, p] - A[q, q]))

        U = np.identity(n)
        U[p, p] = U[q, q] = np.cos(theta)
        U[p, q] = -np.sin(theta)
        U[q, p] = np.sin(theta)

        A = np.dot(np.dot(U.T, A), U)
        V = np.dot(V, U)

    eigenvalues = np.diag(A)
    eigenvectors = V
    return eigenvalues, eigenvectors

def compare_eigenvalues(matrix, tol=1e-9, max_iter=1000):
    eigenvalues_jacobi, eigenvectors_jacobi = method_rotations(np.array(matrix), tol, max_iter)

    eigenvalues_numpy, eigenvectors_numpy = np.linalg.eig(np.array(matrix))

    eigenvalues_rotations_sorted = np.sort(eigenvalues_jacobi)
    eigenvectors_jacobi_sorted = eigenvectors_jacobi[:, eigenvalues_jacobi.argsort()]

    eigenvalues_numpy_sorted = np.sort(eigenvalues_numpy)
    eigenvectors_numpy_sorted = eigenvectors_numpy[:, eigenvalues_numpy.argsort()]

    for i in range(len(eigenvalues_jacobi)):
        print(f"lambda_{i + 1}:")
        print(f"Метод вращений: {eigenvalues_rotations_sorted[i]}")
        print(f"NumPy: {eigenvalues_numpy_sorted[i]}")
        print("Отклонение:", eigenvalues_numpy_sorted[i] - eigenvalues_rotations_sorted[i])
        print("")

        print(f"Собственный к lambda_{i + 1}:")
        print(f"Метод вращений:")
        print(eigenvectors_jacobi_sorted[:, i])
        print("")
        print(f"NumPy:")
        print(eigenvectors_numpy_sorted[:, i])
        print("")
        print("")

def power_method(matrix, tol=1e-6, max_iter=1000):
    n = len(matrix)
    
    v = np.random.rand(n)
    v /= np.linalg.norm(v)

    for _ in range(max_iter):
        Av = np.dot(matrix, v)
        
        max_element_index = np.argmax(np.abs(Av))
        max_element = Av[max_element_index]
        
        v = Av / max_element
        
        if np.abs(max_element) < tol:
            break

    v /= np.linalg.norm(v)

    eigenvalue = np.dot(v, np.dot(matrix, v))

    return eigenvalue, v

#n = int(input("Введите значение n (n >= 10): "))
n = 10

matrix = generate_matrix(n)


print("Сгенерированная матрица:")

for row in matrix:
    print(row)

print('Is matrix symmetric:',is_symmetric(matrix))

compare_eigenvalues(matrix)
print(power_method(matrix))