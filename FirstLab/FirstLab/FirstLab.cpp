#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

/*
Progress tracking:
1: Done
2: Done
3: Done
4а:
4б: Done
4в: Done
4г: Done
4д(only in several cases): Done
*/

using namespace std;

//
// Границы для нормального распределения
//
double leftBound = -10.0, rightBound = 10.0;
//
// Генерирование матрицы A
//
vector< vector<double> > generateMatrix(int matrixSize) {
    vector< vector<double> > matrix(matrixSize, vector<double>(matrixSize, 0.0));

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(leftBound, rightBound);

    for (int i = 0; i < matrixSize; ++i) {
        double diagonalSum = 0.0;

        for (int j = 0; j < matrixSize; ++j) {
            if (i != j) {
                matrix[i][j] = dis(gen);
                diagonalSum += abs(matrix[i][j]);
            }
        }

        matrix[i][i] = diagonalSum + 10;
    }

    return matrix;
}
//
// Генерирование вектора X
//
vector<double> generateSolution(int matrixSize) {
    vector<double> solution(matrixSize, 0.0);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(leftBound, rightBound);

    for (int i = 0; i < matrixSize; ++i) {
        solution[i] = dis(gen);
    }

    return solution;
}
//
// Вычисление вектора B
//
vector<double> generateFreeTerms(int matrixSize, vector< vector<double> > matrix, vector<double> solution) {
    vector<double> freeTerms(matrixSize, 0.0);

    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            freeTerms[i] += matrix[i][j] * solution[j];
        }
    }

    return freeTerms;
}
//
// Вывод всех матриц и векторов
//
void printAll(int matrixSize, vector< vector<double> > matrix, vector<double> solution, vector<double> freeTerms) {
    cout << "Generated matrix: " << endl;

    for (int i = 0; i < matrixSize; ++i) {
        for (int j = 0; j < matrixSize; ++j) {
            cout << fixed << setprecision(2) << matrix[i][j] << "\t";
        }
        cout << endl;
    }

    cout << "Generated X:" << endl;

    for (double currentElem : solution) {
        cout << currentElem << "\t";
    }

    cout << endl << "Generated B:" << endl;

    for (double currentElem : freeTerms) {
        cout << currentElem << "\t";
    }
    
    cout << endl;
}
//
// Вывод матрицы
// 
void printMatrix(const vector<vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (double elem : row) {
            cout << fixed << setprecision(2) << elem << "\t";
        }
        cout << endl;
    }
}
//
// Погрешность
//
void printInaccuracy(int matrixSize, vector< vector<double> > matrix, vector<double> solution, vector<double> freeTerms) {
    cout << "Inaccuracy:" << endl;

    int sum;

    for (int i = 0; i < matrixSize; ++i) {
        sum = 0;

        for (int j = 0; j < matrixSize; ++j) {
            sum += matrix[i][j] * solution[j];
        }

        cout << freeTerms[i] - sum << endl;
    }

}
//
// Транспонирование матрицы
//
vector< vector<double> > transposeMatrix(const vector< vector<double> >& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    vector< vector<double> > result(cols, vector<double>(rows, 0.0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j][i] = matrix[i][j];
        }
    }

    return result;
}
//
// Умножение матриц
//
vector< vector<double> > multiplyMatrices(const vector< vector<double> >& matrix1, const vector< vector<double> >& matrix2) {
    int rows1 = matrix1.size();
    int cols1 = matrix1[0].size();
    int cols2 = matrix2[0].size();

    vector< vector<double> > result(rows1, vector<double>(cols2, 0.0));

    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            for (int k = 0; k < cols1; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}
//
// Преобразование матрицы к симметрической
//
vector< vector<double> > transformToSymmetric(const vector< vector<double> >& matrix) {
    vector< vector<double> > transposedMatrix = transposeMatrix(matrix);
    vector< vector<double> > result = multiplyMatrices(transposedMatrix, matrix);

    int size = result.size();

    for (int i = 0; i < size; ++i) {
        double diagonalSum = 0.0;

        for (int j = 0; j < size; ++j) {
            if (i != j) {
                diagonalSum += abs(result[i][j]);
            }
        }

        result[i][i] += diagonalSum + 1.0;
    }

    return result;
}
//
// Проверка на строгое диагональное доминирование
//
bool hasStrictDiagonalDominance(const vector< vector<double> >& matrix) {
    int size = matrix.size();

    for (int i = 0; i < size; ++i) {
        double diagonalElement = abs(matrix[i][i]);
        double sum = 0.0;

        for (int j = 0; j < size; ++j) {
            if (i != j) {
                sum += abs(matrix[i][j]);
            }
        }

        if (diagonalElement <= sum) {
            return false;
        }
    }

    return true;
}
//
// Проверка матрицы на симметричность
//
bool isSymmetric(const vector< vector<double> >& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    if (rows != cols) {
        return false;
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (matrix[i][j] != matrix[j][i]) {
                return false;
            }
        }
    }

    return true;
}
//
// Нахождение обратной матрицы
//
vector<vector<double>> inverseMatrix(const vector<vector<double>>& matrix) {
    int size = matrix.size();

    vector<vector<double>> augmentedMatrix(size, vector<double>(2 * size, 0.0));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            augmentedMatrix[i][j] = matrix[i][j];
            augmentedMatrix[i][j + size] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int i = 0; i < size; ++i) {
        double pivot = augmentedMatrix[i][i];

        for (int j = 0; j < 2 * size; ++j) {
            augmentedMatrix[i][j] /= pivot;
        }

        for (int k = 0; k < size; ++k) {
            if (k != i) {
                double factor = augmentedMatrix[k][i];
                for (int j = 0; j < 2 * size; ++j) {
                    augmentedMatrix[k][j] -= factor * augmentedMatrix[i][j];
                }
            }
        }
    }

    vector<vector<double>> inverse(size, vector<double>(size, 0.0));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            inverse[i][j] = augmentedMatrix[i][j + size];
        }
    }

    return inverse;
}
//
// Функция для вычисления минора матрицы
//
vector<vector<double>> getMinor(const vector<vector<double>>& matrix, int row, int col) {
    int size = matrix.size();
    vector<vector<double>> minor(size - 1, vector<double>(size - 1, 0.0));

    for (int i = 0, p = 0; i < size; ++i) {
        if (i != row) {
            for (int j = 0, q = 0; j < size; ++j) {
                if (j != col) {
                    minor[p][q++] = matrix[i][j];
                }
            }
            ++p;
        }
    }

    return minor;
}
//
// Функция для вычисления определителя матрицы
//
double determinant(const vector<vector<double>>& matrix) {
    int size = matrix.size();

    if (size == 1) {
        return matrix[0][0];
    }

    if (size == 2) {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }

    double det = 0.0;

    for (int i = 0; i < size; ++i) {
        vector<vector<double>> minor = getMinor(matrix, 0, i);
        det += (i % 2 == 0 ? 1 : -1) * matrix[0][i] * determinant(minor);
    }

    return det;
}
//
// Функция для вычисления нормы матрицы
//
double matrixNorm(const vector<vector<double>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    double norm = 0.0;

    for (int j = 0; j < cols; ++j) {
        double colSum = 0.0;
        for (int i = 0; i < rows; ++i) {
            colSum += abs(matrix[i][j]);
        }
        norm = max(norm, colSum);
    }

    return norm;
}
//
// Функция для вычисления числа обусловленности матрицы
//
double matrixConditionNumber(const vector<vector<double>>& matrix) {
    double normA = matrixNorm(matrix);

    if (normA == 0.0) {
        cout << "Матрица вырожденная" << endl;
        return numeric_limits<double>::infinity();
    }

    vector<vector<double>> matrixInverse = inverseMatrix(matrix);

    if (matrixInverse.empty()) {
        cout << "Обратной матрицы не существует" << endl;
        return numeric_limits<double>::infinity();
    }

    double normAInversed = matrixNorm(matrixInverse);

    return normA * normAInversed;
}
//
// Функция для проверки положительной определенности матрицы
//
bool isPositiveDefinite(const vector< vector<double> >& matrix) {
    int n = matrix.size();

    for (int i = 1; i <= n; ++i) {
        vector< vector<double> > submatrix(i, vector<double>(i, 0.0));

        for (int row = 0; row < i; ++row) {
            for (int col = 0; col < i; ++col) {
                submatrix[row][col] = matrix[row][col];
            }
        }

        double det = determinant(submatrix);

        if (det <= 0) {
            return false;
        }
    }

    return true;
}
//
// Прямой ход метода прямого хода
//
vector<double> forwardSqrt(const vector<vector<double>>& L, const vector<double>& b) {
    int n = L.size();
    vector<double> y(n, 0.0);

    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }

    return y;
}
//
// Обратный ход метода обратного хода
//
vector<double> backwardSqrt(const vector<vector<double>>& LT, const vector<double>& y) {
    int n = LT.size();
    vector<double> x(n, 0.0);

    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += LT[j][i] * x[j];
        }
        x[i] = (y[i] - sum) / LT[i][i];
    }

    return x;
}
//
// Метод квадратного корня (Cholesky decomposition) для решения системы Ax = b
//
vector<double> choleskySolve(const vector<vector<double>>& A, const vector<double>& b) {
    int n = A.size();
    vector<vector<double>> L(n, vector<double>(n, 0.0));

    // Шаг 1: Разложение Cholesky
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            if (i == j) {
                for (int k = 0; k < j; ++k) {
                    sum += pow(L[j][k], 2);
                }
                L[j][j] = sqrt(A[j][j] - sum);
            }
            else {
                for (int k = 0; k < j; ++k) {
                    sum += L[i][k] * L[j][k];
                }
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }

    // Шаг 2: Решение системы Ly = b методом прямого хода
    vector<double> y = forwardSqrt(L, b);

    // Шаг 3: Решение системы L^T x = y методом обратного хода
    vector<vector<double>> LT = transposeMatrix(L);
    vector<double> x = backwardSqrt(LT, y);

    return x;
}


int main() {
    int matrixSize = 3;

    vector< vector<double> > A = generateMatrix(matrixSize);
    vector< vector<double> > AInversed = inverseMatrix(A);

    /*cout << "Generated matrix: " << endl;
    printMatrix(A);

    cout << "Inversed matrix: " << endl;
    printMatrix(AInversed);

    cout << "Checking: " << endl;
    printMatrix(multiplyMatrices(A, AInversed));

    cout << "Determinant: " << endl;
    cout << determinant(A) << endl;

    cout << "Matrix condition number:" << endl;
    cout << matrixConditionNumber(A) << endl;*/



    vector<double> x = generateSolution(matrixSize);
    vector<double> b = generateFreeTerms(matrixSize, A, x);

    printAll(matrixSize, A, x, b);
    cout << endl << endl << endl << endl << endl;
    
    if (isSymmetric(A) && isPositiveDefinite(A)) {
        vector<double> newX = choleskySolve(A, b);
        printAll(matrixSize, A, newX, b);
    }
    else {
        cout << "Wrong matrix" << endl;
    }

    /*printAll(matrixSize, A, x, b);
    printInaccuracy(matrixSize, A, x, b);

    cout << "DiagDominance: " << hasStrictDiagonalDominance(A) << endl;
    cout << "Symmetric: " << isSymmetric(A) << endl;

    A = transformToSymmetric(A);

    cout << "DiagDominance: " << hasStrictDiagonalDominance(A) << endl;
    cout << "Symmetric: " << isSymmetric(A) << endl;*/
    
    return 0;
}
