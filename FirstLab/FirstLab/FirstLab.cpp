#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

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

        matrix[i][i] = diagonalSum + 1;
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
// Погрешность(вектор невязки)
//
void printInaccuracy(int matrixSize, vector< vector<double> > matrix, vector<double> solution, vector<double> freeTerms) {
    cout << "Inaccuracy:" << endl;

    double sum;

    for (int i = 0; i < matrixSize; ++i) {
        sum = 0.00000000000000;

        for (int j = 0; j < matrixSize; ++j) {
            sum += matrix[i][j] * solution[j];
        }

        cout << fixed << setprecision(15) << freeTerms[i] - sum << endl;
    }
    cout << endl << "---------------------------------------------" << endl;
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
// Умножение матрицы на вектор
//
vector<double> multiplyMatrixVector(const vector<vector<double>>& A, const vector<double>& b) {
    int n = A.size();
    vector<double> result(n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i] += A[i][j] * b[j];
        }
    }

    return result;
}
//
// Вычитаение векторов
//
vector<double> subtractVectors(const vector<double>& x, const vector<double>& y) {
    int n = x.size();
    vector<double> result(n);

    for (int i = 0; i < n; ++i) {
        result[i] = x[i] - y[i];
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
// Функция для вычисления нормы вектора
// 
double vectorNorm(const vector<double>& v) {
    int n = v.size();
    double sum = 0.0;

    for (int i = 0; i < n; ++i) {
        sum += v[i] * v[i];
    }

    return sqrt(sum);
}
//
// Скалярное произведение векторов
// 
double dotProduct(const vector<double>& v1, const vector<double>& v2) {
    int n = v1.size();
    double result = 0.0;

    for (int i = 0; i < n; ++i) {
        result += v1[i] * v2[i];
    }

    return result;
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
// Методом вращений
//
vector<double> rotations(vector<vector<double>>& A, vector<double>& b) {
    int size = A.size();
    vector<vector<double>> A_rotations = A;
    vector<double> b_rotations = b;

    if (!isSymmetric(A)) {
        cout << "Matrix is not symmetric. Applying Symmetric transform" << endl;

        vector<vector<double>> AT = transposeMatrix(A);
        A = multiplyMatrices(AT, A);

        vector<double> newB(size, 0.0);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                newB[i] += AT[i][j] * b[j];
            }
        }
        b = newB;
    }

    for (int j = 0; j < size - 1; j++) {
        for (int i = j + 1; i < size; i++) {
            double c = A_rotations[j][j], s = A_rotations[i][j];
            double sqr = sqrt(pow(A_rotations[j][j], 2) + pow(A_rotations[i][j], 2));

            for (int k = j; k < size; k++) {
                double temp = A_rotations[j][k];
                A_rotations[j][k] = (c * A_rotations[j][k] + s * A_rotations[i][k]) / sqr;
                A_rotations[i][k] = (-s * temp + c * A_rotations[i][k]) / sqr;
            }

            double temp = b_rotations[j];
            b_rotations[j] = (c * b_rotations[j] + s * b_rotations[i]) / sqr;
            b_rotations[i] = (-s * temp + c * b_rotations[i]) / sqr;
        }
    }

    for (int i = size - 1; i > 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
            b_rotations[j] -= (b_rotations[i] * A_rotations[j][i] / A_rotations[i][i]);
            A_rotations[j][i] = 0;
        }
        b_rotations[i] /= A_rotations[i][i];
        A_rotations[i][i] = 1;
    }

    b_rotations[0] /= A_rotations[0][0];
    A_rotations[0][0] = 1;

    return b_rotations;
}
//
// Метод квадратного корня
//
vector<double> methodSqrt(vector<vector<double>>& A,vector<double>& b) {
    int n = A.size();

    if (!isSymmetric(A)) {
        cout << "Matrix is not symmetric. Applying Symmetric transform" << endl;
        //A = transformToSymmetric(A);

        vector<vector<double>> AT = transposeMatrix(A);
        A = multiplyMatrices(AT, A);

        vector<double> newB(n, 0.0);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                newB[i] += AT[i][j] * b[j];
            }
        }

        b = newB;
    }

    if (!isPositiveDefinite(A)) {
        cout << "Matrix is not positive defined." << endl;
        vector<double> answer(n, 0.0);
        return answer;
    }

    vector<vector<double>> L(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;

            if (j == i) {
                for (int k = 0; k < j; k++) {
                    sum += pow(L[j][k], 2);
                }
                L[j][j] = sqrt(A[j][j] - sum);
            }
            else {
                for (int k = 0; k < j; k++) {
                    sum += L[i][k] * L[j][k];
                }
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }

    vector<double> y(n, 0.0);
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }

    vector<double> x(n, 0.0);
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += L[j][i] * x[j];
        }
        x[i] = (y[i] - sum) / L[i][i];
    }

    return x;
}
//
// Метод простой итерации
//
vector<double> simpleIteration(vector<vector<double>>& A,vector<double>& b, int maxIterations, double tolerance) {
    int n = A.size();

    vector<double> x(n, 0.0);

    for (int iter = 0; iter < maxIterations; ++iter) {
        vector<double> x_new(n, 0.0);

        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                if (j != i) {
                    sum += A[i][j] * x[j];
                }
            }
            x_new[i] = (b[i] - sum) / A[i][i];
        }

        double error = 0.0;
        for (int i = 0; i < n; ++i) {
            error += abs(x_new[i] - x[i]);
        }

        x = x_new;

        if (error < tolerance) {
            cout << "Converged in " << iter + 1 << " iterations" << endl;
            return x;
        }
    }

    cout << "Did not converge within the specified number of iterations" << endl;
    return x;
}
//
// Метод минимальный невязок
//
vector<double> minResidual(vector<vector<double>>& A,vector<double>& b, int maxIterations, double tolerance) {
    int n = A.size();

    vector<double> x(n, 0.0);
    vector<double> r = b;

    for (int iter = 0; iter < maxIterations; ++iter) {
        vector<double> Ax = multiplyMatrixVector(A, x);
        vector<double> residual = subtractVectors(b, Ax);

        double residualNorm = vectorNorm(residual);
        if (residualNorm < tolerance) {
            cout << "Converged in " << iter + 1 << " iterations." << endl;

            return x;
        }

        vector<double> Ar = multiplyMatrixVector(A, residual);
        double alpha = dotProduct(residual, residual) / dotProduct(residual, Ar);

        for (int i = 0; i < n; ++i) {
            x[i] += alpha * residual[i];
        }

        for (int i = 0; i < n; ++i) {
            r[i] = residual[i] - alpha * Ar[i];
        }
    }

    cout << "Did not converge within the specified number of iterations." << endl;

    return x;
}

int main() {
    int matrixSize = 3;
    int maxIterations = 100000;
    double tolerance = 1e-15;

    /*vector< vector<double> > A = generateMatrix(matrixSize);
    vector<double> x = generateSolution(matrixSize);
    vector<double> b = generateFreeTerms(matrixSize, A, x);*/
    vector<vector<double>> A = {
        {15.61, -9.53, 5.08},
        {0.97, 2.91, -0.94},
        {6.37, 8.69, 16.05}
    };

    vector<double> x = { 9.52, 2.27, -8.20 };

    vector<double> b = { 85.32, 23.49, -51.29 };

    /*vector< vector<double> > AInversed = inverseMatrix(A);

    cout << "Generated matrix: " << endl;
    printMatrix(A);

    cout << "Inversed matrix: " << endl;
    printMatrix(AInversed);

    cout << "Checking: " << endl;
    printMatrix(multiplyMatrices(A, AInversed));

    cout << "Determinant: " << endl;
    cout << determinant(A) << endl;

    cout << "Matrix condition number:" << endl;
    cout << matrixConditionNumber(A) << endl;*/

    printAll(matrixSize, A, x, b);
    printInaccuracy(matrixSize, A, x, b);
    cout << endl << endl << endl;

    cout << "METHOD OF SIMPLE ITERATION" << endl;
    vector<double> xSimpleIter = simpleIteration(A, b, maxIterations, tolerance);
    printAll(matrixSize, A, xSimpleIter, b);
    printInaccuracy(matrixSize, A, xSimpleIter, b);

    cout << "METHOD OF MINIMUM RESIDUALS" << endl;
    vector<double> xMinRes = minResidual(A, b, maxIterations, tolerance);
    printAll(matrixSize, A, xMinRes, b);
    printInaccuracy(matrixSize, A, xMinRes, b);

    cout << "METHOD OF ROTATIONS" << endl;
    vector<double> xRotations = rotations(A, b);
    printAll(matrixSize, A, xRotations, b);
    printInaccuracy(matrixSize, A, xRotations, b);

    cout << "METHOD OF SQRT" << endl;
    vector<double> xSqrt = methodSqrt(A, b);
    printAll(matrixSize, A, xSqrt, b);
    printInaccuracy(matrixSize, A, xSqrt, b);


    // Unknown result
   /* cout << "METHOD OF SIMPLE ITERATION" << endl;
    xSimpleIter = simpleIteration(A, b, maxIterations, tolerance);
    printAll(matrixSize, A, xSimpleIter, b);
    printInaccuracy(matrixSize, A, xSimpleIter, b);*/

    return 0;
}
