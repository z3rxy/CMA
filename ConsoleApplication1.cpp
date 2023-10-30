#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

/*
Progress tracking:
1: Done
2: Done
3: Done
4a:
4b:
4c:
4d:
4e(only in several cases):
*/

using namespace std;

//
// Generating matrix A
//
vector< vector<double> > generateMatrix(int matrixSize) {
    vector< vector<double> > matrix(matrixSize, vector<double>(matrixSize, 0.0));

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-100.0, 100.0);

    for (int i = 0; i < matrixSize; ++i) {
        double diagonalSum = 0.0;

        for (int j = 0; j < matrixSize; ++j) {
            if (i != j) {
                matrix[i][j] = dis(gen);
                diagonalSum += abs(matrix[i][j]);
            }
        }

        matrix[i][i] = diagonalSum + dis(gen);
    }

    return matrix;
}
//
// Generating solution X
//
vector<double> generateSolution(int matrixSize) {
    vector<double> solution(matrixSize, 0.0);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-100.0, 100.0); 

    for (int i = 0; i < matrixSize; ++i) {
        solution[i] = dis(gen);
    }

    return solution;
}
//
// Generating vector B
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
// Printing all vectors
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
}

int main() {
    int matrixSize = 4;

    vector< vector<double> > A = generateMatrix(matrixSize);
    vector<double> x = generateSolution(matrixSize);
    vector<double> b = generateFreeTerms(matrixSize, A, x);

    printAll(matrixSize, A, x, b);

    return 0;
}
