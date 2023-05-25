/*Write a program to implement Parallel matrix matrix multiplication using OpenMp.*/

#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

std::vector<std::vector<int>> parallelMatrixMultiplication(const std::vector<std::vector<int>>& matrix1, const std::vector<std::vector<int>>& matrix2) {
    int numRows1 = matrix1.size();
    int numCols1 = matrix1[0].size();
    int numRows2 = matrix2.size();
    int numCols2 = matrix2[0].size();

    std::vector<std::vector<int>> result(numRows1, std::vector<int>(numCols2, 0));

    #pragma omp parallel for
    for (int i = 0; i < numRows1; ++i) {
        for (int j = 0; j < numCols2; ++j) {
            for (int k = 0; k < numCols1; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}

int main() {
    int numRows1 = 3;  // Number of rows in matrix 1
    int numCols1 = 3;  // Number of columns in matrix 1
    int numRows2 = 3;  // Number of rows in matrix 2
    int numCols2 = 3;  // Number of columns in matrix 2

    std::vector<std::vector<int>> matrix1 = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    std::vector<std::vector<int>> matrix2 = {
        {9, 8, 7},
        {6, 5, 4},
        {3, 2, 1}
    };

    // Measure the execution time of the parallel matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> result = parallelMatrixMultiplication(matrix1, matrix2);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Output the execution time
    std::cout << "Parallel Matrix Multiplication Time: " << duration.count() << " seconds" << std::endl;

    // Output the result matrix
    std::cout << "Result Matrix:" << std::endl;
    for (int i = 0; i < numRows1; ++i) {
        for (int j = 0; j < numCols2; ++j) {
            std::cout << result[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
