/*
Write a program to implement Parallel matrix vector multiplication using OpenMp. 
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

std::vector<int> parallelMatrixVectorMultiplication(const std::vector<std::vector<int>>& matrix, const std::vector<int>& vector) {
    int numRows = matrix.size();
    int numCols = matrix[0].size();
    int vectorSize = vector.size();

    std::vector<int> result(numRows, 0);

    #pragma omp parallel for
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    return result;
}

int main() {
    int numRows = 3;  // Number of rows in the matrix
    int numCols = 3;  // Number of columns in the matrix
    int vectorSize = 3;  // Size of the vector

    std::vector<std::vector<int>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    std::vector<int> vector = {1, 2, 3};

    // Measure the execution time of the parallel matrix-vector multiplication
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> result = parallelMatrixVectorMultiplication(matrix, vector);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Output the execution time
    std::cout << "Parallel Matrix-Vector Multiplication Time: " << duration.count() << " seconds" << std::endl;

    // Output the result vector
    std::cout << "Result Vector:" << std::endl;
    for (int i = 0; i < numRows; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
