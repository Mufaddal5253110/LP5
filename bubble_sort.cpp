/*
Write a program to implement Parallel Bubble Sort using OpenMP.
Use existing algorithms and measure the performance of sequential and parallel algorithms. in c++
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

void parallelBubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    bool swapped;

    #pragma omp parallel
    {
        do {
            swapped = false;

            #pragma omp for
            for (int i = 0; i < n - 1; ++i) {
                if (arr[i] > arr[i + 1]) {
                    std::swap(arr[i], arr[i + 1]);
                    swapped = true;
                }
            }
        } while (swapped);
    }
}

void sequentialBubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    bool swapped;

    do {
        swapped = false;

        for (int i = 0; i < n - 1; ++i) {
            if (arr[i] > arr[i + 1]) {
                std::swap(arr[i], arr[i + 1]);
                swapped = true;
            }
        }
    } while (swapped);
}

int main() {
    int size = 10000;  // Size of the array
    std::vector<int> arr(size);

    // Initialize the array with random values
    for (int i = 0; i < size; ++i) {
        arr[i] = rand() % 1000;
    }

    // Measure the execution time of the sequential version
    auto startSeq = std::chrono::high_resolution_clock::now();
    sequentialBubbleSort(arr);
    auto endSeq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> seqDuration = endSeq - startSeq;

    // Measure the execution time of the parallel version
    auto startPar = std::chrono::high_resolution_clock::now();
    parallelBubbleSort(arr);
    auto endPar = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> parDuration = endPar - startPar;

    // Output the execution times
    std::cout << "Sequential Bubble Sort Time: " << seqDuration.count() << " seconds" << std::endl;
    std::cout << "Parallel Bubble Sort Time: " << parDuration.count() << " seconds" << std::endl;

    return 0;
}

