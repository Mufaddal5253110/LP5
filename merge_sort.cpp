/*
5.	Write a program to implement Parallel Merge sort using OpenMP.
Use existing algorithms and measure the performance of sequential and parallel algorithms.
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

void merge(std::vector<int>& arr, int left, int middle, int right) {
    int n1 = middle - left + 1;
    int n2 = right - middle;

    std::vector<int> leftArr(n1);
    std::vector<int> rightArr(n2);

    for (int i = 0; i < n1; ++i) {
        leftArr[i] = arr[left + i];
    }

    for (int j = 0; j < n2; ++j) {
        rightArr[j] = arr[middle + 1 + j];
    }

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k] = leftArr[i];
            ++i;
        } else {
            arr[k] = rightArr[j];
            ++j;
        }
        ++k;
    }

    while (i < n1) {
        arr[k] = leftArr[i];
        ++i;
        ++k;
    }

    while (j < n2) {
        arr[k] = rightArr[j];
        ++j;
        ++k;
    }
}

void sequentialMergeSort(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int middle = left + (right - left) / 2;

        sequentialMergeSort(arr, left, middle);
        sequentialMergeSort(arr, middle + 1, right);

        merge(arr, left, middle, right);
    }
}

void parallelMergeSort(std::vector<int>& arr, int left, int right) {
    if (left < right) {
        int middle = left + (right - left) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                parallelMergeSort(arr, left, middle);
            }

            #pragma omp section
            {
                parallelMergeSort(arr, middle + 1, right);
            }
        }

        merge(arr, left, middle, right);
    }
}

int main() {
    int size = 100000;  // Size of the array
    std::vector<int> arr(size);

    // Initialize the array with random values
    for (int i = 0; i < size; ++i) {
        arr[i] = rand() % 1000;
    }

    // Measure the execution time of the sequential version
    auto startSeq = std::chrono::high_resolution_clock::now();
    sequentialMergeSort(arr, 0, size - 1);
    auto endSeq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> seqDuration = endSeq - startSeq;

    // Measure the execution time of the parallel version
    auto startPar = std::chrono::high_resolution_clock::now();
    parallelMergeSort(arr, 0, size - 1);
    auto endPar = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> parDuration = endPar - startPar;

    // Output the execution times
    std::cout << "Sequential Merge Sort Time: " << seqDuration.count() << " seconds" << std::endl;
    std::cout << "Parallel Bubble Sort Time: " << parDuration.count() << " seconds" << std::endl;
    
    return 0;
}