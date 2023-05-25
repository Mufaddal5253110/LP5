#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j <= high - 1; ++j) {
        if (arr[j] < pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }

    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

void sequentialQuickSort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pivot = partition(arr, low, high);

        sequentialQuickSort(arr, low, pivot - 1);
        sequentialQuickSort(arr, pivot + 1, high);
    }
}

void parallelQuickSort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pivot = partition(arr, low, high);

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                parallelQuickSort(arr, low, pivot - 1);
            }

            #pragma omp section
            {
                parallelQuickSort(arr, pivot + 1, high);
            }
        }
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
    sequentialQuickSort(arr, 0, size - 1);
    auto endSeq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> seqDuration = endSeq - startSeq;

    // Measure the execution time of the parallel version
    auto startPar = std::chrono::high_resolution_clock::now();
    parallelQuickSort(arr, 0, size - 1);
    auto endPar = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> parDuration = endPar - startPar;

    // Output the execution times
    std::cout << "Sequential Quick Sort Time: " << seqDuration.count() << " seconds" << std::endl;
    std::cout << "Parallel Quick Sort Time: " << parDuration.count() << " seconds" << std::endl;

    return 0;
}  
