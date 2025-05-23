#include <stdio.h>
#include <iostream>
#include <chrono>
#include "matar.h"

// Required for MATAR data structures
using namespace mtr; 

#define MATRIX_SIZE 1024

// Timer class for timing the execution of the matrix multiplication
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool is_running;

public:
    Timer() : is_running(false) {}
    
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        is_running = true;
    }
    
    double stop() {
        if (!is_running) {
            std::cerr << "Timer was not running!" << std::endl;
            return 0.0;
        }
        end_time = std::chrono::high_resolution_clock::now();
        is_running = false;
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // Convert to milliseconds
    }
};

// Function to calculate theoretical FLOPS
double calculate_flops(int size, double time_ms) {
    // For matrix multiplication C = A * B:
    // Each element C(i,j) requires 2*size operations (size multiplications + size-1 additions)
    // Total operations = size * size * (2*size)
    double total_ops = static_cast<double>(size) * size * (2.0 * size);
    double time_seconds = time_ms / 1000.0;
    return total_ops / time_seconds;
}

// main
int main(int argc, char* argv[])
{
    MATAR_INITIALIZE(argc, argv);
    { // MATAR scope
    printf("Starting MATAR Matrix Multiplication test \n");
    printf("Matrix size: %d x %d\n", MATRIX_SIZE, MATRIX_SIZE);

    // Create arrays on the device, where the device is either the CPU or GPU depending on how it is compiled
    CArrayDevice<int> A(MATRIX_SIZE, MATRIX_SIZE);
    FArrayDevice<int> B(MATRIX_SIZE, MATRIX_SIZE);
    CArrayDevice<int> C(MATRIX_SIZE, MATRIX_SIZE);

    // Initialize arrays (NOTE: This is on the device)
    A.set_values(2);
    B.set_values(2);
    C.set_values(0);

    // Create and start timer
    Timer timer;
    timer.start();

    // Perform C = A * B
    FOR_ALL(i, 0, MATRIX_SIZE,
            j, 0, MATRIX_SIZE,
            k, 0, MATRIX_SIZE, {
        C(i,j) += A(i,k) * B(k,j);
    });

    // Add a fence to ensure all the operations are completed to get correct timing
    MATAR_FENCE();

    // Stop timer and get execution time
    double time_ms = timer.stop();
    
    // Calculate and print performance metrics
    double flops = calculate_flops(MATRIX_SIZE, time_ms);
    
    printf("Execution time: %.2f ms\n", time_ms);
    printf("Performance: %.2f GFLOPS\n", flops / 1e9);

    }
    MATAR_FINALIZE();

    return 0;
}