#include <stdio.h>
#include <iostream>
#include <chrono>
#include "matar.h"

// Required for MATAR data structures
using namespace mtr; 

#define MATRIX_SIZE 1000

// Tile size for matrix multiplication
// Adjust this value based on hardware cache sizes
// Smaller values (16-32) typically work well for GPUs
// Larger values (32-64) may be better for CPUs with larger caches
#define TILE_SIZE 32

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
    // Note: A is row-major (CArrayDevice), B is column-major (FArrayDevice)
    // This layout optimizes memory access patterns in the multiplication
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

    // Calculate number of tiles in each dimension
    // This handles cases where the matrix size isn't a multiple of tile size
    int num_tiles = (MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE;

    /**
     * Tiled Matrix Multiplication Algorithm
     *
     * This implementation uses a 3D tiling approach:
     * 1. The output matrix C is divided into tiles of size TILE_SIZE × TILE_SIZE
     * 2. The k-dimension is also tiled to maximize cache reuse (critical optimization)
     * 3. Each tile is assigned to a team (using FOR_FIRST)
     * 4. Within each team, rows are distributed across threads (using FOR_SECOND)
     *
     * Benefits of 3D tiling:
     * - Spatial locality: Elements accessed together are stored together
     * - Temporal locality: Data loaded into cache is reused multiple times before eviction
     * - Reduced memory traffic: Each tile of A and B can be loaded once and reused
     * - Arithmetic intensity: Ratio of computations to memory accesses is improved
     */
    FOR_FIRST(tile_idx, 0, num_tiles*num_tiles, {
        // Convert linear tile index to 2D tile coordinates
        int tile_i = tile_idx / num_tiles;
        int tile_j = tile_idx % num_tiles;
        
        // Calculate starting indices for this tile
        int i_start = tile_i * TILE_SIZE;
        int j_start = tile_j * TILE_SIZE;
        
        // Calculate ending indices, clamping to matrix size
        int i_end = (i_start + TILE_SIZE < MATRIX_SIZE) ? 
                     i_start + TILE_SIZE : MATRIX_SIZE;
        int j_end = (j_start + TILE_SIZE < MATRIX_SIZE) ? 
                     j_start + TILE_SIZE : MATRIX_SIZE;
        
        // Process this tile - each thread handles one row of the tile
        FOR_SECOND(local_i, 0, i_end - i_start, {
            int i = i_start + local_i;
            
            // For each column in this tile
            for (int j = j_start; j < j_end; j++) {
                // Initialize local accumulator for C(i,j)
                int local_sum = 0;
                
                // Tile the k-dimension for better cache locality
                // Process k in chunks of TILE_SIZE
                for (int k_tile = 0; k_tile < MATRIX_SIZE; k_tile += TILE_SIZE) {
                    // Calculate ending index for k-tile, clamping to matrix size
                    int k_end = (k_tile + TILE_SIZE < MATRIX_SIZE) ? 
                                k_tile + TILE_SIZE : MATRIX_SIZE;
                    
                    // Process this k-tile
                    for (int k = k_tile; k < k_end; k++) {
                        // Accumulate into local sum - this prevents repeatedly
                        // reading/writing to global memory
                        local_sum += A(i, k) * B(k, j);
                    }
                }
                
                // Write final result to global memory only once
                C(i, j) = local_sum;
            }
        });
    });

    // Add a fence to ensure all the operations are completed before timing
    MATAR_FENCE();

    // Stop timer and get execution time
    double time_ms = timer.stop();
    
    // Calculate and print performance metrics
    double flops = calculate_flops(MATRIX_SIZE, time_ms);
    
    printf("Execution time: %.2f ms\n", time_ms);
    printf("Performance: %.2f GFLOPS\n", flops / 1e9);
    printf("Arithmetic intensity: %.2f FLOPS/byte\n", 
           (2.0 * MATRIX_SIZE) / (3.0 * sizeof(int))); // 2N operations per 3 elements

    }
    MATAR_FINALIZE();

    return 0;
}