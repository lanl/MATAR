MATAR Matrix Multiplication Example
=================================

This example demonstrates how to perform matrix multiplication using MATAR's data structures and parallel execution capabilities. The example is designed to work both with and without Kokkos, making it a good starting point for learning MATAR.

Overview
--------
The example implements a simple matrix multiplication (C = A * B) using MATAR's parallel execution model. It includes:
- Matrix initialization
- Parallel matrix multiplication using FOR_ALL
- Performance measurement and reporting
- Proper initialization and finalization of MATAR/Kokkos

Key Features
-----------
- Uses MATAR's device arrays (CArrayDevice) for GPU/CPU execution
- Implements parallel matrix multiplication
- Includes performance measurement and FLOPS calculation
- Demonstrates proper MATAR initialization and finalization
- Shows how to use MATAR_FENCE for synchronization

Building and Running
------------------
1. Compile with Kokkos support for all back ends
   ```
   ./build.sh -t all
   ./matmul
   ```


2. Run the example:
   ```
   cd <build_backend>
   ./matmul
   ```

The program will:
1. Initialize MATAR (and Kokkos if enabled). If Kokkos is not enabled, the program will serially run on the CPU.
2. Create and initialize matrices
3. Perform parallel matrix multiplication
4. Measure and report execution time and performance
5. Clean up resources

Performance Metrics
-----------------
The example calculates and reports:
- Execution time in milliseconds
- Performance in GFLOPS (Giga Floating Point Operations Per Second)

The theoretical FLOPS calculation accounts for:
- Size * Size * (2 * Size) operations
  - Size multiplications per element
  - Size-1 additions per element

Tasks
-----
This example is structured into four parts, each focusing on different aspects of parallel matrix multiplication:

Part 1: The Basics
-----------------
Goal: Perform matrix matrix multiplication using different parallelism backends

Tasks:
1. Review code
2. Create and initialize 2 multi-dimensional arrays on the device
3. Initialize arrays
4. Implement multiplication. Hint: $$C_{ij} = A_{ik}\cdot B_{jk}$$
5. Build for all backends
6. Run for each backend, document performance
7. Test with multiple CPU cores: export OMP_NUM_THREADS=N
8. Discuss

Part 2: Thread Safety with Atomics
--------------------------------
Goal: Perform matrix matrix multiplication using different parallelism backends, this time getting the correct answer.

Tasks:
1. Leverage Kokkos::atomic_add to make the code thread safe
   - Kokkos::atomic_add(&C(i,j), A(i,k) * B(k,j));
2. Compile and run on all backends
3. Compare performance, discuss

Part 3: Thread Safety by Design
-----------------------------
Goal: Perform matrix matrix multiplication using different parallelism backends, this time without atomics

Tasks:
1. Find what is causing the race condition
2. Refactor FOR_ALL to remove race condition
3. Recompile, rerun, and document performance
4. Compare performance, discuss

Part 4: Performance Optimization
------------------------------
Goal: Perform matrix matrix multiplication using different parallelism backends, while minimizing memory writes

Tasks:
1. Find what is causing the race condition
2. Refactor FOR_ALL to remove race condition
3. Recompile, rerun, and document performance
4. Compare performance, discuss

Note: This is a template example where the actual matrix multiplication implementation is left as an exercise. The solution can be found in the solution/ directory.
