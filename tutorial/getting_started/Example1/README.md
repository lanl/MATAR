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

Note: This is a template example where the actual matrix multiplication implementation is left as an exercise. The solution can be found in the solution/ directory.
