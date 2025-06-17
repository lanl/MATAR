# Heat Equation Solver Using Jacobi Method

## Overview

This example demonstrates how to solve the 2D heat equation using the Jacobi iterative method with the MATAR library. The heat equation is a partial differential equation that describes how heat distributes through a material over time.

## The Heat Equation and Laplace Equation

The heat equation in 2D is:

$$\frac{\partial T}{\partial t} = \alpha\left(\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2}\right)$$

When we're looking for the steady-state solution (where temperature no longer changes with time), this simplifies to the Laplace equation:

$$\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0$$

## Mathematical Derivation

Here we derive the numerical method used to solve the Laplace equation:

$$\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0$$

### Finite Difference Approximation

We first discretize the domain into a grid of points with spacing $\Delta x$ and $\Delta y$. Using the central difference approximation for the second derivatives:

$$\frac{\partial^2 T}{\partial x^2} \approx \frac{T_{i+1,j} - 2T_{i,j} + T_{i-1,j}}{(\Delta x)^2}$$

$$\frac{\partial^2 T}{\partial y^2} \approx \frac{T_{i,j+1} - 2T_{i,j} + T_{i,j-1}}{(\Delta y)^2}$$

Substituting these approximations into the Laplace equation:

$$\frac{T_{i+1,j} - 2T_{i,j} + T_{i-1,j}}{(\Delta x)^2} + \frac{T_{i,j+1} - 2T_{i,j} + T_{i,j-1}}{(\Delta y)^2} = 0$$

For simplicity, we assume $\Delta x = \Delta y$ (uniform grid spacing), giving us:

$$T_{i+1,j} - 4T_{i,j} + T_{i-1,j} + T_{i,j+1} + T_{i,j-1} = 0$$

Solving for $T_{i,j}$:

$$T_{i,j} = \frac{1}{4}(T_{i+1,j} + T_{i-1,j} + T_{i,j+1} + T_{i,j-1})$$

### Jacobi Iteration

This leads to the Jacobi iteration method, where we calculate new temperature values based on the previous iteration:

$$T_{i,j}^{(k+1)} = \frac{1}{4}(T_{i+1,j}^{(k)} + T_{i-1,j}^{(k)} + T_{i,j+1}^{(k)} + T_{i,j-1}^{(k)})$$

Where $k$ is the iteration number. We continue this process until convergence, determined by:

$$\max_{i,j} |T_{i,j}^{(k+1)} - T_{i,j}^{(k)}| < \text{tolerance}$$

The algorithm converges to the solution of the Laplace equation as $k \rightarrow \infty$, subject to the specified boundary conditions.

## The Jacobi Method

The Jacobi method is an iterative technique for solving systems of linear equations. For the Laplace equation on a 2D grid, it works by:

1. Starting with an initial guess for temperature at all grid points
2. For each interior grid point, updating its value based on the average of its four neighbors:
   ```
   T_new(i,j) = 0.25 * (T_old(i+1,j) + T_old(i-1,j) + T_old(i,j+1) + T_old(i,j-1))
   ```
3. Repeating until convergence (when the maximum change between iterations is below a threshold)

This is easy to understand and parallelize, though it converges slowly for large grids.

## Data Structures

This example uses MATAR's data structures:

- `CArrayDual<double>`: A dual-memory array that exists on both CPU (host) and GPU (device). 
  - The "Dual" part means the data can be accessed and modified on either CPU or GPU.
  - The `update_host()` method synchronizes data from the device to the host (important for visualization).

## Key Components of the Code

### Initialization

The `initialize()` function sets up:
- Interior points to 0.0 (initial guess)
- Boundary conditions:
  - Left boundary: 0.0
  - Right boundary: Linear gradient from 0 to 1000
  - Top boundary: 0.0
  - Bottom boundary: Linear gradient from 0 to 1000

### Main Computation Loop

The computation occurs in these key steps:

1. Update all interior points using the Jacobi method:
   ```cpp
   FOR_ALL(i, 1, height + 1,
           j, 1, width + 1, {
       temperature(i, j) = 0.25 * (temperature_previous(i + 1, j)
                                  + temperature_previous(i - 1, j)
                                  + temperature_previous(i, j + 1)
                                  + temperature_previous(i, j - 1));
   });
   ```

2. Check for convergence using a reduction operation:
   ```cpp
   FOR_REDUCE_MAX(i, 1, height + 1,
                  j, 1, width + 1,
                  local_max_value, {
       double value = fabs(temperature(i, j) - temperature_previous(i, j));
       
       if (value > local_max_value) {
           local_max_value = value;
       }
       
       // update temperature_previous
       temperature_previous(i, j) = temperature(i, j);
   }, max_value);
   ```

3. Visualize the temperature distribution periodically

### Parallel Computing Constructs

MATAR provides several parallel constructs:

- `FOR_ALL`: Executes a block of code for all elements in a specified range in parallel
- `FOR_REDUCE_MAX`: Performs a parallel reduction to find the maximum value across all elements

## Building and Running

1. Compile with Kokkos support for all back ends
   ```
   ./build.sh -t all
   ```

2. Run the example:
   ```
   cd <build_backend>
   ./heat
   ```

The visualization shows the temperature distribution using ASCII characters with colors, where cold regions are blue (represented as '.') and hot regions are red (represented as '#').

## Modifying the Example

You can experiment with:
- Different grid sizes (width and height)
- Different boundary conditions in the initialize() function
- Different convergence thresholds (temp_tolerance)
- Different visualization parameters

## Data-Oriented Design

This example follows data-oriented programming principles by:
- Separating data (temperature arrays) from operations
- Using efficient memory layouts for computation
- Enabling parallel operations through data-parallel constructs
- Minimizing data movement between host and device

## Tasks

### Part 1: The Basics

1. Review the code
2. Build serial non-MATAR version:
   ```
   ./build.sh -t serial
   ```
3. Run the code and document performance
4. Rebuild with optimization flags by modifying the CMakeLists.txt
   - Uncomment lines <>
5. Rebuild from build script
6. Run the code and note performance

### Part 2: MATARize

1. MATARize the code!
2. Change the data structures from default C++ arrays to be CArrayDual types
3. Update function signatures to take in CArrayDual<double>& 
4. Update all accesses to temperature and temperature previous to use () instead of []
5. Build for serial and run, document performance
   ```
   ./build.sh -t serial
   ```
6. Discuss issues for GPU portability

### Part 3: Parallelize

1. Parallelize the code!
2. Change the for loops and reduction operations to use FOR_ALL and FOR_REDUCE_MAX operations. 
3. Rebuild in serial, note performance
4. Rebuild with OpenMP
   ```
   ./build.sh -t openmp
   export OMP_NUM_THREADS=N
   ```
5. Run with multiple different N values, document performance
6. Discuss

### Part 4: GPU Portability

1. Make the code GPU ready
2. Add .update_host() and .update_device() to the data structures as needed
3. Add MATAR_FENCE() where needed
4. Add .host() when accessing data on the host side for outputs
5. Build with CUDA
   ```
   ./build.sh -t cuda
   ```
6. Run, and document performance
7. Discuss and perform micro-optimizations
8. Rebuild for other backends and document performance
