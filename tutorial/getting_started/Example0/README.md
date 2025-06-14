# MATAR Data Types Example

This example demonstrates the core data types and parallel programming patterns supported by MATAR, highlighting data-oriented design principles and performance portability across CPU and GPU architectures.

## Overview

The MATAR_data.cpp example demonstrates:
1. Dense array data structures (C-style and F-style)
2. Array views for zero-copy data access
3. Dual arrays for host/device memory management
4. Ragged arrays for variable-length data
5. Sparse matrix formats for memory-efficient storage
6. Parallel execution patterns using MATAR macros
7. Memory synchronization and device coordination

## Tasks

To work through this example:

1. Review build.sh
2. Review CMakeLists.txt
3. Review MATAR_data.cpp
4. Load the CUDA module:
   ```bash
   module load cuda
   ```
5. Build for serial backend:
   ```bash
   ./build.sh -t serial
   ```
   Note: You may need to make the build script executable first:
   ```bash
   chmod +x build.sh
   ```
6. Run serially:
   ```bash
   ./build_serial/MATAR_data
   ```
7. Build for all backends:
   ```bash
   ./build.sh -t all
   ```
8. Run other backends as desired

## Data Types and Concepts Demonstrated

### 1. Dense Arrays
- **C-style (Row-major)**
  - `CArrayDevice<T>`: Device arrays with 0-based indexing
  - `CMatrixDevice<T>`: Device arrays with 1-based indexing (for mathematical applications)
- **F-style (Column-major)**
  - `FArrayDevice<T>`: Device arrays with column-major memory layout

### 2. Array Views
- **Views of existing arrays**
  - `ViewCArrayDevice<T>`: Views of C-style device arrays
  - `ViewFArrayDevice<T>`: Views of F-style device arrays
  - `ViewCArrayDual<T>`: Views with both host and device access

### 3. Dual Arrays (Host+Device)
- **Arrays with explicit host/device memory management**
  - `CArrayDual<T>`: C-style arrays with separate host/device accessors
  - Includes methods: `host()`, `update_device()`, `update_host()`

### 4. Ragged Arrays
- **Variable-length arrays**
  - `RaggedCArrayDevice<T>`: Device-side ragged arrays
  - `RaggedCArrayDual<T>`: Ragged arrays with host/device memory management

### 5. Sparse Arrays
- **Compressed Sparse Column (CSC)**
  - `CSCArrayDevice<T>`: Memory-efficient storage for sparse matrices
  - Demonstrated with explicit values, rows, and column indices

## MATAR Macros for Parallel Programming

The example demonstrates several key MATAR macros:

### 1. Execution Macros
- **FOR_ALL(i, start, end, { ... })**: Parallel execution over index ranges
  - Supports 1D, 2D, and 3D parallelism
  - Automatically maps to appropriate hardware (CPU threads or GPU)

- **RUN({ ... })**: Execute code block once on the device in serial
  - Useful for initialization and diagnostics

### 2. Reduction Macros
- **FOR_REDUCE_SUM(i, start, end, local_var, { ... }, result)**: Parallel reduction
  - Efficiently combines results from parallel operations
  - Supports sum, min, and max reductions (example shows sum)

### 3. Synchronization
- **MATAR_FENCE()**: Synchronization barrier
  - Ensures device operations complete before proceeding
  - Critical for correctness when there are data dependencies

## Key Programming Patterns

### 1. Host Initialization, Device Computation
```cpp
// Initialize on host
for (int i = 0; i < 10; i++) {
    d_carr_1D.host(i) = i;
}

// Transfer to device
d_carr_1D.update_device();

// Compute on device
FOR_ALL(i, 0, 10, {
    // Device-side operations
});
```

### 2. View-based Operations
```cpp
// Create view of existing data
ViewCArrayDevice<int> view_carr_dev_1D(carr_dev_1D.pointer(), 10);

// Operate on view (modifies original data)
FOR_ALL(i, 0, 10, {
    view_carr_dev_1D(i) -= i;
});
```

### 3. Parallel Reductions
```cpp
// Sum reduction across array elements
int loc_sum_1D = 0;
int sum_1D = 0;
FOR_REDUCE_SUM(i, 0, 10,
               loc_sum_1D, {
    loc_sum_1D += d_carr_1D(i);
}, sum_1D);
```

### 4. Working with Ragged Data
```cpp
// Create and fill ragged array
RaggedCArrayDevice<int> ragged_carr_dev(some_strides, "test_1D");
FOR_ALL(i, 0, num_strides, {
    for(int j = 0; j < ragged_carr_dev.stride(i); j++) {
        ragged_carr_dev(i, j) = j + 1;
    }
});
```

## Memory Management Best Practices

1. **Use explicit synchronization**
   - Call `MATAR_FENCE()` after device operations to ensure completion
   - Necessary when results are needed by subsequent operations

2. **Minimize host/device transfers**
   - Group operations on the same memory space
   - Use dual arrays only when both host and device access is needed

3. **Use views when possible**
   - Avoid unnecessary data copying with views
   - Views provide zero-overhead access to existing data

4. **Choose appropriate memory layout**
   - C-style (row-major) for row-wise operations
   - F-style (column-major) for column-wise operations

## Building and Running

1. Compile with Kokkos support:
   ```bash
   cd tutorial/getting_started/Example0
   mkdir build && cd build
   cmake .. -DMATAR_ENABLE_KOKKOS=ON
   make
   ```

2. Run the example:
   ```bash
   ./MATAR_data
   ```

The program will demonstrate the creation and usage of MATAR data structures with output indicating successful completion of each test.

## Performance Portability

This example automatically runs efficiently across:
- Multi-core CPUs (via OpenMP or Pthreads)
- NVIDIA GPUs (via CUDA)
- AMD GPUs (via HIP)
- Intel GPUs (via SYCL)

The same code delivers performance across all these architectures without modification, demonstrating MATAR's performance portability features. 