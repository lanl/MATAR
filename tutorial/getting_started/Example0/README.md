# MATAR Data Types Example

This example demonstrates all the data types supported by MATAR, including both serial (host) and Kokkos (device) variants. It serves as a comprehensive reference for understanding and using MATAR's data structures.

## Overview

The example covers:
1. Serial (Host) Data Types
2. Kokkos (Device) Data Types (when Kokkos is enabled)
3. Views of existing data structures
4. Different memory layouts (C-style and F-style)
5. Specialized data structures (ragged and sparse arrays)

## Data Types Demonstrated

### 1. Dense Arrays
- **C-style (Row-major)**
  - `CArray<T>`: 1D, 2D, and 3D arrays
  - `CArrayDevice<T>`: Kokkos version for GPU/CPU execution
- **F-style (Column-major)**
  - `FArray<T>`: 1D, 2D, and 3D arrays
  - `FArrayDevice<T>`: Kokkos version for GPU/CPU execution

### 2. Views
- **C-style Views**
  - `ViewCArray<T>`: Views of C-style arrays
  - `ViewCArrayDevice<T>`: Kokkos version
- **F-style Views**
  - `ViewFArray<T>`: Views of F-style arrays
  - `ViewFArrayDevice<T>`: Kokkos version

Views can be created from:
- Existing MATAR arrays
- Raw C arrays
- std::vector (with caution - must be contiguous)

### 3. Ragged Arrays
- **C-style Ragged**
  - `RaggedCArray<T>`: Fixed-size ragged arrays
  - `RaggedCArrayDevice<T>`: Kokkos version
- **F-style Ragged**
  - `RaggedFArray<T>`: Fixed-size ragged arrays
  - `RaggedFArrayDevice<T>`: Kokkos version

### 4. Dynamic Ragged Arrays
- **C-style Dynamic**
  - `DynamicRaggedCArray<T>`: Resizable ragged arrays
  - `DynamicRaggedCArrayDevice<T>`: Kokkos version
- **F-style Dynamic**
  - `DynamicRaggedFArray<T>`: Resizable ragged arrays
  - `DynamicRaggedFArrayDevice<T>`: Kokkos version

### 5. Sparse Arrays
- **CSR Format**
  - `CSRArray<T>`: Compressed Sparse Row format
  - `CSRArrayDevice<T>`: Kokkos version
- **CSC Format**
  - `CSCArray<T>`: Compressed Sparse Column format
  - `CSCArrayDevice<T>`: Kokkos version

## Usage Examples

### Creating Arrays
```cpp
// Dense arrays
CArray<int> carr_1D(10);           // 1D C-style array
FArray<int> farr_2D(10, 10);       // 2D F-style array

// Views
int A[10];
ViewCArray<int> view_A(A, 10);     // View of C array
std::vector<int> B(10);
ViewCArray<int> view_B(B.data(), 10); // View of vector (must be contiguous)

// Ragged arrays
RaggedCArray<int> ragged_carr(10);  // 10 rows
FOR_ALL(i, 0, 10, {
    ragged_carr(i) = CArray<int>(i+1);  // Each row has i+1 elements
});
```

### Kokkos Arrays (when enabled)
```cpp
#ifdef HAVE_KOKKOS
CArrayDevice<int> carr_dev_1D(10);
FArrayDevice<int> farr_dev_2D(10, 10);
#endif
```

## Best Practices

1. **Memory Layout Selection**
   - Use C-style for row-major operations
   - Use F-style for column-major operations
   - Consider your primary access pattern

2. **Views**
   - Use views to avoid data copying
   - Be careful with std::vector views (must be contiguous)
   - Views maintain the original memory layout

3. **Ragged Arrays**
   - Use fixed-size when dimensions are known
   - Use dynamic when dimensions change
   - Consider memory overhead vs flexibility

4. **Sparse Arrays**
   - Use CSR for row-wise operations
   - Use CSC for column-wise operations
   - Consider sparsity pattern when choosing format

## Building and Running

1. Compile with appropriate MATAR and Kokkos support:
   ```bash
   make
   ```

2. Run the example:
   ```bash
   ./matar_data
   ```

The program will demonstrate the creation and basic usage of all MATAR data types.

## Notes

- Kokkos types are only available when compiled with Kokkos support
- Views provide zero-copy access to existing data
- Memory layout affects performance based on access patterns
- Choose data types based on your specific needs and access patterns 