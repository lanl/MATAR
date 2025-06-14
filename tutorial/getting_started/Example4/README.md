# Linking MATAR with Fortran

This example demonstrates how to link MATAR (a C++ library) with Fortran code, showing the key concepts and best practices for interoperability between the two languages.

## Overview

The example consists of three main components:
1. A Fortran main program (`main.f90`)
2. A C++ file containing MATAR functions (`matar_function.cpp`)
3. The necessary linking between them

## Key Concepts

### 1. C++ Function Declarations
- Functions that will be called from Fortran must be declared with `extern "C"` to prevent name mangling
- Function names must end with an underscore (`_`) to match Fortran's calling convention
- Example:
```cpp
extern "C" void square_array_elements_(double* array, int* nx, int* ny);
```

### 2. Fortran-C++ Type Mapping
- Use `iso_c_binding` module in Fortran to ensure correct type mapping
- Common mappings:
  - `real(kind=c_double)` in Fortran $$\rightarrow$$ `double` in C++
  - `integer(kind=c_int)` in Fortran $$\rightarrow$$ `int` in C++

### 3. Array Handling
- Fortran arrays are passed as pointers in C++
- MATAR provides `DViewFMatrixKokkos` for handling Fortran-allocated arrays
- Example:
```cpp
auto array_2D_dual_view = DViewFMatrixKokkos<double>(array, nx_, ny_);
```

### 4. Kokkos Integration
- Kokkos must be initialized before using MATAR functions
- Initialization and finalization are handled through C++ functions called from Fortran
```fortran
call kokkos_initialize()
! ... use MATAR functions ...
call kokkos_finalize()
```

## Example Walkthrough

This example demonstrates:
1. Initializing a 2D array in Fortran
2. Passing it to a C++ function that squares each element using MATAR
3. Computing the sum of elements using MATAR's reduction capabilities

### Key Functions

1. `square_array_elements_`: Squares each element of a 2D array using MATAR's parallel execution
2. `sum_array_elements_`: Computes the sum of array elements using MATAR's reduction capabilities

## Building and Running

To build this example:
1. Ensure MATAR and Kokkos are properly installed
2. Compile the C++ code with appropriate Kokkos flags
3. Compile the Fortran code
4. Link the object files together

## Best Practices

1. Always use `iso_c_binding` for type safety
2. Keep C++ function names consistent with Fortran calling conventions
3. Use MATAR's view types for array operations
4. Properly initialize and finalize Kokkos
5. Handle array indexing carefully (Fortran is 1-based, C++ is 0-based)

## Tasks

1. Review the code
2. Create the required data type to hold the incoming array from Fortran
3. Fill in the `sum_array_elements_` and the `square_array_elements_` functions using a `FOR_ALL` and a `FOR_REDUCE_SUM`
4. Build for serial, openMP, and CUDA backends
5. Run each and document performance
6. Replace the `FOR_` with a `DO_`
7. Rebuild for serial, openMP, and CUDA, rerun and note performance

## Notes

- This example uses a $$4 \times 4$$ array for demonstration
- MATAR's parallel execution capabilities are demonstrated through the `DO_ALL` and `DO_REDUCE_SUM` macros
