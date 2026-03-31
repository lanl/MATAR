# MATAR Code Conversion Reference

This document provides comprehensive context for converting existing C/C++ (and Fortran-interop) code to use the MATAR library. MATAR (Multi-dimensional Arrays, Tensors, And Ragged data structures) is a header-only C++ library that provides performance-portable data structures and parallel loop macros. It abstracts over Kokkos to support CPU serial, OpenMP, Pthreads, CUDA, and HIP backends with a single source code.

---

## Table of Contents

1. [Setup and Boilerplate](#1-setup-and-boilerplate)
2. [Data Structure Taxonomy](#2-data-structure-taxonomy)
3. [Dense Data Types — Detailed API](#3-dense-data-types--detailed-api)
4. [Sparse and Ragged Data Types](#4-sparse-and-ragged-data-types)
5. [Parallel Loop Macros](#5-parallel-loop-macros)
6. [Reduction Macros](#6-reduction-macros)
7. [Hierarchical (Team/Thread) Parallelism](#7-hierarchical-teamthread-parallelism)
8. [Class-Member Lambda Macros](#8-class-member-lambda-macros)
9. [Host/Device Data Transfer (Dual Types)](#9-hostdevice-data-transfer-dual-types)
10. [Conversion Rules: Standard C++ to MATAR](#10-conversion-rules-standard-c-to-matar)
11. [Complete Before/After Examples](#11-complete-beforeafter-examples)
12. [Common Pitfalls](#12-common-pitfalls)
13. [Type Alias Quick Reference](#13-type-alias-quick-reference)
14. [Device Kernel Constraints](#14-device-kernel-constraints)
15. [Fortran Interoperability](#15-fortran-interoperability)
16. [Build Configuration](#16-build-configuration)

---

## 1. Setup and Boilerplate

Every MATAR program requires:

```cpp
#include <matar.h>
using namespace mtr;

int main(int argc, char* argv[])
{
    MATAR_INITIALIZE(argc, argv);  // or MATAR_INITIALIZE() without args
    {
        // All MATAR data structures and parallel work go here.
        // The scoping braces ensure all MATAR objects are destroyed
        // before MATAR_FINALIZE is called.
    }
    MATAR_FINALIZE();
    return 0;
}
```

| Macro | Behavior with Kokkos | Behavior without Kokkos |
|-------|----------------------|------------------------|
| `MATAR_INITIALIZE(...)` | `Kokkos::initialize(...)` | No-op |
| `MATAR_FINALIZE()` | `Kokkos::finalize()` | No-op |
| `MATAR_FENCE()` | `Kokkos::fence()` | No-op |

`MATAR_FENCE()` is a synchronization barrier. It ensures all device operations have completed before the CPU proceeds. Place it after any parallel kernel that produces data consumed by subsequent host code or another kernel with a data dependency.

---

## 2. Data Structure Taxonomy

MATAR data structures are organized along three axes:

### Axis 1: Memory Layout

| Prefix | Layout | Index Convention | Best Loop Order |
|--------|--------|-----------------|-----------------|
| **C** | Row-major (C-style) | Last index varies fastest in memory | Outermost loop = first index |
| **F** | Column-major (Fortran-style) | First index varies fastest in memory | Outermost loop = last index |

### Axis 2: Index Base

| Suffix | Index Base | Range |
|--------|-----------|-------|
| **Array** | 0-based | `[0, N)` |
| **Matrix** | 1-based | `[1, N]` |

### Axis 3: Memory Residence

| Suffix | Resides On | Usage |
|--------|-----------|-------|
| **Host** | CPU only | Serial code, I/O, initialization |
| **Device** | Device only (GPU or CPU depending on build) | Parallel kernels |
| **Dual** | Both host and device, with explicit sync | Mixed host/device workflows |

### Axis 4: Ownership

| Prefix | Owns Memory? | Construction |
|--------|-------------|--------------|
| (none) | Yes — allocates and manages storage | `CArrayDevice<int>(10, 10)` |
| **View** | No — wraps an existing pointer | `ViewCArrayDevice<int>(ptr, 10, 10)` |

### Complete Naming Map (via `mtr::` aliases)

| Alias (preferred) | Underlying Class | Layout | Index | Residence |
|---|---|---|---|---|
| `CArrayHost<T>` | `CArray<T>` | Row-major | 0-based | Host |
| `CMatrixHost<T>` | `CMatrix<T>` | Row-major | 1-based | Host |
| `FArrayHost<T>` | `FArray<T>` | Column-major | 0-based | Host |
| `FMatrixHost<T>` | `FMatrix<T>` | Column-major | 1-based | Host |
| `ViewCArrayHost<T>` | `ViewCArray<T>` | Row-major | 0-based | Host (non-owning) |
| `ViewCMatrixHost<T>` | `ViewCMatrix<T>` | Row-major | 1-based | Host (non-owning) |
| `ViewFArrayHost<T>` | `ViewFArray<T>` | Column-major | 0-based | Host (non-owning) |
| `ViewFMatrixHost<T>` | `ViewFMatrix<T>` | Column-major | 1-based | Host (non-owning) |
| `CArrayDevice<T>` | `CArrayKokkos<T>` | Row-major | 0-based | Device |
| `CMatrixDevice<T>` | `CMatrixKokkos<T>` | Row-major | 1-based | Device |
| `FArrayDevice<T>` | `FArrayKokkos<T>` | Column-major | 0-based | Device |
| `FMatrixDevice<T>` | `FMatrixKokkos<T>` | Column-major | 1-based | Device |
| `ViewCArrayDevice<T>` | `ViewCArrayKokkos<T>` | Row-major | 0-based | Device (non-owning) |
| `ViewCMatrixDevice<T>` | `ViewCMatrixKokkos<T>` | Row-major | 1-based | Device (non-owning) |
| `ViewFArrayDevice<T>` | `ViewFArrayKokkos<T>` | Column-major | 0-based | Device (non-owning) |
| `ViewFMatrixDevice<T>` | `ViewFMatrixKokkos<T>` | Column-major | 1-based | Device (non-owning) |
| `CArrayDual<T>` | `DCArrayKokkos<T>` | Row-major | 0-based | Dual |
| `CMatrixDual<T>` | `DCMatrixKokkos<T>` | Row-major | 1-based | Dual |
| `FArrayDual<T>` | `DFArrayKokkos<T>` | Column-major | 0-based | Dual |
| `FMatrixDual<T>` | `DFMatrixKokkos<T>` | Column-major | 1-based | Dual |
| `ViewCArrayDual<T>` | `DViewCArrayKokkos<T>` | Row-major | 0-based | Dual (non-owning) |
| `ViewCMatrixDual<T>` | `DViewCMatrixKokkos<T>` | Row-major | 1-based | Dual (non-owning) |
| `ViewFArrayDual<T>` | `DViewFArrayKokkos<T>` | Column-major | 0-based | Dual (non-owning) |
| `ViewFMatrixDual<T>` | `DViewFMatrixKokkos<T>` | Column-major | 1-based | Dual (non-owning) |

### Sparse/Ragged Type Aliases

| Alias | Underlying Class | Description |
|---|---|---|
| `RaggedCArrayHost<T>` | `RaggedRightArray<T>` | Variable-length rows, host |
| `RaggedFArrayHost<T>` | `RaggedDownArray<T>` | Variable-length columns, host |
| `DynamicRaggedCArrayHost<T>` | `DynamicRaggedRightArray<T>` | Dynamic row lengths, host |
| `DynamicRaggedFArrayHost<T>` | `DynamicRaggedDownArray<T>` | Dynamic column lengths, host |
| `CSRArrayHost<T>` | `CSRArray<T>` | Compressed Sparse Row, host |
| `CSCArrayHost<T>` | `CSCArray<T>` | Compressed Sparse Column, host |
| `RaggedCArrayDevice<T>` | `RaggedRightArrayKokkos<T>` | Variable-length rows, device |
| `RaggedCArrayDual<T>` | `DRaggedRightArrayKokkos<T>` | Variable-length rows, dual |
| `RaggedFArrayDevice<T>` | `RaggedDownArrayKokkos<T>` | Variable-length columns, device |
| `DynamicRaggedCArrayDevice<T>` | `DynamicRaggedRightArrayKokkos<T>` | Dynamic rows, device |
| `DynamicRaggedFArrayDevice<T>` | `DynamicRaggedDownArrayKokkos<T>` | Dynamic columns, device |
| `CSRArrayDevice<T>` | `CSRArrayKokkos<T>` | Compressed Sparse Row, device |
| `CSCArrayDevice<T>` | `CSCArrayKokkos<T>` | Compressed Sparse Column, device |

---

## 3. Dense Data Types — Detailed API

All dense types support up to 7 dimensions.

### Construction

```cpp
// Owning types — allocate memory
CArrayDevice<double> a(10);                  // 1D, 10 elements
CArrayDevice<double> b(10, 20);              // 2D, 10x20
CArrayDevice<double> c(10, 20, 30);          // 3D
CArrayDual<double> d(10, 20, "my_label");    // Dual with optional debug label

// View types — wrap existing pointer, no allocation
ViewCArrayDevice<int> v(ptr, 10, 20);        // Wraps raw pointer with 2D shape

// Host-only types
CArray<double> h(100, 100);                  // Host-only, owning
ViewCArray<double> hv(raw_ptr, 100, 100);    // Host-only, non-owning
```

### Element Access

All types use parenthesis `()` operator for multidimensional indexing — never `[]`.

```cpp
// CArray / CArrayDevice / CArrayDual: 0-based, row-major
a(i)           // 1D
b(i, j)        // 2D: row i, col j
c(i, j, k)     // 3D

// CMatrix / CMatrixDevice / CMatrixDual: 1-based, row-major
m(1)           // first element
m(i, j)        // 1-based: i in [1..rows], j in [1..cols]

// FArray / FArrayDevice / FArrayDual: 0-based, column-major
f(i, j)        // column-major: j varies slowest

// FMatrix / FMatrixDevice / FMatrixDual: 1-based, column-major
fm(i, j)       // 1-based, column-major

// Dual types — host accessor
d.host(i, j)   // access on host side (CPU)
d(i, j)        // access on device side (inside FOR_ALL / RUN)
```

### Common Methods

| Method | Description |
|--------|-------------|
| `size()` | Total number of elements |
| `extent()` | Same as `size()` |
| `dims(n)` | Size along dimension `n` (0-indexed for Arrays, 1-indexed for Matrices) |
| `order()` | Number of dimensions |
| `pointer()` | Raw pointer to underlying data (device pointer for Device types) |
| `device_pointer()` | Explicit device pointer (Dual types) |
| `host_pointer()` | Explicit host pointer (Dual types) |
| `set_values(val)` | Set all elements to `val` (parallel on device) |
| `update_host()` | Copy device data to host (Dual types only) |
| `update_device()` | Copy host data to device (Dual types only) |
| `get_kokkos_view()` | Access the underlying `Kokkos::View` (Device types) |
| `get_kokkos_dual_view()` | Access the underlying `Kokkos::DualView` (Dual types) |

---

## 4. Sparse and Ragged Data Types

### Ragged Arrays

Ragged arrays store rows (or columns) of variable length. They are packed contiguously in memory with no wasted space.

```cpp
// Define per-row lengths
CArrayDevice<size_t> strides(num_rows);
FOR_ALL(i, 0, num_rows, {
    strides(i) = compute_row_length(i);
});

// Construct ragged array
RaggedCArrayDevice<int> ragged(strides, "my_ragged");

// Access: row i, position j within that row
FOR_ALL(i, 0, num_rows, {
    for (int j = 0; j < ragged.stride(i); j++) {
        ragged(i, j) = some_value;
    }
});

// Dual ragged — initialize from host-side stride array
size_t host_strides[5] = {1, 2, 3, 4, 5};
RaggedCArrayDual<int> ragged_dual(host_strides, 5, "dual_ragged");

// Fill on host
for (int i = 0; i < 5; i++) {
    for (int j = 0; j < ragged_dual.stride_host(i); j++) {
        ragged_dual.host(i, j) = value;
    }
}
ragged_dual.update_device();  // sync to device
```

### Dynamic Ragged Arrays

Row lengths can be updated dynamically during computation. Useful when the structure is not known a priori.

```cpp
DynamicRaggedCArrayDevice<double> dyn_ragged(num_rows, max_cols_per_row);

FOR_ALL(i, 0, num_rows, {
    for (int j = 0; j < some_condition(i); j++) {
        dyn_ragged.stride(i)++;  // extend this row
        dyn_ragged(i, j) = value;
    }
});
```

### CSR / CSC Arrays

Compressed sparse formats for matrix operations.

```cpp
// CSC construction from components
CArrayDevice<size_t> col_starts(num_cols + 1);
CArrayDevice<size_t> row_indices(nnz);
CArrayDevice<double> values(nnz);

// Fill col_starts, row_indices, values in a RUN block or FOR_ALL...

CSCArrayDevice<double> csc(values, col_starts, row_indices, num_rows, num_cols, "sparse_mat");

// Access
RUN({
    printf("Matrix is %ld x %ld with %ld nonzeros\n", csc.dim1(), csc.dim2(), csc.nnz());
    double val = csc(row, col);  // random access (expensive for sparse)
});
```

---

## 5. Parallel Loop Macros

### Macro Argument Count Reference

The macros use variadic dispatch based on argument count. This table shows the exact signatures:

| Macro | 1D args (4) | 2D args (7) | 3D args (10) |
|-------|------------|------------|-------------|
| `FOR_ALL` | `(i, lo, hi, {body})` | `(i, lo, hi, j, lo, hi, {body})` | `(i, lo, hi, j, lo, hi, k, lo, hi, {body})` |
| `DO_ALL` | same pattern | same pattern | same pattern |
| `FOR_REDUCE_SUM` | `(i, lo, hi, var, {body}, result)` (6 args) | `(i, lo, hi, j, lo, hi, var, {body}, result)` (9 args) | `(i, lo, hi, j, lo, hi, k, lo, hi, var, {body}, result)` (12 args) |
| `FOR_REDUCE_MAX` | same as SUM | same as SUM | same as SUM |
| `FOR_REDUCE_MIN` | same as SUM | same as SUM | same as SUM |
| `FOR_REDUCE_PRODUCT` | same as SUM | same as SUM | same as SUM |

The `DO_REDUCE_*` variants follow the same argument patterns as `FOR_REDUCE_*` but with inclusive upper bounds.

### `FOR_ALL` — C-style half-open range `[start, end)`

The primary parallel loop macro. Indices iterate over `[start, end)`. Supports 1D, 2D, and 3D.

```cpp
// 1D: 4 arguments
FOR_ALL(i, 0, N, {
    a(i) = i;
});

// 2D: 7 arguments
FOR_ALL(i, 0, N,
        j, 0, M, {
    a(i, j) = i + j;
});

// 3D: 10 arguments
FOR_ALL(i, 0, N,
        j, 0, M,
        k, 0, P, {
    a(i, j, k) = i + j + k;
});
```

**Loop ordering:** The inner loop varies the fastest and the outer loop varies the slowest. For optimal cache performance, match the loop index order to the data type's memory layout:
- `CArray` / `CArrayDevice`: last index fastest → `FOR_ALL(i, ..., j, ..., { a(i,j) })` is correct
- `FArray` / `FArrayDevice`: first index fastest → `FOR_ALL(i, ..., j, ..., { a(j,i) })` maps the fast-varying loop index to the fast-varying array index

### `DO_ALL` — Fortran-style inclusive range `[start, end]`

Identical to `FOR_ALL` but the upper bound is **inclusive**. Designed for 1-based Fortran-style indexing.

```cpp
// 1D: iterates i = 1, 2, ..., N
DO_ALL(i, 1, N, {
    matrix(i) = i;
});

// 2D: iterates i = 1..N, j = 1..M
DO_ALL(j, 1, M,
       i, 1, N, {
    matrix(i, j) = i * j;
});
```

### `RUN` — Execute Once on Device

Runs a block of code once (single iteration) on the device. Useful for serial device-side operations like printing, single-element assignments, or constructing sparse data.

```cpp
RUN({
    for (int i = 0; i < n; i++) {
        printf("a(%d) = %d\n", i, a(i));
    }
});
```

### `FOR_LOOP` / `DO_LOOP` — Serial Convenience Loops

Serial (non-parallel) loops with lambda-based syntax. `FOR_LOOP` uses half-open ranges; `DO_LOOP` uses inclusive ranges.

```cpp
FOR_LOOP(i, 0, N, {
    // serial iteration i = 0 .. N-1
});

DO_LOOP(i, 1, N, {
    // serial iteration i = 1 .. N
});
```

---

## 6. Reduction Macros

Reduction macros perform parallel reductions over index spaces. Each thread maintains a local reduction variable; the final result is combined into the output variable.

### `FOR_REDUCE_SUM` — Sum Reduction (half-open `[start, end)`)

```cpp
// 1D
double loc_sum = 0.0;
double total = 0.0;
FOR_REDUCE_SUM(i, 0, N,
               loc_sum, {
    loc_sum += a(i);
}, total);

// 2D
FOR_REDUCE_SUM(i, 0, N,
               j, 0, M,
               loc_sum, {
    loc_sum += a(i, j);
}, total);

// 3D
FOR_REDUCE_SUM(i, 0, N,
               j, 0, M,
               k, 0, P,
               loc_sum, {
    loc_sum += a(i, j, k);
}, total);
```

### `FOR_REDUCE_MAX` — Maximum Reduction

```cpp
double loc_max = 0.0;
double global_max = 0.0;
FOR_REDUCE_MAX(i, 0, N,
               j, 0, M,
               loc_max, {
    double val = fabs(a(i, j));
    if (val > loc_max) loc_max = val;
}, global_max);
```

### `FOR_REDUCE_MIN` — Minimum Reduction

```cpp
double loc_min = 0.0;
double global_min = 0.0;
FOR_REDUCE_MIN(i, 0, N,
               loc_min, {
    if (a(i) < loc_min) loc_min = a(i);
}, global_min);
```

### `FOR_REDUCE_PRODUCT` — Product Reduction

```cpp
double loc_prod = 1.0;
double global_prod = 1.0;
FOR_REDUCE_PRODUCT(i, 0, N,
                   loc_prod, {
    loc_prod *= a(i);
}, global_prod);
```

### `DO_REDUCE_*` — Inclusive-Range Variants

All reduction macros have `DO_REDUCE_*` counterparts with inclusive upper bounds:

```cpp
DO_REDUCE_SUM(i, 1, N, loc_sum, { loc_sum += a(i); }, total);
DO_REDUCE_MAX(i, 1, N, loc_max, { ... }, global_max);
DO_REDUCE_MIN(i, 1, N, loc_min, { ... }, global_min);
```

---

## 7. Hierarchical (Team/Thread) Parallelism

For nested parallelism (e.g., tiled matrix multiply), MATAR provides hierarchical macros that map to Kokkos team policies.

```cpp
// Outer level: teams (one per league member)
FOR_FIRST(i, 0, N, {

    // Middle level: team threads
    FOR_SECOND(j, 0, M, {

        // Inner level: vector lanes
        FOR_THIRD(k, 0, P, {
            // work
        });
    });
});
```

**Inclusive-range variants:** `DO_FIRST`, `DO_SECOND`, `DO_THIRD`

**Hierarchical reductions:**

```cpp
FOR_FIRST(i, 0, N, {
    double row_sum = 0.0;
    FOR_REDUCE_SUM_SECOND(j, 0, M, loc_sum, {
        loc_sum += A(i, j) * x(j);
    }, row_sum);
    // row_sum now holds the dot product for row i
});
```

Available hierarchical reductions:
- `FOR_REDUCE_SUM_SECOND`, `FOR_REDUCE_SUM_THIRD`
- `DO_REDUCE_SUM_SECOND`, `DO_REDUCE_SUM_THIRD`
- `FOR_REDUCE_MAX_SECOND`, `DO_REDUCE_MAX_THIRD`
- `FOR_REDUCE_MIN_SECOND`, `DO_REDUCE_MIN_THIRD`

Inside hierarchical blocks, `TEAM_ID` and `THREAD_ID` give the league rank and team rank respectively.

---

## 8. Class-Member Lambda Macros

When using MATAR macros inside class member functions, use the `_CLASS` variants. These use `KOKKOS_CLASS_LAMBDA` (which captures `*this` by value) instead of `KOKKOS_LAMBDA`.

```cpp
class MySimulation {
    CArrayDevice<double> data_;

    void update() {
        FOR_ALL_CLASS(i, 0, N, {
            data_(i) *= 2.0;
        });
    }

    double total() {
        double loc = 0.0, result = 0.0;
        FOR_REDUCE_SUM_CLASS(i, 0, N, loc, {
            loc += data_(i);
        }, result);
        return result;
    }
};
```

Available `_CLASS` variants:
- `FOR_ALL_CLASS`
- `FOR_REDUCE_SUM_CLASS`
- `FOR_REDUCE_MAX_CLASS`
- `FOR_REDUCE_MIN_CLASS`
- `RUN_CLASS`

---

## 9. Host/Device Data Transfer (Dual Types)

Dual types (`CArrayDual`, `FMatrixDual`, `ViewCArrayDual`, etc.) maintain separate host and device copies. Explicit synchronization is required.

### Pattern: Initialize on Host, Compute on Device, Read on Host

```cpp
CArrayDual<double> data(N, M);

// 1. Fill on host
for (int i = 0; i < N; i++)
    for (int j = 0; j < M; j++)
        data.host(i, j) = initial_value(i, j);

// 2. Push to device
data.update_device();

// 3. Compute on device
FOR_ALL(i, 0, N,
        j, 0, M, {
    data(i, j) = transform(data(i, j));  // device accessor: operator()
});
MATAR_FENCE();

// 4. Pull back to host
data.update_host();

// 5. Read on host
for (int i = 0; i < N; i++)
    printf("data(%d, 0) = %f\n", i, data.host(i, 0));
```

**Key rules:**
- Inside `FOR_ALL`/`RUN` blocks: use `data(i, j)` — this is the device accessor.
- Outside parallel regions on the CPU: use `data.host(i, j)`.
- Call `update_device()` after host writes, before device reads.
- Call `update_host()` after device writes, before host reads.
- `set_values(val)` operates on the device copy.

---

## 10. Conversion Rules: Standard C++ to MATAR

### Step 1: Replace Data Structures

| Original C++ | MATAR Replacement | When to Use |
|---|---|---|
| `double a[N]` | `CArrayDevice<double> a(N)` | Device-only computation |
| `double a[N][M]` | `CArrayDevice<double> a(N, M)` | Device-only computation |
| `double a[N][M][P]` | `CArrayDevice<double> a(N, M, P)` | Device-only computation |
| `double a[N]` | `CArrayDual<double> a(N)` | Need host and device access |
| `double* a = new double[N]` | `CArrayDevice<double> a(N)` | Dynamic allocation → MATAR |
| `std::vector<double> a(N)` | `CArrayDual<double> a(N)` | Need both host and device |
| `double a[N]` (used as function arg) | `ViewCArrayDevice<double> a(ptr, N)` | Wrap existing memory |
| Fortran-style 1-based `A(i,j)` | `FMatrixDevice<double> A(N, M)` | Fortran interop |

### Step 2: Replace Index Access

| Original | MATAR |
|----------|-------|
| `a[i]` | `a(i)` |
| `a[i][j]` | `a(i, j)` |
| `a[i][j][k]` | `a(i, j, k)` |
| Pointer arithmetic `*(a + i*M + j)` | `a(i, j)` |

### Step 3: Replace Loops with Macros

| Original | MATAR | Notes |
|----------|-------|-------|
| `for (int i = 0; i < N; i++) { ... }` | `FOR_ALL(i, 0, N, { ... });` | Half-open range |
| `for (int i = 0; i < N; i++) for (int j = 0; j < M; j++) { ... }` | `FOR_ALL(i, 0, N, j, 0, M, { ... });` | 2D parallel |
| `for (int i = 1; i <= N; i++) { ... }` | `DO_ALL(i, 1, N, { ... });` | Inclusive range |
| Serial `for` with accumulator `sum += ...` | `FOR_REDUCE_SUM(i, 0, N, loc, { loc += ...; }, sum);` | Parallel reduction |
| Serial `for` tracking max | `FOR_REDUCE_MAX(i, 0, N, loc, { if(x > loc) loc = x; }, max_val);` | Max reduction |
| Serial `for` tracking min | `FOR_REDUCE_MIN(i, 0, N, loc, { if(x < loc) loc = x; }, min_val);` | Min reduction |

### Step 4: Handle Dependencies and Synchronization

- Place `MATAR_FENCE()` after any parallel kernel whose output is needed by subsequent code.
- For Dual types: call `update_host()` before host reads, `update_device()` before device reads.
- Within `FOR_ALL`, any inner serial `for` loop is fine — only the outer dimensions are parallelized.

### Step 5: Handle Race Conditions

When multiple parallel iterations write to the same output location:

```cpp
// WRONG: race condition — multiple (i,j,k) combos update the same C(i,j)
FOR_ALL(i, 0, N, j, 0, N, k, 0, N, {
    C(i, j) += A(i, k) * B(k, j);
});

// FIX OPTION 1: Atomics
FOR_ALL(i, 0, N, j, 0, N, k, 0, N, {
    Kokkos::atomic_add(&C(i, j), A(i, k) * B(k, j));
});

// FIX OPTION 2: Inner serial loop (preferred for matmul)
FOR_ALL(i, 0, N, j, 0, N, {
    double local_sum = 0.0;
    for (int k = 0; k < N; k++) {
        local_sum += A(i, k) * B(k, j);
    }
    C(i, j) = local_sum;
});
```

---

## 11. Complete Before/After Examples

### Example A: 2D Jacobi Heat Equation

**Before (standard C++):**

```cpp
double temperature[height + 2][width + 2];
double temperature_previous[height + 2][width + 2];

// Initialize
for (i = 0; i <= height + 1; i++)
    for (j = 0; j <= width + 1; j++)
        temperature_previous[i][j] = 0.0;

// Jacobi iteration
while (worst_dt > tolerance) {
    for (i = 1; i < height + 1; i++) {
        for (j = 1; j < width + 1; j++) {
            temperature[i][j] = 0.25 * (temperature_previous[i+1][j]
                                       + temperature_previous[i-1][j]
                                       + temperature_previous[i][j+1]
                                       + temperature_previous[i][j-1]);
        }
    }

    worst_dt = 0.0;
    for (i = 1; i < height + 1; i++) {
        for (j = 1; j < width + 1; j++) {
            worst_dt = fmax(fabs(temperature[i][j] - temperature_previous[i][j]), worst_dt);
            temperature_previous[i][j] = temperature[i][j];
        }
    }
}
```

**After (MATAR with GPU support):**

```cpp
MATAR_INITIALIZE(argc, argv);
{
    CArrayDual<double> temperature(height + 2, width + 2);
    CArrayDual<double> temperature_previous(height + 2, width + 2);

    // Initialize
    temperature_previous.set_values(0.0);
    FOR_ALL(i, 0, height + 2, {
        temperature_previous(i, width + 1) = (1000.0 / height) * i;
    });
    temperature_previous.update_host();

    while (worst_dt > tolerance) {
        // Parallel Jacobi stencil
        FOR_ALL(i, 1, height + 1,
                j, 1, width + 1, {
            temperature(i, j) = 0.25 * (temperature_previous(i+1, j)
                                       + temperature_previous(i-1, j)
                                       + temperature_previous(i, j+1)
                                       + temperature_previous(i, j-1));
        });
        MATAR_FENCE();

        // Parallel max-reduction + copy-back
        double local_max = 0.0;
        double max_value = 0.0;
        FOR_REDUCE_MAX(i, 1, height + 1,
                       j, 1, width + 1,
                       local_max, {
            double val = fabs(temperature(i, j) - temperature_previous(i, j));
            if (val > local_max) local_max = val;
            temperature_previous(i, j) = temperature(i, j);
        }, max_value);

        worst_dt = max_value;

        // For host-side I/O: sync back
        if (iteration % 1000 == 0) {
            temperature.update_host();
            print_heatmap(temperature);  // uses temperature.host(i, j)
        }
    }
}
MATAR_FINALIZE();
```

### Example B: Array Initialization and Sum

**Before:**

```cpp
int A[10];
for (int i = 0; i < 10; i++) A[i] = 314;

int sum = 0;
for (int i = 0; i < 10; i++) sum += A[i] * A[i];
```

**After:**

```cpp
CArrayDevice<int> A(10);
FOR_ALL(i, 0, 10, {
    A(i) = 314;
});

int loc_sum = 0, sum = 0;
FOR_REDUCE_SUM(i, 0, 10, loc_sum, {
    loc_sum += A(i) * A(i);
}, sum);
```

### Example C: Matrix Multiply

**Before:**

```cpp
for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
        C[i][j] = 0.0;
        for (int k = 0; k < N; k++)
            C[i][j] += A[i][k] * B[k][j];
    }
```

**After:**

```cpp
CArrayDevice<double> A(N, N), B(N, N), C(N, N);
// ... initialize A, B ...

FOR_ALL(i, 0, N, j, 0, N, {
    double local_sum = 0.0;
    for (int k = 0; k < N; k++) {
        local_sum += A(i, k) * B(k, j);
    }
    C(i, j) = local_sum;
});
MATAR_FENCE();
```

### Example D: Wrapping Existing Memory

**Before:**

```cpp
int some_array[9];
for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
        some_array[i * 3 + j] = i + j;
```

**After:**

```cpp
int some_array[9];
for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
        some_array[i * 3 + j] = i + j;

// Wrap as 2D dual view — no copy
ViewCArrayDual<int> view(&some_array[0], 3, 3);

// Now usable in parallel regions on device
FOR_ALL(i, 0, 3, j, 0, 3, {
    view(i, j) *= 2;
});
MATAR_FENCE();
```

### Example E: Virtual Functions on Device

```cpp
// Allocate raw device memory for polymorphic objects
auto shapes = Kokkos::kokkos_malloc<DefaultMemSpace>(num_shapes * sizeof(Shape*));

// Placement new inside device kernel
FOR_ALL(i, 0, num_shapes, {
    if (i % 2 == 0)
        new (&shapes[i]) Circle(radius);
    else
        new (&shapes[i]) Square(side);
});
MATAR_FENCE();

// Call virtual functions in parallel
FOR_ALL(i, 0, num_shapes, {
    double a = shapes[i]->area();
});
```

---

## 12. Common Pitfalls

### 1. Wrong Loop Order for Memory Layout

```cpp
// BAD: inner loop on i with CArray (row-major) — poor cache behavior
FOR_ALL(j, 0, M, i, 0, N, {
    carr(i, j) = ...;  // i changes in inner loop but is the slow index in CArray
});

// GOOD: inner loop on j matches CArray row-major layout
FOR_ALL(i, 0, N, j, 0, M, {
    carr(i, j) = ...;  // j changes in inner loop = fast index = good cache use
});
```

### 2. Missing MATAR_FENCE()

```cpp
FOR_ALL(i, 0, N, { a(i) = compute(i); });
// BAD: immediately reading a(i) on host without fence
double val = a.host(0);  // may not reflect device computation

// GOOD:
FOR_ALL(i, 0, N, { a(i) = compute(i); });
MATAR_FENCE();
a.update_host();
double val = a.host(0);  // correct
```

### 3. Forgetting update_host / update_device

```cpp
CArrayDual<double> data(N);
for (int i = 0; i < N; i++) data.host(i) = i;  // writes to host copy

// BAD: device copy is stale
FOR_ALL(i, 0, N, { data(i) *= 2.0; });  // reads stale device data

// GOOD:
data.update_device();  // sync host → device
FOR_ALL(i, 0, N, { data(i) *= 2.0; });  // reads fresh device data
```

### 4. Race Conditions in Reductions

```cpp
// BAD: manual sum in FOR_ALL has a race condition
double sum = 0.0;
FOR_ALL(i, 0, N, { sum += a(i); });  // race: multiple threads write to sum

// GOOD: use a reduction macro
double loc_sum = 0.0, sum = 0.0;
FOR_REDUCE_SUM(i, 0, N, loc_sum, { loc_sum += a(i); }, sum);
```

### 5. Using [] Instead of ()

MATAR uses `()` for all indexing. Using `[]` will not compile or will access wrong memory.

```cpp
// BAD
a[i][j] = 5;

// GOOD
a(i, j) = 5;
```

### 6. Scope and Lifetime

All MATAR objects must be destroyed before `MATAR_FINALIZE()`. Use scoping braces:

```cpp
MATAR_INITIALIZE();
{
    CArrayDevice<double> a(100);  // created inside scope
    // ... use a ...
}  // a destroyed here, before finalize
MATAR_FINALIZE();
```

### 7. `set_values` on Dual Types

`set_values()` operates on the **device** copy. If you need the values on the host, call `update_host()` afterward:

```cpp
CArrayDual<double> data(N);
data.set_values(0.0);      // sets device copy to 0.0
data.update_host();         // now host copy also has 0.0
```

### 8. Using printf Inside Device Kernels

Inside `FOR_ALL`, `RUN`, and other device macros, use `printf` — not `std::cout`. The `<<` stream operators are not available in device code.

```cpp
// BAD: std::cout inside device kernel
FOR_ALL(i, 0, N, {
    std::cout << a(i) << std::endl;  // will not compile for GPU
});

// GOOD: printf inside device kernel
FOR_ALL(i, 0, N, {
    printf("a(%d) = %f\n", i, a(i));
});
```

### 9. Converting std::vector of vectors to Ragged Arrays

```cpp
// BAD: host-only, non-portable, poor data locality
std::vector<std::vector<int>> neighbors(num_nodes);
for (int i = 0; i < num_nodes; i++) {
    for (auto n : get_neighbors(i))
        neighbors[i].push_back(n);
}

// GOOD: MATAR ragged array — contiguous, device-portable
CArrayDevice<size_t> strides(num_nodes);
// ... fill strides with row lengths ...
RaggedCArrayDevice<int> neighbors(strides, "neighbors");
FOR_ALL(i, 0, num_nodes, {
    for (int j = 0; j < neighbors.stride(i); j++) {
        neighbors(i, j) = compute_neighbor(i, j);
    }
});
```

---

## 13. Type Alias Quick Reference

### Global Type Aliases

```cpp
using real_t = double;
using u_int  = unsigned int;
```

### Choosing the Right Type

Use this decision tree:

1. **Do you need host and device access?**
   - Yes → Use a **Dual** type
   - No, device only → Use a **Device** type
   - No, host only → Use a **Host** type

2. **Do you own the memory?**
   - Yes → Use an owning type (no `View` prefix)
   - No, wrapping existing pointer → Use a **View** type

3. **What index convention?**
   - 0-based `[0, N)` → Use an **Array** type
   - 1-based `[1, N]` → Use a **Matrix** type

4. **What memory layout?**
   - Row-major (C-style, last index fastest) → Use a **C** prefix
   - Column-major (Fortran-style, first index fastest) → Use an **F** prefix

**Example decision:** "I need a 2D row-major 0-based array that lives on both host and device"
→ `CArrayDual<double> a(N, M);`

**Example decision:** "I need to wrap a Fortran column-major 1-based array for device use"
→ `ViewFMatrixDual<double> view(fortran_ptr, N, M);`

---

## 14. Device Kernel Constraints

Code inside `FOR_ALL`, `DO_ALL`, `RUN`, and reduction macros runs on the device (GPU when built for CUDA/HIP). The following restrictions apply:

### What You Cannot Do Inside Device Kernels

| Forbidden | Why | Alternative |
|-----------|-----|-------------|
| `std::cout`, `std::cerr` | Not available on GPU | Use `printf` |
| `new` / `delete` / `malloc` / `free` | Host memory allocators | Use `Kokkos::kokkos_malloc` / placement `new` |
| `std::vector`, `std::map`, etc. | STL containers are host-only | Use MATAR arrays |
| `throw` / `try` / `catch` | Exceptions not supported on GPU | Use error codes or assertions |
| File I/O (`fopen`, `fstream`) | No filesystem on GPU | Do I/O on host, sync with `update_host()` |
| Virtual function calls | Only work via placement `new` on device memory | See Example E in Section 11 |
| `std::string` | Host-only | Use `const char*` or integer codes |
| Recursive function calls | Limited/unsupported on some GPU architectures | Rewrite iteratively |

### Lambda Capture Rules

`FOR_ALL` expands to `KOKKOS_LAMBDA`, which captures local variables **by value** (by copy). This means:

```cpp
int N = 100;
CArrayDevice<double> a(N);

FOR_ALL(i, 0, N, {
    a(i) = i * 2.0;  // N and a are captured by value — works correctly
});
```

You **cannot** capture:
- Host-only objects (`std::vector`, `std::string`, etc.)
- References to local variables that won't exist on the device
- Large stack objects (copies are expensive)

For class member functions, use `FOR_ALL_CLASS` (captures `*this` by value via `KOKKOS_CLASS_LAMBDA`).

### Raw Device Memory Allocation

For advanced use cases requiring raw device memory (e.g., polymorphism):

```cpp
// Allocate on device
void* ptr = Kokkos::kokkos_malloc<DefaultMemSpace>(num_bytes);

// Free on device
Kokkos::kokkos_free(ptr);
```

---

## 15. Fortran Interoperability

MATAR supports calling device-parallel C++ code from Fortran.

### C++ Side

```cpp
extern "C" void square_array_elements_(double* array, int* nx, int* ny) {
    int nx_ = *nx;
    int ny_ = *ny;
    auto a = ViewFMatrixDual<double>(array, nx_, ny_);

    DO_ALL(j, 1, ny_,
           i, 1, nx_, {
        a(i, j) = pow(a(i, j), 2);
    });
    a.update_host();
}

extern "C" void sum_array_elements_(double* array, int* nx, int* ny, double* result) {
    int nx_ = *nx;
    int ny_ = *ny;
    auto a = ViewFMatrixDual<double>(array, nx_, ny_);

    double loc_sum = 0.0, global_sum = 0.0;
    DO_REDUCE_SUM(j, 1, ny_,
                  i, 1, nx_,
                  loc_sum, {
        loc_sum += a(i, j);
    }, global_sum);
    *result = global_sum;
}

extern "C" void matar_initialize_() { MATAR_INITIALIZE(); }
extern "C" void matar_finalize_()   { MATAR_FINALIZE(); }
```

### Fortran Side

```fortran
program main
    use iso_c_binding
    implicit none
    interface
        subroutine matar_initialize() bind(C, name="matar_initialize_")
        end subroutine
        subroutine matar_finalize() bind(C, name="matar_finalize_")
        end subroutine
        subroutine square_array_elements(array, nx, ny) bind(C, name="square_array_elements_")
            use iso_c_binding
            real(c_double), intent(inout) :: array(*)
            integer(c_int), intent(in) :: nx, ny
        end subroutine
    end interface

    real(c_double), allocatable :: A(:,:)
    integer(c_int) :: nx, ny
    nx = 10; ny = 10
    allocate(A(nx, ny))
    ! ... fill A ...
    call matar_initialize()
    call square_array_elements(A, nx, ny)
    call matar_finalize()
end program
```

Key rules:
- Use `ViewFMatrixDual` to wrap Fortran arrays (column-major, 1-based).
- Use `DO_ALL` with 1-based ranges for natural Fortran indexing.
- C++ functions callable from Fortran must use `extern "C"` and trailing `_` in the name.
- Call `update_host()` after device work so Fortran sees the results.

---

## 16. Build Configuration

### CMake Integration

```cmake
find_package(Matar REQUIRED)
target_link_libraries(my_target matar)
```

### Preprocessor Defines

| Define | Effect |
|--------|--------|
| `HAVE_KOKKOS` | Enables Kokkos-backed device/dual types and parallel macros |
| `HAVE_CUDA` | Kokkos execution space = CUDA; layout defaults to `LayoutLeft` |
| `HAVE_HIP` | Kokkos execution space = HIP |
| `HAVE_OPENMP` | Kokkos execution space = OpenMP; layout = `LayoutRight` |
| `HAVE_THREADS` | Kokkos execution space = Threads |
| `HAVE_MPI` | Enables MPI types (`MPICArrayKokkos`, `PartitionMap`, etc.) |
| `TRILINOS_INTERFACE` | Enables Tpetra wrapper types |

When no GPU backend is specified with Kokkos, the default execution space and layout from Kokkos are used.

### Running with Backend Options

```bash
# Serial
./my_program

# OpenMP
export OMP_NUM_THREADS=8
./my_program

# Kokkos threads
./my_program --kokkos-threads=8

# CUDA (built with HAVE_CUDA)
./my_program
```
