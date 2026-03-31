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
17. [MPI and Distributed Communication](#17-mpi-and-distributed-communication)
18. [Ground Truth Constraints for LLMs](#18-ground-truth-constraints-for-llms)
19. [LLM Output Contract](#19-llm-output-contract)

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

`MATAR_FENCE()` is a synchronization barrier. It ensures all device operations have completed before the CPU proceeds. See the [fence placement rules](#fence-placement-rules) below for when fences are required vs. redundant.

### Fence Placement Rules

A fence is **needed** between two `FOR_ALL` blocks only if the second reads data written by the first. Over-fencing kills performance; under-fencing causes correctness bugs. Apply these rules:

| Situation | Fence Needed? | Why |
|-----------|:---:|-----|
| `FOR_ALL` writes to `A`, next `FOR_ALL` reads `A` | **Yes** | Data dependency between kernels |
| `FOR_ALL` writes to `A`, next `FOR_ALL` writes/reads only `B` | **No** | Independent data, no dependency |
| `FOR_ALL` writes to `A`, then host code reads `A.host(i)` | **Yes** | Host reads device-written data |
| `FOR_REDUCE_SUM(..., total)`, then host uses `total` | **No** | Reduction macros implicitly fence on the result variable |
| `FOR_ALL` followed by `update_host()` | **Yes** | Must ensure kernel completes before sync |
| `FOR_ALL` followed by timing (`std::chrono`) | **Yes** | Timer must see completed work |
| Two `FOR_ALL` blocks, both only read (no writes) | **No** | Read-only kernels have no conflict |

```cpp
// NO fence needed: independent data
FOR_ALL(i, 0, N, { A(i) = i; });
FOR_ALL(i, 0, N, { B(i) = i * 2; });  // B is independent of A

// FENCE needed: B reads A
FOR_ALL(i, 0, N, { A(i) = i; });
MATAR_FENCE();
FOR_ALL(i, 0, N, { B(i) = A(i) * 2; });

// NO fence needed after reduction: result is available immediately
double loc = 0.0, total = 0.0;
FOR_REDUCE_SUM(i, 0, N, loc, { loc += A(i); }, total);
printf("total = %f\n", total);  // total is valid here without MATAR_FENCE()
```

---

## 2. Data Structure Taxonomy

MATAR data structures are organized along four axes:

### Axis 1: Memory Layout

| Prefix | Layout | Index Convention | Best Loop Order |
|--------|--------|-----------------|-----------------|
| **C** | Row-major (C-style) | Last index varies fastest in memory | Outermost loop = first index |
| **F** | Column-major (Fortran-style) | First index varies fastest in memory | Outermost loop = last index |

**GPU coalescing rule:** CUDA and HIP achieve coalesced memory access when adjacent threads access adjacent memory addresses. In `FOR_ALL(i, 0, N, j, 0, M, {...})`, the **last** index (`j`) is mapped to thread IDs. Therefore:

- **`CArrayDevice` (last index fastest):** `FOR_ALL(i, ..., j, ..., { a(i,j) })` gives coalesced access because adjacent threads (adjacent `j`) access adjacent memory locations.
- **`FArrayDevice` (first index fastest):** The same `FOR_ALL` ordering does **not** give coalesced access because adjacent threads (adjacent `j`) hit non-adjacent memory.

**Optimization rule for GPU targets:** Prefer `CArrayDevice` with loop nests ordered so the innermost `FOR_ALL` index corresponds to the last array dimension. When correctness of an initial translation from Fortran is the priority and the access pattern is complex, use `FArrayDevice` to preserve Fortran semantics and optimize later. The `FArray` types give a correct first-pass translation from Fortran (preserving the original memory layout and loop ordering), but they may not give optimal GPU performance without reordering.

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

### Views as Array Slices

View types can wrap a pointer into the **middle** of an existing array, reinterpreting a subregion as a new array with different dimensionality. This is the MATAR equivalent of Fortran's ability to pass `A(1,5)` to a subroutine that treats it as a 1D array starting from that element.

```cpp
// 3D array: B(num_blocks, rows, cols)
CArrayDual<double> B(num_blocks, rows, cols);

// Create a 2D view into block 1: starts at &B(1, 0, 0), shape is (rows, cols)
ViewCArray<double> block1(B.host_pointer() + 1 * rows * cols, rows, cols);

// block1(i, j) now aliases B(1, i, j) — no data copy
for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
        block1(i, j) = some_value;

// Device-side slicing for passing subarrays to functions
ViewCArrayDevice<double> dev_slice(A.pointer() + offset, slice_len);
FOR_ALL(i, 0, slice_len, {
    dev_slice(i) *= 2.0;
});
```

This pattern naturally handles legacy codes that pass array slices to subroutines. If the original code does `call sub(A(1,5), N)` in Fortran, MATAR typically wraps `&A(1,5)` as a **1D `ViewFArray`** (or `ViewFArrayDual`) of length `N`. Use `ViewFMatrix` only when the callee expects a 2D shape/leading dimension interpretation.

### Common Methods

| Method | Description |
|--------|-------------|
| `size()` | Total number of elements |
| `extent()` | Same as `size()` |
| `dims(n)` | Size along dimension `n` (0-indexed for Arrays, 1-indexed for Matrices) |
| `order()` | Number of dimensions |
| `pointer()` | Raw pointer to underlying data. Available on Device and Host types only. **Dual types do NOT have `pointer()`** — use `device_pointer()` or `host_pointer()` instead. |
| `device_pointer()` | Raw pointer to device-side memory (Dual types only) |
| `host_pointer()` | Raw pointer to host-side memory (Dual types only) |
| `set_values(val)` | Set all elements to `val` via `parallel_for` on the device. **Does NOT fence** — follow with `MATAR_FENCE()` if host code depends on the result, or `update_host()` if you need the values on the host. |
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
| `FOR_REDUCE_PRODUCT` | `(i, lo, hi, var, {body}, result)` (6 args) | `(i, lo, hi, j, lo, hi, var, {body}, result)` (9 args) | `(i, lo, hi, j, lo, hi, k, lo, hi, var, {body}, result)` (12 args) |

The `DO_REDUCE_*` variants follow the same argument patterns as `FOR_REDUCE_*` but with inclusive upper bounds **for SUM/MAX/MIN**. There is no `DO_REDUCE_PRODUCT` macro in the current `macros.h`.

### `FOR_ALL` — C-style half-open range `[start, end)`

The primary parallel loop macro. Indices iterate over `[start, end)`. Supports 1D, 2D, and 3D.

**All index dimensions in a `FOR_ALL` are parallelized.** There is no "outer parallel, inner sequential" variant in the flat `FOR_ALL` form. A 2D `FOR_ALL(i, 0, N, j, 0, M, {...})` parallelizes over the entire `N * M` index space — both `i` and `j` are distributed across threads simultaneously via `Kokkos::MDRangePolicy`.

```cpp
// COMPILES_AS_IS (assuming variables are declared)
// 1D: 4 arguments — parallelizes over N iterations
FOR_ALL(i, 0, N, {
    a(i) = i;
});

// 2D: 7 arguments — parallelizes over N*M iterations (both i and j)
FOR_ALL(i, 0, N,
        j, 0, M, {
    a(i, j) = i + j;
});

// 3D: 10 arguments — parallelizes over N*M*P iterations (all of i, j, k)
FOR_ALL(i, 0, N,
        j, 0, M,
        k, 0, P, {
    a(i, j, k) = i + j + k;
});
```

**Mixing parallel and serial:** When the inner loop has a data dependency (e.g., accumulating a sum along one dimension), use `FOR_ALL` for the parallel dimensions and a plain `for` loop inside the body for the serial dimension:

```cpp
// COMPILES_AS_IS (assuming variables are declared)
// Outer i,j are parallel; inner k is serial (accumulates into local_sum)
FOR_ALL(i, 0, N, j, 0, M, {
    double local_sum = 0.0;
    for (int k = 0; k < P; k++) {
        local_sum += A(i, k) * B(k, j);
    }
    C(i, j) = local_sum;
});
```

For finer-grained control over nested parallelism (e.g., tiled algorithms), use the hierarchical `FOR_FIRST`/`FOR_SECOND`/`FOR_THIRD` macros described in Section 7.

**Loop ordering:** The inner loop varies the fastest and the outer loop varies the slowest. For optimal cache performance, match the loop index order to the data type's memory layout:
- `CArray` / `CArrayDevice`: last index fastest → `FOR_ALL(i, ..., j, ..., { a(i,j) })` is correct
- `FArray` / `FArrayDevice`: first index fastest → `FOR_ALL(i, ..., j, ..., { a(j,i) })` maps the fast-varying loop index to the fast-varying array index

### `DO_ALL` — Fortran-style inclusive range `[start, end]`

Same parallelization semantics as `FOR_ALL` (all dimensions parallelized), but the upper bound is **inclusive**. Designed for 1-based Fortran-style indexing with `FMatrix` types.

**Underlying macro difference:** `FOR_ALL` passes ranges directly to `Kokkos::RangePolicy(x0, x1)` or `MDRangePolicy({x0,y0}, {x1,y1})`, making the upper bound exclusive. `DO_ALL` adds `+1` to every upper bound before passing to Kokkos: `RangePolicy(x0, x1+1)` or `MDRangePolicy({x0,y0}, {x1+1,y1+1})`. Both ultimately expand to `Kokkos::parallel_for` with `KOKKOS_LAMBDA`. In the current `macros.h`, `LOOP_ORDER` and `F_LOOP_ORDER` are both `Kokkos::Iterate::Right`, so iteration order is effectively the same unless those macros are changed.

```cpp
// COMPILES_AS_IS (assuming variables are declared)
// 1D: iterates i = 1, 2, ..., N (inclusive)
DO_ALL(i, 1, N, {
    matrix(i) = i;
});

// 2D: iterates i = 1..N, j = 1..M (both inclusive, both parallel)
DO_ALL(j, 1, M,
       i, 1, N, {
    matrix(i, j) = i * j;
});
```

**When to use which:**
- `FOR_ALL` with `CArray`/`CArrayDevice`: C-style code, 0-based indexing, `[0, N)`
- `DO_ALL` with `FMatrix`/`FMatrixDevice`: Fortran-style code, 1-based indexing, `[1, N]`
- The same `FOR_ALL` vs `DO_ALL` distinction applies to all reduction variants (`FOR_REDUCE_*` vs `DO_REDUCE_*`)

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

Serial (non-parallel) loops with lambda-based syntax. `FOR_LOOP` uses half-open ranges; `DO_LOOP` uses inclusive ranges. Both support 1D, 2D, and 3D forms, plus **stride (delta) variants** for loops with a custom increment.

```cpp
// 1D — basic
FOR_LOOP(i, 0, N, {
    // serial iteration i = 0 .. N-1
});

DO_LOOP(i, 1, N, {
    // serial iteration i = 1 .. N
});

// 1D — with stride/delta (5 arguments before body)
FOR_LOOP(i, 0, N, 2, {
    // serial iteration i = 0, 2, 4, ... (stride of 2, half-open)
});

DO_LOOP(i, 1, N, 3, {
    // serial iteration i = 1, 4, 7, ... (stride of 3, inclusive)
});

// 2D — with strides (9 arguments before body)
FOR_LOOP(i, 0, N, i_delta, j, 0, M, j_delta, {
    // serial 2D loop with custom strides on both dimensions
});

// 3D — with strides (13 arguments before body)
FOR_LOOP(i, 0, N, i_delta, j, 0, M, j_delta, k, 0, P, k_delta, {
    // serial 3D loop with custom strides on all dimensions
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

SUM, MAX, and MIN reductions have `DO_REDUCE_*` counterparts with inclusive upper bounds. **There is no `DO_REDUCE_PRODUCT`.**

```cpp
DO_REDUCE_SUM(i, 1, N, loc_sum, { loc_sum += a(i); }, total);
DO_REDUCE_MAX(i, 1, N, loc_max, { ... }, global_max);
DO_REDUCE_MIN(i, 1, N, loc_min, { ... }, global_min);
// DO_REDUCE_PRODUCT does NOT exist — use FOR_REDUCE_PRODUCT with half-open range instead
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
- `set_values(val)` launches an **async** `parallel_for` on the device copy — it does NOT fence. Follow with `update_host()` to get the values on the host.

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

**Before (standard C++):** `PSEUDOCODE_PATTERN`

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

**After (MATAR with GPU support):** `PSEUDOCODE_PATTERN`

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

**Before:** `PSEUDOCODE_PATTERN`

```cpp
int A[10];
for (int i = 0; i < 10; i++) A[i] = 314;

int sum = 0;
for (int i = 0; i < 10; i++) sum += A[i] * A[i];
```

**After:** `COMPILES_AS_IS` (inside `MATAR_INITIALIZE`/`MATAR_FINALIZE` block)

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

**Before:** `PSEUDOCODE_PATTERN`

```cpp
for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
        C[i][j] = 0.0;
        for (int k = 0; k < N; k++)
            C[i][j] += A[i][k] * B[k][j];
    }
```

**After:** `PSEUDOCODE_PATTERN`

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

**Before:** `PSEUDOCODE_PATTERN`

```cpp
int some_array[9];
for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
        some_array[i * 3 + j] = i + j;
```

**After:** `PSEUDOCODE_PATTERN`

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

### 2. Missing or Excessive MATAR_FENCE()

```cpp
// BAD: immediately reading on host without fence
FOR_ALL(i, 0, N, { a(i) = compute(i); });
double val = a.host(0);  // may not reflect device computation

// GOOD: fence then sync to host
FOR_ALL(i, 0, N, { a(i) = compute(i); });
MATAR_FENCE();
a.update_host();
double val = a.host(0);  // correct

// BAD: unnecessary fence between independent kernels (kills performance)
FOR_ALL(i, 0, N, { A(i) = i; });
MATAR_FENCE();  // wasteful — B doesn't depend on A
FOR_ALL(i, 0, N, { B(i) = i * 2; });

// GOOD: no fence needed for independent data
FOR_ALL(i, 0, N, { A(i) = i; });
FOR_ALL(i, 0, N, { B(i) = i * 2; });

// NOTE: reduction results are immediately available (implicit fence)
double loc = 0.0, total = 0.0;
FOR_REDUCE_SUM(i, 0, N, loc, { loc += a(i); }, total);
// total is valid here — no MATAR_FENCE() needed
```

See the [Fence Placement Rules](#fence-placement-rules) in Section 1 for the complete decision table.

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

`set_values()` launches a `Kokkos::parallel_for` on the **device** copy. It is **asynchronous** — it does NOT fence internally. Rules:

- If the next operation is another device kernel (`FOR_ALL`, `set_values`, etc.) on the **same** data, Kokkos execution-order guarantees serialization on the default stream — **no fence needed**.
- If you need to **read** the values on the **host**, call `update_host()` afterward (which internally fences the device copy).
- If you need a **device-side** guarantee before a kernel on **different** data that reads this data via a different view/pointer, insert `MATAR_FENCE()`.

```cpp
// PSEUDOCODE_PATTERN
CArrayDual<double> data(N);
data.set_values(0.0);      // launches async parallel_for on device
// No fence needed here — next FOR_ALL is on the same default execution space
FOR_ALL(i, 0, N, {
    data(i) += 1.0;        // reads data set by set_values — safe without fence
});
data.update_host();         // fences, then copies device → host
// Now data.host(i) has 1.0 for all i
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

---

## 17. MPI and Distributed Communication

MATAR provides MPI-aware data types and a communication plan abstraction for distributed-memory parallelism. These are enabled when built with `HAVE_MPI` and `HAVE_KOKKOS`.

### When to Use MPICArrayKokkos vs. Plain DCArrayKokkos

| Type | Use Case |
|------|----------|
| `DCArrayKokkos<T>` (`CArrayDual<T>`) | Single-rank data. No MPI communication needed. Host/device sync only. |
| `MPICArrayKokkos<T>` | Distributed data with ghost/halo regions. Wraps `DCArrayKokkos` internally and adds MPI send/recv buffer management, a `CommunicationPlan`, and a `communicate()` method that handles the full pack → exchange → unpack cycle. |

Use `MPICArrayKokkos` when a data array is partitioned across MPI ranks and neighboring ranks need to exchange boundary (ghost/halo) data. Use plain `DCArrayKokkos` for rank-local data that never crosses MPI boundaries.

### MPICArrayKokkos Overview

`MPICArrayKokkos<T>` is a template class that wraps a `DCArrayKokkos<T>` with:
- **Send and receive buffers** (`DCArrayKokkos<T>`) for packing/unpacking halo data
- A **stride** value computed from trailing dimensions (for multi-dimensional arrays, each first-index element is a contiguous block of `dim1 * dim2 * ... * dimN` values)
- A pointer to a shared **`CommunicationPlan`** that defines the neighbor topology
- A `host` member (`ViewCArray<T>`) for convenient host-side access

```cpp
// Construction: same as CArrayDual, but MPI-aware
MPICArrayKokkos<double> field(num_nodes, num_vars, "my_field");

// Host access via .host member (ViewCArray)
field.host(i, j) = value;

// Device access inside FOR_ALL
FOR_ALL(i, 0, num_nodes, j, 0, num_vars, {
    field(i, j) = compute(i, j);
});
```

### CommunicationPlan

The `CommunicationPlan` struct encapsulates the MPI neighbor topology and send/recv metadata. It uses an MPI distributed graph communicator for efficient sparse neighbor communication.

#### Construction and Setup (3 Steps)

```cpp
CommunicationPlan comm_plan;

// Step 1: Initialize with MPI communicator
comm_plan.initialize(MPI_COMM_WORLD);

// Step 2: Define neighbor topology
// Each rank specifies which ranks it sends to and receives from
int send_ranks[] = {rank - 1, rank + 1};  // neighbors
int recv_ranks[] = {rank - 1, rank + 1};
comm_plan.initialize_graph_communicator(
    2, send_ranks,   // num_send_ranks, send_rank_ids
    2, recv_ranks    // num_recv_ranks, recv_rank_ids
);

// Step 3: Define which data elements to send/recv per neighbor
// rank_send_ids: ragged array — row i lists element indices to send to neighbor i
// rank_recv_ids: ragged array — row i lists element indices to receive from neighbor i
DRaggedRightArrayKokkos<int> rank_send_ids(...);
DRaggedRightArrayKokkos<int> rank_recv_ids(...);
// ... fill send/recv index lists ...
comm_plan.setup_send_recv(rank_send_ids, rank_recv_ids);
```

#### CommunicationPlan Members

| Member | Type | Description |
|--------|------|-------------|
| `mpi_comm_world` | `MPI_Comm` | The base MPI communicator |
| `mpi_comm_graph` | `MPI_Comm` | Distributed graph communicator for neighbor collectives |
| `num_send_ranks` | `int` | Number of ranks this process sends to (out-degree) |
| `num_recv_ranks` | `int` | Number of ranks this process receives from (in-degree) |
| `send_rank_ids` | `DCArrayKokkos<int>` | Destination rank IDs |
| `recv_rank_ids` | `DCArrayKokkos<int>` | Source rank IDs |
| `send_indices_` | `DRaggedRightArrayKokkos<int>` | Per-neighbor indices of elements to send |
| `recv_indices_` | `DRaggedRightArrayKokkos<int>` | Per-neighbor indices of elements to receive |
| `send_counts_` / `recv_counts_` | `DCArrayKokkos<int>` | Number of elements per neighbor |
| `send_displs_` / `recv_displs_` | `DCArrayKokkos<int>` | Displacement offsets for packing |
| `total_send_count` / `total_recv_count` | `int` | Total elements across all neighbors |

### Connecting MPICArrayKokkos to a CommunicationPlan

```cpp
// Create the MPI-aware array
MPICArrayKokkos<double> field(num_owned + num_ghost, "field");

// Connect to the communication plan
field.initialize_comm_plan(comm_plan);
// This allocates send/recv buffers sized to total_send_count * stride
// and copies the per-neighbor counts/displacements (scaled by stride)
```

### The communicate() Cycle

Calling `field.communicate()` performs the full halo exchange:

1. **`fill_send_buffer()`**: Copies `this_array_` to host (`update_host`), then packs elements listed in `send_indices_` into the contiguous `send_buffer_` on the host. For multi-dimensional arrays, each element index packs `stride_` contiguous values.

2. **`MPI_Neighbor_alltoallv()`**: Exchanges data with all neighbors using the graph communicator. Send/recv counts and displacements are pre-computed and scaled by stride.

3. **`copy_recv_buffer()`**: Unpacks `recv_buffer_` into the ghost positions of `this_array_` using `recv_indices_`.

4. **`update_device()`**: Syncs the updated array (with fresh ghost data) back to the device.

```cpp
// COMPILES_AS_IS (assuming variables are declared)
// Typical time-step pattern:
FOR_ALL(i, 0, num_owned, {
    field(i) = compute_new_value(i);
});
MATAR_FENCE();

field.communicate();  // pack → MPI exchange → unpack → update_device

FOR_ALL(i, 0, num_owned + num_ghost, {
    // Now ghost values from neighbors are available
    result(i) = stencil(field, i);
});
```

### Interaction Between update_host/update_device and MPI

The `communicate()` method internally calls `update_host()` before packing and `update_device()` after unpacking. This means:
- Before calling `communicate()`, ensure device data is current (place a `MATAR_FENCE()` after any kernel that wrote to the array).
- After `communicate()` returns, the device copy already has fresh ghost data — no additional `update_device()` is needed.
- If you need to inspect the data on the host after communication, call `update_host()` explicitly.

### MPI Convenience Macros

| Macro | Expands To |
|-------|-----------|
| `MATAR_MPI_INIT` | `MPI_Init(&argc, &argv)` |
| `MATAR_MPI_FINALIZE` | `MPI_Finalize()` |
| `MATAR_MPI_TIME` | `MPI_Wtime()` |
| `MATAR_MPI_BARRIER` | `MPI_Barrier(MPI_COMM_WORLD)` |

### Complete MPI Example Pattern

```cpp
// PSEUDOCODE_PATTERN: fill in application-specific pieces
#include <mpi.h>
#include <matar.h>
#include <communication_plan.h>

using namespace mtr;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MATAR_INITIALIZE(argc, argv);
    {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // 1. Build communication plan
        CommunicationPlan comm_plan;
        comm_plan.initialize(MPI_COMM_WORLD);

        // Define neighbors (e.g., 1D domain decomposition)
        std::vector<int> send_ids, recv_ids;
        if (rank > 0)        { send_ids.push_back(rank - 1); recv_ids.push_back(rank - 1); }
        if (rank < size - 1) { send_ids.push_back(rank + 1); recv_ids.push_back(rank + 1); }

        comm_plan.initialize_graph_communicator(
            send_ids.size(), send_ids.data(),
            recv_ids.size(), recv_ids.data()
        );

        // Define which elements to send/recv (application-specific)
        // ... build rank_send_ids, rank_recv_ids as DRaggedRightArrayKokkos<int> ...
        comm_plan.setup_send_recv(rank_send_ids, rank_recv_ids);

        // 2. Create MPI-aware array and connect
        size_t num_owned = local_cells;
        size_t num_ghost = ghost_cells;
        MPICArrayKokkos<double> field(num_owned + num_ghost, "field");
        field.initialize_comm_plan(comm_plan);

        // 3. Compute + communicate loop
        for (int step = 0; step < num_steps; step++) {
            FOR_ALL(i, 0, num_owned, {
                field(i) = update(field, i);
            });
            MATAR_FENCE();

            field.communicate();  // halo exchange

            // Now ghost data is fresh on device
            FOR_ALL(i, 0, num_owned, {
                result(i) = stencil_with_ghosts(field, i);
            });
            MATAR_FENCE();
        }
    }
    MATAR_FINALIZE();
    MPI_Finalize();
    return 0;
}
```

---

## 18. Ground Truth Constraints for LLMs

Use these constraints as hard rules when generating MATAR code:

1. **Use only macros that exist in `MATAR/src/include/macros.h`.**
   - **Flat parallel loops:** `FOR_ALL`, `DO_ALL`, `RUN`
   - **Serial convenience loops:** `FOR_LOOP`, `DO_LOOP` (and stride variants `FOR_LOOP_DIM` / `DO_LOOP_DIM` for loops with a delta/increment)
   - **Reductions (half-open):** `FOR_REDUCE_SUM`, `FOR_REDUCE_MAX`, `FOR_REDUCE_MIN`, `FOR_REDUCE_PRODUCT`
   - **Reductions (inclusive):** `DO_REDUCE_SUM`, `DO_REDUCE_MAX`, `DO_REDUCE_MIN` — **`DO_REDUCE_PRODUCT` does NOT exist**
   - **Class-member variants** (use `KOKKOS_CLASS_LAMBDA`): `FOR_ALL_CLASS`, `RUN_CLASS`, `FOR_REDUCE_SUM_CLASS`, `FOR_REDUCE_MAX_CLASS`, `FOR_REDUCE_MIN_CLASS`
   - **Hierarchical (team/thread/vector):** `FOR_FIRST`, `FOR_SECOND`, `FOR_THIRD`, `DO_FIRST`, `DO_SECOND`, `DO_THIRD`
   - **Hierarchical reductions:** `FOR_REDUCE_SUM_SECOND`, `FOR_REDUCE_SUM_THIRD`, `DO_REDUCE_SUM_SECOND`, `DO_REDUCE_SUM_THIRD`, `FOR_REDUCE_MAX_SECOND`, `DO_REDUCE_MAX_THIRD`, `FOR_REDUCE_MIN_SECOND`, `DO_REDUCE_MIN_THIRD`

2. **`FOR_ALL` and `DO_ALL` parallelize all listed dimensions.**
   - If a dimension must remain sequential, keep it as a plain inner `for` loop inside the macro body, or use hierarchical macros where appropriate.

3. **Indexing is always `()` for MATAR arrays.**
   - Never emit `[]` indexing for MATAR types.

4. **For dual types, synchronize explicitly.**
   - Host writes require `update_device()` before device reads.
   - Device writes require `update_host()` before host reads.

5. **Fence only when needed.**
   - Required at dependency boundaries or before host/timing use.
   - Avoid unnecessary fences between independent kernels.

6. **Prefer aliases in `mtr::` namespace.**
   - Example: `CArrayDual<T>` instead of direct underlying class names unless low-level control is needed.

7. **MPI-aware distributed arrays use `MPICArrayKokkos`.**
   - Plain `DCArrayKokkos` / `CArrayDual` are rank-local and do not perform communication.

---

## 19. LLM Output Contract

When producing MATAR conversions, output must satisfy this checklist:

- **Boilerplate**
  - Include `#include <matar.h>`
  - Use `using namespace mtr;`
  - Wrap compute region between `MATAR_INITIALIZE(...)` and `MATAR_FINALIZE()`

- **Data structure mapping**
  - Choose `C*` vs `F*` based on target layout/coalescing goals
  - Choose `Array` vs `Matrix` based on 0-based vs 1-based indexing
  - Choose `Device` vs `Dual` vs `Host` based on execution/sync needs
  - Use `View*` when wrapping existing pointers or slices

- **Loop mapping**
  - Replace parallelizable loops with `FOR_ALL` / `DO_ALL`
  - Replace reductions with `FOR_REDUCE_*` / `DO_REDUCE_*`
  - Keep dependency-carrying inner loops serial inside macro bodies

- **Correctness and synchronization**
  - Add `update_device()` / `update_host()` where required
  - Add `MATAR_FENCE()` only at true synchronization boundaries
  - Avoid race conditions (use reductions, atomics, or serial inner loops)

- **MPI (if distributed)**
  - Build `CommunicationPlan`
  - Attach with `initialize_comm_plan(...)`
  - Use `communicate()` at halo exchange points

- **Output quality**
  - Label examples as either:
    - `COMPILES_AS_IS` (fully concrete except obvious variable declarations), or
    - `PSEUDOCODE_PATTERN` (contains app-specific placeholders)
