# MATAR
<p align="center"><img src="https://github.com/lanl/MATAR/blob/main/MATAR-logo.png" width="350">

MATAR (LANL open-source code number C23032) is a C++ library that addresses the need for simple, fast, and memory-efficient multi-dimensional data representations for dense and sparse storage that arise with numerical methods and in software applications. The data representations are designed to perform well across multiple computer architectures, including CPUs and GPUs. MATAR allows users to easily create and use intricate data representations that are also portable across disparate architectures using Kokkos. The performance aspect is achieved by forcing contiguous memory layout (or as close to contiguous as possible) for multi-dimensional and multi-size dense or sparse MATrix and ARray (hence, MATAR) types. Results show that MATAR has the capability to improve memory utilization, performance, and programmer productivity in scientific computing. This is achieved by fitting more work into the available memory, minimizing memory loads required, and by loading memory in the most efficient order. 


## Examples
* [ELEMENTS](https://github.com/lanl/ELEMENTS/):   MATAR is a part of the ELEMENTS Library (LANL C# C20058) and it underpins the routines implemented in ELEMENTS.  MATAR is available in a stand-alone directory outside of the ELEMENTS directory because it can aid many code applications.  The dense and sparse storage types in MATAR are the foundation for the ELEMENTS library, which contains mathematical functions to support a very broad range of element types including: linear, quadratic, and cubic serendipity elements in 2D and 3D; high-order spectral elements; and a linear 4D element. An unstructured high-order mesh class is available in ELEMENTS and it takes advantage of MATAR for efficient access of various mesh entities. 

* [Fierro](https://github.com/lanl/Fierro): The MATAR library underpins the Fierro code that is designed to simulate quasi-static solid mechanics problems and material dynamics problems.  
    
* Simple examples are in the /example folder

## Descriptions

* All Array MATAR types (e.g., CArray, ViewCArray, FArray, RaggedRightArray, etc.) start with an index of 0 and stop at an index of N-1, where N is the number of entries.  

* All Matrix MATAR types  (e.g., CMatrix, ViewCMatrix, FMatrix, etc.)  start with an index of 1 and stop at an index of N, where N is the number of entries. 

* The MATAR View types (e.g., ViewCArray, ViewCMatrix, ViewFArray, etc. ) are designed to accept a pointer to an existing 1D array and then access that 1D data as a multi-dimensional array.  The MATAR View types can also be used to slice an existing View.  

* The C dense storage and View types (e.g., CArray, ViewCArray, CMatrix, etc.) access the data following the C/C++ language convection of having the last index in a multi-dimensional array vary the quickest.  In a 2D CArray A, the index j in A(i,j) varies first followed by the index i, so the optimal performance is achieved using the following loop ordering.
```
// Optimal use of CArray
for (i=0,i<N,i++){
    for (j=0,j<N,j++){
        A(i,j) = 0.0;
    }
}
```

* The F dense storage and View types (e.g., FArray, ViewFArray, FMatrix, etc.) access the data following the Fortran language convection of having the first index in a multi-dimensional array vary the quickest.  In a 2D FMatrix M, the index i in M(i,j) varies first followed by the index j, so the optimal performance is achieved using the following loop ordering.

```
// Optimal use of FMatrix
for (j=1,j<=N,j++){
    for (i=1,i<=N,i++){
        M(i,j) = 0.0;
    }
}
```

* The ragged data types (e.g., RaggedRightArray, RaggedDownArray, etc) in MATAR are special sparse storage types.  The Right access types are for R(i,j) where the number of column entries varies in width across the array.  The Down access types are for D(i,j) where the number of row entries vary in length across the array.

* The SparseRowArray MATAR type is the idetical to the Compressed Sparse Row (CSR) or Compressed Row Storage (CSR) respresentation.

* The SparseColumnArray MATAR type is identical to the Compressed Sparse Column (CSC) or Compressed Column Storage (CCS) respresentation.


## Usage
```
// create a 1D array of integers and then access as a 2D array
int A[9];
auto A_array = ViewCArray <int> (A, 3, 3); // access as A(i,j) 

// create a 3D array of doubles
auto B = CArray <double> (3,3,3); // access as B(i,j,k)

// create a slice of the 3D array at index 1
auto C = ViewCArray <double> (&B(1,0,0),3,3); // access as C(j,k)


// create a 4D matrix of doubles, indices start at 1 
auto D = CMatrix <double> (10,9,8,7); // access as D(i,j,k,l)


// create a 2D view of a standard array
std::array<int, 9> E1d;
auto E = ViewCArray<int> (&E1d[0], 3, 3);
E(0,0) = 1;  // and so on


// create a ragged-right array of integers
//
// [1, 2, 3]
// [4, 5]
// [6]
// [7, 8, 9, 10]
//
size_t my_strides[4] = {3, 2, 1, 4};
RaggedRightArray <int> ragged(my_strides, 4);
    
int value = 1;
for (int i=0; i<4; i++){
    for (int j=0; j<my_ragged.stride(i); j++){
        ragged(i,j) = value;
        value++;
    }
}


```
More information about the capabilities and usage of MATAR can be found in this presentation [here](https://www.researchgate.net/publication/360744549_General_purpose_GPU_programming_made_easy).
    
## Cloning the code
If your SSH keys are set in github, then from the terminal type:
```
git clone --recursive ssh://git@github.com/lanl/MATAR.git    
```
The code can also be cloned using
```
git clone --recursive https://github.com/lanl/MATAR.git
```

## Building MATAR
MATAR is built entirely with CMake. Kokkos is bundled as a git submodule (`src/Kokkos/kokkos`), pinned to the Kokkos **5.2.1** release, and is built automatically with the backend you select, so no separate Kokkos install is needed. Kokkos 5 requires **C++20** and CMake >= 3.22 (CUDA builds additionally need CMake >= 3.25.2 for C++20 in the CUDA language), so `matar::matar` requests `cxx_std_20`.

The provided CMake presets configure, build, and test MATAR with a given Kokkos backend:
```
cmake --preset serial          # also: openmp, pthreads, cuda, hip,
cmake --build --preset serial  #       serial-mpi, openmp-mpi, cuda-mpi, hip-mpi,
ctest --preset serial          #       plus -debug variants (e.g. serial-debug)
```
The unit tests are built with the presets but only execute when `ctest` is invoked.
Each preset builds into `build/<preset-name>`, with example executables in `build/<preset-name>/bin`. Debug presets include checks on array and matrix dimensions and index bounds. On HPC machines, load your compiler/MPI/CUDA modules first, then run the preset; site-specific toolchains can be layered with a `CMakeUserPresets.json`.

Configuring manually instead of with presets works the same way:
```
cmake -B build -DMATAR_BUILD_EXAMPLES=ON -DKokkos_ENABLE_OPENMP=ON
cmake --build build -j
```

The CMake options are:

| Option | Default | Description |
|---|---|---|
| `MATAR_ENABLE_KOKKOS` | ON | Build the Kokkos-backed device/dual types (builds the bundled Kokkos submodule) |
| `MATAR_ENABLE_MPI` | OFF | Enable the MPI-aware types (`MPICArrayKokkos`, `CommunicationPlan`) |
| `MATAR_ENABLE_GPU_AWARE_MPI` | OFF | Assume the MPI implementation is GPU-aware |
| `MATAR_USE_EXTERNAL_KOKKOS` | OFF | Use an installed Kokkos (`-DKokkos_ROOT=<prefix>`) instead of the submodule |
| `MATAR_BUILD_EXAMPLES` | OFF | Build the example programs |
| `MATAR_BUILD_TESTS` | OFF | Build the unit tests (`ctest` to run) |
| `MATAR_BUILD_BENCHMARKS` | OFF | Build the benchmarks |
| `MATAR_REAL` | double | Precision of the `real_t` tier: `double`, `float`, `half`, `bfloat16`, `quad` |
| `MATAR_HIGH_REAL` | double | Precision of the `high_real_t` tier: `double`, `float`, `quad` |
| `MATAR_LOW_REAL` | double | Precision of the `low_real_t` tier: `double`, `float`, `half`, `bfloat16` |

## Host-side parallelism

Every MATAR macro (`FOR_ALL`, `DO_ALL`, `FOR_REDUCE_*`, ...) has a `_HOST` twin that runs in the **host** execution space instead of the device space. Because device kernels are asynchronous and host kernels only block the calling thread, CPU work can proceed while the GPU is still busy — useful for file I/O and for algorithms that are simply faster on a CPU:

```c++
FOR_ALL(i, 0, n, { device_field(i) = compute(i); });   // GPU, returns immediately

FOR_ALL_HOST(i, 0, n, {                                 // CPU, runs concurrently
    out_lines[i] = format_record(host_field(i));
});
MATAR_FENCE_HOST();                                     // wait for the host work
MATAR_FENCE_DEVICE();                                   // wait for the GPU work
```

The `_HOST` macros differ from their device counterparts in two ways: the loop body captures **by reference**, so `std::` containers, file streams, and other non-device-copyable objects can be used directly; and no `_CLASS` variants are needed (`_CLASS` spellings exist as aliases).

**A host kernel may only touch host-accessible data** — the `*Host` types, the `.host()` side of a Dual type, or plain `std::` data. Passing a device array compiles but aborts at run time with `attempt to access inaccessible memory space`.

Select the two backends independently:

| Option | Values | Description |
|---|---|---|
| `MATAR_DEVICE_BACKEND` | `serial`, `openmp`, `pthreads`, `cuda`, `hip`, `sycl` | Backend for `FOR_ALL` and friends |
| `MATAR_HOST_BACKEND` | `serial`, `openmp`, `pthreads` | Backend for the `_HOST` macros |

```
cmake --preset cuda-hostomp     # CUDA device + OpenMP host: genuinely concurrent
cmake -B build -DMATAR_DEVICE_BACKEND=cuda -DMATAR_HOST_BACKEND=openmp
```

Both options are optional; leaving them unset keeps the historical behavior of setting `Kokkos_ENABLE_*` directly. Note that Kokkos permits only one host-parallel backend per build, so `openmp` and `pthreads` cannot be paired with each other, and selecting the same backend on both sides is valid but gives no concurrency (the two macro families then share one execution space — MATAR warns at configure time).

## Precision tiers

MATAR provides three floating-point type names whose meaning is fixed at configure time — code just uses the names, and the CMake flags decide what they are per build/architecture:

* `real_t` — the default working precision for field data
* `high_real_t` — fields that must stay accurate (coordinates, conserved-quantity sums)
* `low_real_t` — tolerant or bulk-storage fields (history buffers, gradients, output)

```
cmake --preset serial-fp32                 # real_t = float
cmake -B build -DMATAR_REAL=half           # real_t = Kokkos half_t (native on GPU backends)
cmake -B build -DMATAR_REAL=float -DMATAR_HIGH_REAL=double   # mixed
```

Notes:
* `half`/`bfloat16` map to the native 16-bit types on CUDA/HIP/SYCL and are transparently float-backed on CPU backends (the macros `MATAR_FP16_IS_EMULATED`/`MATAR_BF16_IS_EMULATED` report which at compile time). `quad` is `__float128`, host backends only. Non-Kokkos builds support only `double` and `float`; anything else fails at configure.
* User code only ever writes the three tier names — declare fields as `CArrayDevice<real_t>` (or `high_real_t`/`low_real_t`) and write constants as `real_t(0.5)`; the build flags decide what those names mean. The solvers in `solvers/` compute in `real_t`, so they run at whatever working precision the build selects.
* The MPI-aware types work at every tier: `MPICArrayKokkos<real_t>` halo exchange and `all_reduce` need nothing extra from the user. Internally, native 16-bit types travel as bytes and reduce by promoting to double on the wire (exact), and quad uses a custom MPI datatype and reduction op.

The Kokkos backend is selected with the standard Kokkos CMake variables (`Kokkos_ENABLE_OPENMP`, `Kokkos_ENABLE_CUDA`, `Kokkos_ENABLE_HIP`, `Kokkos_ARCH_*`, ...), which are passed through to the bundled Kokkos build. With `MATAR_ENABLE_KOKKOS=OFF`, MATAR is a dependency-free serial header-only library.

## Using MATAR as a third-party library
MATAR is header-only and exports a single CMake target, **`matar::matar`**. Linking it propagates everything a consumer needs: the include paths, the C++20 requirement, the `HAVE_KOKKOS`/`HAVE_CUDA`/`HAVE_HIP`/`HAVE_OPENMP`/`HAVE_THREADS`/`HAVE_MPI` compile definitions, and the Kokkos/MPI link dependencies. There is nothing to link manually and no MATAR library to build.

Consumers need **CMake >= 3.22** and a **C++20** compiler; both come from Kokkos 5. Whichever method you choose, set the Kokkos backend variables *before* MATAR is added, because the bundled Kokkos is configured as part of that step.

### Option 1: FetchContent
CMake downloads MATAR at configure time. Nothing needs to be vendored into your repository.

```cmake
cmake_minimum_required(VERSION 3.22)
project(myapp LANGUAGES CXX)

# Pick the Kokkos backend BEFORE MATAR is made available.
set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
    matar
    GIT_REPOSITORY https://github.com/lanl/MATAR.git
    GIT_TAG        <tag-or-commit>   # pin a release; avoid tracking a branch
)
FetchContent_MakeAvailable(matar)    # also configures the bundled Kokkos

add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE matar::matar)
```

MATAR carries Kokkos as its own submodule, and FetchContent updates submodules recursively by default, so the bundled Kokkos is fetched and built automatically. No extra steps are needed.

### Option 2: git submodule
Vendors MATAR into your repository, which pins the exact commit and allows offline builds.

```bash
git submodule add https://github.com/lanl/MATAR.git external/MATAR
git submodule update --init --recursive
```

`--recursive` is required: MATAR contains Kokkos as a nested submodule, and without it `external/MATAR/src/Kokkos/kokkos` is left empty and the MATAR build stops with an error telling you to re-run the command. Anyone cloning your project afterwards needs `git clone --recursive`, or the same `git submodule update --init --recursive` after cloning.

```cmake
cmake_minimum_required(VERSION 3.22)
project(myapp LANGUAGES CXX)

set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
add_subdirectory(external/MATAR)

add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE matar::matar)
```

### Option 3: an installed MATAR
Against a MATAR installed with `cmake --install build/<preset> --prefix <prefix>`:

```cmake
find_package(Matar REQUIRED)         # configure with -DCMAKE_PREFIX_PATH=<prefix>
target_link_libraries(myapp PRIVATE matar::matar)
```

When the bundled Kokkos was used, it is installed into the same prefix and `MatarConfig.cmake` resolves it through `find_dependency(Kokkos)`.

### Selecting a backend, and reusing your own Kokkos
The backend is chosen with the standard Kokkos cache variables (`Kokkos_ENABLE_SERIAL`, `Kokkos_ENABLE_OPENMP`, `Kokkos_ENABLE_CUDA`, `Kokkos_ENABLE_HIP`, `Kokkos_ARCH_*`, ...), which are passed straight through to the bundled Kokkos build. Set them from your own `CMakeLists.txt` as shown above, or on the command line with `-DKokkos_ENABLE_CUDA=ON`.

If your project already provides Kokkos — through `find_package(Kokkos)` or an `add_subdirectory` of its own Kokkos copy — do that **before** adding MATAR. MATAR detects the existing `Kokkos::kokkos` target and links against it instead of building its bundled submodule, so only one Kokkos ends up in the build:

```cmake
find_package(Kokkos REQUIRED)        # your Kokkos wins
FetchContent_MakeAvailable(matar)    # MATAR reuses it
```

Other options worth knowing when embedding MATAR:

| Option | Effect |
|---|---|
| `MATAR_ENABLE_KOKKOS=OFF` | Host-only types; MATAR becomes a dependency-free header library and the parallel macros compile to plain serial loops |
| `MATAR_ENABLE_MPI=ON` | Adds `MPICArrayKokkos` and `CommunicationPlan` (requires MPI) |
| `MATAR_INSTALL` | Defaults to off when MATAR is embedded in another project, so it contributes no install rules to your package |

Note that while an embedded MATAR installs nothing, the *bundled Kokkos* still adds its own install rules, so `cmake --install` on your project will also deposit the Kokkos headers and CMake package files into your prefix. Use an external or parent-provided Kokkos if you need to keep it out of your install tree.

## Running codes in parallel
The openMP and pthread Kokkos backends require the user to specify the number of threads used to run the code in parallel. 
To specify the number of threads with the Kokkos pthread backend, add the following command line argument when executing the code,
```
--kokkos-threads=4
```
in otherwords,
```
./mycode --kokkos-threads=4
```
The above command runs the code with fine grained parallelism using 4 threads.  In your code, ensure you pass the command line argument variables to the MATAR_INITIALIZE macro (which wraps Kokkos::initialize) as shown below here.
```
int main(int argc, char* argv[])
{
    MATAR_INITIALIZE(argc, argv);

    // coding goes here

    MATAR_FINALIZE();

    return 0;
}
```
The above coding will still work if no command line arguments are given; that is key because the other kokkos backends do not need command line arguments.

For the openMP backend, set the number of threads as an environement variable; this is done by typing the following command in the terminal,
```
export OMP_NUM_THREADS=4
```
The CUDA and HIP backends do not need the number of threads specified.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
This program is open source under the BSD-3 License.

## Citation
```
@article{MATAR,
title = "{MATAR: A Performance Portability and Productivity Implementation of Data-Oriented Design with Kokkos}",
journal = {Journal of Parallel and Distributed Computing},
pages = {86-104},
volume = {157},
year = {2021},
author = {Daniel J. Dunning and Nathaniel R. Morgan and Jacob L. Moore and Eappen Nelluvelil and Tanya V. Tafolla and Robert W. Robey},
keywords = {Performance, Portability, Productivity, Memory Efficiency, GPUs, dense and sparse storage}
```

