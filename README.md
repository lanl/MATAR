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
MATAR is built entirely with CMake. Kokkos is bundled as a git submodule (`src/Kokkos/kokkos`) and is built automatically with the backend you select, so no separate Kokkos install is needed.

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

The Kokkos backend is selected with the standard Kokkos CMake variables (`Kokkos_ENABLE_OPENMP`, `Kokkos_ENABLE_CUDA`, `Kokkos_ENABLE_HIP`, `Kokkos_ARCH_*`, ...), which are passed through to the bundled Kokkos build. With `MATAR_ENABLE_KOKKOS=OFF`, MATAR is a dependency-free serial header-only library.

## Using MATAR in your CMake project
MATAR and Kokkos can be pulled into another CMake project with FetchContent:
```cmake
include(FetchContent)
FetchContent_Declare(
    matar
    GIT_REPOSITORY https://github.com/lanl/MATAR
    GIT_TAG        <tag-or-branch>
)
FetchContent_MakeAvailable(matar)   # builds bundled Kokkos with your Kokkos_ENABLE_* settings
target_link_libraries(myapp PRIVATE matar::matar)
```
or by adding the repository as a git submodule:
```cmake
add_subdirectory(path/to/MATAR)
target_link_libraries(myapp PRIVATE matar::matar)
```
or against an installed MATAR (`cmake --install build/<preset> --prefix <prefix>`):
```cmake
find_package(Matar REQUIRED)        # -DCMAKE_PREFIX_PATH=<prefix>
target_link_libraries(myapp PRIVATE matar::matar)
```
Linking `matar::matar` carries the include paths, `HAVE_KOKKOS`/`HAVE_MPI` definitions, and the Kokkos/MPI link dependencies automatically. If your project provides its own Kokkos (via `add_subdirectory` or `find_package` before MATAR), MATAR uses that Kokkos rather than the submodule.

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

