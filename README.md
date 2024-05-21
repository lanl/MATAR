# MATAR
<p align="center"><img src="https://github.com/lanl/MATAR/blob/main/MATAR_Logo.png" width="350">

MATAR is a C++ library that addresses the need for simple, fast, and memory-efficient multi-dimensional data representations for dense and sparse storage that arise with numerical methods and in software applications. The data representations are designed to perform well across multiple computer architectures, including CPUs and GPUs. MATAR allows users to easily create and use intricate data representations that are also portable across disparate architectures using Kokkos. The performance aspect is achieved by forcing contiguous memory layout (or as close to contiguous as possible) for multi-dimensional and multi-size dense or sparse MATrix and ARray (hence, MATAR) types. Results show that MATAR has the capability to improve memory utilization, performance, and programmer productivity in scientific computing. This is achieved by fitting more work into the available memory, minimizing memory loads required, and by loading memory in the most efficient order. 


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

## Basic build
The basic build is for users only interested in the serial CPU only MATAR data types.  For this build, we recommend making a folder perhaps called build then go into the build folder and type
```
cmake ..
make
```
The compiled code will be in the build folder.

## Debug basic build 

To build serial CPU only MATAR data types in the debug mode, please use
```
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```
The debug flag includes checks on array and matrix dimensions and index bounds.


## Building MATAR with Kokkos
A building script is provided to build the MATAR examples and tests, with or without Kokkos. The simplest build with all defaults can be run with
```
source {path-to-repo}/scripts/build-matar.sh
```
Running with the argument ```--help``` will give a full list of all possible arguments.
If an argument is not changed, it will be set to the default action, which can all be found from the help command
If the scripts fail to build, then carefully review the modules used and the computer architecture settings.

## Building MATAR with Anaconda
The recommended way to build **MATAR** is inside an Anaconda environment. As a starting place, follow the steps for your platform to install [anaconda](https://docs.anaconda.com/free/anaconda/install/index.html)/[miniconda](https://docs.conda.io/en/latest/miniconda.html)/[mamba](https://mamba.readthedocs.io/en/latest/installation.html). 

Open a terminal on your machine and go to a folder where you want to run the **MATAR** code. Activate a bash terminal by typing:
```
bash
```
Then create and activate an Anaconda environment by typing:  
```
conda create -n MATAR
conda activate MATAR  
```
In this example, the enviroment is called MATAR, but any name can be used.  In some cases, the text to activate an enviroment is `source activate MATAR`.  Likewise, if an enviroment already exists, then just activate the desired environment. 

Now install a compiler and cmake, which are needed to build the MATAR library.
```
conda install -c conda-forge "cxx-compiler=1.5.2"     
conda install -c conda-forge "fortran-compiler=1.5.2"
conda install cmake
```
By using cxx-compiler=1.5.2., it install gcc 11.  Omit the version number and gcc 12 will be installed (at this time).  If building for a GPU, it is recommended to use an older gcc version. For example, we have success using gcc 11 with CUDA 12.

If running on an Nvidia GPU, install cudatoolkit by typing:
```
conda install -c conda-forge cudatoolkit    
conda install -c conda-forge cudatoolkit-dev
```
This installs CUDA 12 (at this time).  

The build script is located at
```
source {path-to-repo}/scripts/build-matar.sh
```

To build the MATAR library and examples with CUDA, type:
```
source build-matar.sh --kokkos_build_type=cuda --build_cores=16
```
The executables for the examples that run in parallel Nvidia GPUs using CUDA are located in:
```
MATAR/build-matar-cuda/bin
```

To build the MATAR library and examples with OpenMP, type:
```
source build-matar.sh --kokkos_build_type=openmp --build_cores=16
```

The executables for the examples that run in parallel on multi-core CPUs using OpenMP are located in:
```
MATAR/build-matar-openmp/bin
```
Using the main_kokkos.cpp executable as an example, it can be run by typing:
```
./mtestkokkos
```

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
The above command runs the code with fine grained parallelism using 4 threads.  For the openMP backend, set the number of threads as an environement variable; this is done by typing the following command in the terminal,
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

