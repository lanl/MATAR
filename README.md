# MATAR
<p align="center"><img src="https://github.com/lanl/MATAR/blob/main/MATAR_Logo.png" width="350">

MATAR is a C++ library that addresses the need for simple, fast, and memory-efficient multi-dimensional data representations for dense and sparse storage that arise with numerical methods and in software applications. The data representations are designed to perform well across multiple computer architectures, including CPUs and GPUs. MATAR allows users to easily create and use intricate data representations that are also portable across disparate architectures using Kokkos. The performance aspect is achieved by forcing contiguous memory layout (or as close to contiguous as possible) for multi-dimensional and multi-size dense or sparse MATrix and ARray (hence, MATAR) types. Results show that MATAR has the capability to improve memory utilization, performance, and programmer productivity in scientific computing. This is achieved by fitting more work into the available memory, minimizing memory loads required, and by loading memory in the most efficient order. 


## Examples
* [ELEMENTS](https://github.com/lanl/ELEMENTS/):   MATAR is a part of the ELEMENTS Library (LANL C# C20058) and it underpins the routines implemented in ELEMENTS.  MATAR is available in a stand-alone directory outside of the ELEMENTS directory because it can aid many code applications.  The dense and sparse storage types in MATAR are the foundation for the ELEMENTS library, which contains mathematical functions to support a very broad range of element types including: linear, quadratic, and cubic serendipity elements in 2D and 3D; high-order spectral elements; and a linear 4D element. An unstructured high-order mesh class is available in ELEMENTS and it takes advantage of MATAR for efficient access of various mesh entities. 

* Simple examples are in the /test folder

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

## Standard build
```
cmake .
make
```

## Kokkos build
```
cmake -DKOKKOS=1 .
make
```

## Kokkos with CUDA build
```
cmake -DKOKKOS=1  -DCUDA=1 .
make
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
This program is open source under the BSD-3 License.

