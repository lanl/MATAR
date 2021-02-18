# MATAR
![MATAR](./MATAR_Logo.png){:height="50%" width="50%"}

MATAR addresses the need for simple, fast, and memory-efficient multi-dimensional data representations for dense and sparse storage that arise with numerical methods and in software applications. The data representations are designed to perform well across multiple computer architectures, including CPUs and GPUs. For this purpose, we developed MATAR, a C++ software library that allows for simple creation and use of intricate data representations that is also portable across disparate architectures using Kokkos. The performance aspect is achieved by forcing contiguous memory layout (or as close to contiguous as possible) for multi-dimensional and multi-size dense or sparse MATrix and ARray (hence, MATAR) types. Results show that MATAR has the capability to improve memory utilization, performance, and programmer productivity in scientific computing. This is achieved by fitting more work into the available memory, minimizing memory loads required, and by loading memory in the most efficient order. 


## Examples
* [ELEMENTS](https://github.com/lanl/ELEMENTS/):   MATAR is a part of the ELEMENTS Library (LANL C# C20058) and it underpins the routines implemented in ELEMENTS.  MATAR is available in a stand-alone directory outside of the ELEMENTS directory because it can aid many code applications.  The dense and sparse storage types in MATAR are the foundation for the ELEMENTS library, which contains mathematical functions to support a very broad range of element types including: linear, quadratic, and cubic serendipity elements in 2D and 3D; high-order spectral elements; and a linear 4D element. An unstructured high-order mesh class is available in ELEMENTS and it takes advantage of MATAR for efficient access of various mesh entities. 

* Simple examples are in the /test folder


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


## License
This program is open source under the BSD-3 License.

