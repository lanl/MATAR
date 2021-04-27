#ifndef MATAR_H
#define MATAR_H
/*****************************************************************************
 © 2020. Triad National Security, LLC. All rights reserved.
 This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
 National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
 Department of Energy/National Nuclear Security Administration. All rights in the program are
 reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
 Security Administration. The Government is granted for itself and others acting on its behalf a
 nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
 derivative works, distribute copies to the public, perform publicly and display publicly, and
 to permit others to do so.
 This program is open source under the BSD-3 License.
 Redistribution and use in source and binary forms, with or without modification, are permitted
 provided that the following conditions are met:
 
 1.  Redistributions of source code must retain the above copyright notice, this list of
 conditions and the following disclaimer.
 
 2.  Redistributions in binary form must reproduce the above copyright notice, this list of
 conditions and the following disclaimer in the documentation and/or other materials
 provided with the distribution.
 
 3.  Neither the name of the copyright holder nor the names of its contributors may be used
 to endorse or promote products derived from this software without specific prior
 written permission.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **********************************************************************************************/

// Order
//
//  Standard (non-Kokkos data structures)
//   1. FArray
//   2. ViewFArray
//   3. FMatrix
//   4. ViewFMatrix
//   5. CArray
//   6. ViewCArray
//   7. CMatrix
//   8. ViewCMatrix
//   9. RaggedRightArray
//   10. RaggedDownArray
//   11. DynamicRaggedRightArray
//   12. DynamicRaggedDownArray
//   13. SparseRowArray
//   14. SparseColArray
//
//   Kokkos Data structures
//   15. FArrayKokkos
//   16. ViewFArrayKokkos
//   17. FMatrixKokkos
//   18. ViewFMatrixKokkos
//   19. CArrayKokkos
//   20. ViewCArrayKokkos
//   21. CMatrixKokkos
//   22. ViewCMatrixKokkos
//   23. RaggedRightArrayKokkos
//   24. RaggedDownArrayKokkos
//   25. DynamicRaggedRightArrayKokkos
//   26. DynamicRaggedDownArrayKokkos
//   27. SparseRowArrayKokkos
//   28. SparseColArrayKokkos


#include <stdio.h>
#include <stdlib.h>

#include <assert.h>

using real_t = double;
using u_int  = unsigned int;


#ifdef HAVE_KOKKOS
#include <Kokkos_Core.hpp>

//MACROS to make the code less scary
#define kmalloc(size) ( Kokkos::kokkos_malloc<MemSpace>(size) )
#define kfree(pnt)        (  Kokkos::kokkos_free(pnt) ) 
#define ProfileRegionStart  ( Kokkos::Profiling::pushRegion )
#define ProfileRegionEnd  ( Kokkos::Profiling::popRegion )
using HostSpace    = Kokkos::HostSpace;
using MemoryUnmanaged = Kokkos::MemoryUnmanaged;

#ifdef HAVE_CUDA
//using UVMMemSpace     = Kokkos::CudaUVMSpace;
using DefaultMemSpace  = Kokkos::CudaSpace;
using DefaultExecSpace = Kokkos::Cuda;
using DefaultLayout    = Kokkos::LayoutLeft;
#elif HAVE_OPENMP
using DefaultMemSpace  = Kokkos::HostSpace;
using DefaultExecSpace = Kokkos::OpenMP;
using DefaultLayout    = Kokkos::LayoutRight;
#elif TRILINOS_INTERFACE
using DefaultMemSpace  = void;
using DefaultExecSpace = void;
using DefaultLayout    = void;
#elif HAVE_HIP
using DefaultMemSpace  = Kokkos::HipSpace;
using DefaultExecSpace = Kokkos::Hip;
using DefaultLayout    = Kokkos::LayoutLeft;
#endif

using policy1D = Kokkos::RangePolicy<DefaultExecSpace>;
using policy2D = Kokkos::MDRangePolicy< Kokkos::Rank<2> >;
using policy3D = Kokkos::MDRangePolicy< Kokkos::Rank<3> >;
using policy4D = Kokkos::MDRangePolicy< Kokkos::Rank<4> >;

using TeamPolicy = Kokkos::TeamPolicy<DefaultExecSpace>;
//using mdrange_policy2 = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
//using mdrange_policy3 = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

using RMatrix1D    = Kokkos::View<real_t *,DefaultLayout,DefaultExecSpace>;
using RMatrix2D    = Kokkos::View<real_t **,DefaultLayout,DefaultExecSpace>;
using RMatrix3D    = Kokkos::View<real_t ***,DefaultLayout,DefaultExecSpace>;
using RMatrix4D    = Kokkos::View<real_t ****,DefaultLayout,DefaultExecSpace>;
using RMatrix5D    = Kokkos::View<real_t *****,DefaultLayout,DefaultExecSpace>;
using IMatrix1D    = Kokkos::View<int *,DefaultLayout,DefaultExecSpace>;
using IMatrix2D    = Kokkos::View<int **,DefaultLayout,DefaultExecSpace>;
using IMatrix3D    = Kokkos::View<int ***,DefaultLayout,DefaultExecSpace>;
using IMatrix4D    = Kokkos::View<int ****,DefaultLayout,DefaultExecSpace>;
using IMatrix5D    = Kokkos::View<int *****,DefaultLayout,DefaultExecSpace>;
using SVar         = Kokkos::View<size_t,DefaultLayout,DefaultExecSpace>;
using SArray1D     = Kokkos::View<size_t *,DefaultLayout,DefaultExecSpace>;
using SArray2D     = Kokkos::View<size_t **,DefaultLayout,DefaultExecSpace>;
using SArray3D     = Kokkos::View<size_t ***,DefaultLayout,DefaultExecSpace>;
using SArray4D     = Kokkos::View<size_t ****,DefaultLayout,DefaultExecSpace>;
using SArray5D     = Kokkos::View<size_t *****,DefaultLayout,DefaultExecSpace>;

using SHArray1D     = Kokkos::View<size_t *,DefaultLayout,Kokkos::HostSpace>;
#endif

//To disable asserts, uncomment the following line
//#define NDEBUG


//---Begin Standard Data Structures---

//1. FArray
// indicies are [0:N-1]
template <typename T>
class FArray {
    
private:
    size_t dims_[7];
    size_t length_;
    T * array_;
    
public:
    
    // default constructor
   FArray ();
   
    //overload constructors from 1D to 7D
     
   FArray(size_t dim0);
    
   FArray(size_t dim0,
          size_t dim1);
    
   FArray(size_t dim0,
          size_t dim1,
          size_t dim2);
    
   FArray(size_t dim0,
          size_t dim1,
          size_t dim2,
          size_t dim3);
    
   FArray(size_t dim0,
          size_t dim1,
          size_t dim2,
          size_t dim3,
          size_t dim4);

   FArray(size_t dim0,
          size_t dim1,
          size_t dim2,
          size_t dim3,
          size_t dim4,
          size_t dim5);

   FArray(size_t dim0,
          size_t dim1,
          size_t dim2,
          size_t dim3,
          size_t dim4,
          size_t dim5,
          size_t dim6);
    
    // overload operator() to access data as array(i,....,n);
    T& operator()(size_t i) const;
    
    T& operator()(size_t i,
                  size_t j) const;
    
    T& operator()(size_t i,
                  size_t j,
                  size_t k) const;
    
    T& operator()(size_t i,
                  size_t j,
                  size_t k,
                  size_t l) const;
    
    T& operator()(size_t i,
                  size_t j,
                  size_t k,
                  size_t l,
                  size_t m) const;
    
    T& operator()(size_t i,
                  size_t j,
                  size_t k,
                  size_t l,
                  size_t m,
                  size_t n) const;
    
    T& operator()(size_t i,
                  size_t j,
                  size_t k,
                  size_t l,
                  size_t m,
                  size_t n,
                  size_t o) const;
    
    //overload = operator
    FArray& operator=(const FArray& temp);
    
    //return array size
    size_t size() const;

    size_t dims(size_t i) const;
    
    //return pointer
    T* get_pointer() const;
    
    // deconstructor
    ~FArray ( );
    
}; // end of f_array_t

//---FArray class definnitions----

//constructors
template <typename T>
FArray<T>::FArray(){
    array_ = NULL;
}

//1D
template <typename T>
FArray<T>::FArray(size_t dim0)
{
    dims_[0] = dim0;
    length_ = dim0;
    array_ = new T[length_];
}

template <typename T>
FArray<T>::FArray(size_t dim0,
                  size_t dim1)
{
    dims_[0] = dim0;
    dims_[1] = dim1;
    length_ = dim0*dim1;
    array_ = new T[length_];
}

//3D
template <typename T>
FArray<T>::FArray(size_t dim0,
                  size_t dim1,
                  size_t dim2)
{
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    length_ = dim0*dim1*dim2;
    array_ = new T[length_];
}

//4D
template <typename T>
FArray<T>::FArray(size_t dim0,
                  size_t dim1,
                  size_t dim2,
                  size_t dim3)
{
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    length_ = dim0*dim1*dim2*dim3;
    array_ = new T[length_];
}

//5D
template <typename T>
FArray<T>::FArray(size_t dim0,
                  size_t dim1,
                  size_t dim2,
                  size_t dim3,
                  size_t dim4)
{
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    length_ = dim0*dim1*dim2*dim3*dim4;
    array_ = new T[length_];
}

//6D
template <typename T>
FArray<T>::FArray(size_t dim0,
                  size_t dim1,
                  size_t dim2,
                  size_t dim3,
                  size_t dim4,
                  size_t dim5)
{
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    dims_[5] = dim5;
    length_ = dim0*dim1*dim2*dim3*dim4*dim5;
    array_ = new T[length_];
}


//7D
template <typename T>
FArray<T>::FArray(size_t dim0,
                  size_t dim1,
                  size_t dim2,
                  size_t dim3,
                  size_t dim4,
                  size_t dim5,
                  size_t dim6)
{
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    dims_[5] = dim5;
    dims_[6] = dim6;
    length_ = dim0*dim1*dim2*dim3*dim4*dim5*dim6;
    array_ = new T[length_];
        
}

//overload operator () for 1D to 7D
//indices are from [0:N-1]

//1D
template <typename T>
T& FArray<T>::operator()(size_t i) const
{
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in FArray 1D!");
    return array_[i];
}

//2D
template <typename T>
T& FArray<T>::operator()(size_t i,
                         size_t j) const
{
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in FArray 2D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in FArray 2D!");
    return array_[i + j*dims_[0]];
}

//3D
template <typename T>
T& FArray<T>::operator()(size_t i,
                         size_t j,
                         size_t k) const
{
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in FArray 3D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in Farray 3D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in FArray 3D!");
    return array_[i + j*dims_[0] + k*dims_[0]*dims_[1]];
}

//4D
template <typename T>
T& FArray<T>::operator()(size_t i,
                         size_t j,
                         size_t k,
                         size_t l) const
{
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in FArray 4D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in FArray 4D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in FArray 4D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in FArray 4D!");
    return array_[i + j*dims_[0]
                    + k*dims_[0]*dims_[1]
                    + l*dims_[0]*dims_[1]*dims_[2]];
}

//5D
template <typename T>
T& FArray<T>::operator()(size_t i,
                         size_t j,
                         size_t k,
                         size_t l,
                         size_t m) const
{
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in FArray 5D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in FArray 5D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in FArray 5D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in FArray 5D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in FArray 5D!");
    return array_[i + j*dims_[0]
                    + k*dims_[0]*dims_[1]
                    + l*dims_[0]*dims_[1]*dims_[2]
                    + m*dims_[0]*dims_[1]*dims_[2]*dims_[3]];
}

//6D
template <typename T>
T& FArray<T>::operator()(size_t i,
                         size_t j,
                         size_t k,
                         size_t l,
                         size_t m,
                         size_t n) const
{

    assert(i >= 0 && i < dims_[0] && "i is out of bounds in FArray 6D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in FArray 6D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in FArray 6D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in FArray 6D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in FArray 6D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in FArray 6D!");
    return array_[i + j*dims_[0]
                    + k*dims_[0]*dims_[1]
                    + l*dims_[0]*dims_[1]*dims_[2]
                    + m*dims_[0]*dims_[1]*dims_[2]*dims_[3]
                    + n*dims_[0]*dims_[1]*dims_[2]*dims_[3]*dims_[4]];
}

//7D
template <typename T>
T& FArray<T>::operator()(size_t i,
                         size_t j,
                         size_t k,
                         size_t l,
                         size_t m,
                         size_t n,
                         size_t o) const
{
    
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in FArray 7D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in FArray 7D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in FArray 7D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in FArray 7D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in FArray 7D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in FArray 7D!");
    assert(o >= 0 && o < dims_[6] && "o is out of bounds in FArray 7D!");
    return array_[i + j*dims_[0]
                    + k*dims_[0]*dims_[1]
                    + l*dims_[0]*dims_[1]*dims_[2]
                    + m*dims_[0]*dims_[1]*dims_[2]*dims_[3]
                    + n*dims_[0]*dims_[1]*dims_[2]*dims_[3]*dims_[4]
                    + o*dims_[0]*dims_[1]*dims_[2]*dims_[3]*dims_[4]*dims_[5]];
}
    
// = operator
//THIS = FArray <> TEMP(n,m,...)
template <typename T>
FArray<T>& FArray<T>::operator= (const FArray& temp)
{
	if(this != & temp) {
	  dims_[0] = temp.dims_[0];
	  dims_[1] = temp.dims_[1];
	  dims_[2] = temp.dims_[2];
	  dims_[3] = temp.dims_[3];
	  dims_[4] = temp.dims_[4];
	  dims_[5] = temp.dims_[5];
      dims_[6] = temp.dims_[6];
	  length_ = temp.length_;
	  array_ = new T[length_];
	}
  return *this;
}

template <typename T>
inline size_t FArray<T>::size() const {
    return length_;
}

template <typename T>
inline size_t FArray<T>::dims(size_t i) const {
    return dims_[i];
}

template <typename T>
inline T* FArray<T>::get_pointer() const {
    return array_;
}

//delete FArray
template <typename T>
FArray<T>::~FArray(){
    delete [] array_;
}

//---end of FArray class definitions----


//2. ViewFArray
// indicies are [0:N-1]
template <typename T>
class ViewFArray {

private:
    size_t dims_[7];
    size_t length_; // Length of 1D array
    T * array_;
    
public:
    
    // default constructor
    ViewFArray ();

    //---1D to 7D array ---
    ViewFArray(T *array,
               size_t dim0);
    
    ViewFArray (T *array,
                size_t dim0,
                size_t dim1);

    ViewFArray (T *array,
                size_t dim0,
                size_t dim1,
                size_t dim2);

    ViewFArray (T *array,
                size_t dim0,
                size_t dim1,
                size_t dim2,
                size_t dim3);
    
    ViewFArray (T *array,
                size_t dim0,
                size_t dim1,
                size_t dim2,
                size_t dim3,
                size_t dim4);

    ViewFArray (T *array,
                size_t dim0,
                size_t dim1,
                size_t dim2,
                size_t dim3,
                size_t dim4,
                size_t dim5);
    
    ViewFArray (T *array,
                size_t dim0,
                size_t dim1,
                size_t dim2,
                size_t dim3,
                size_t dim4,
                size_t dim5,
                size_t dim6);
    
    T& operator()(size_t i) const;
    
    T& operator()(size_t i,
                  size_t j) const;
    
    T& operator()(size_t i,
                  size_t j,
                  size_t k) const;
    
    T& operator()(size_t i,
                  size_t j,
                  size_t k,
                  size_t l) const;
    
    T& operator()(size_t i,
                  size_t j,
                  size_t k,
                  size_t l,
                  size_t m) const;
    
    T& operator()(size_t i,
                  size_t j,
                  size_t k,
                  size_t l,
                  size_t m,
                  size_t n) const;
    
    T& operator()(size_t i,
                  size_t j,
                  size_t k,
                  size_t l,
                  size_t m,
                  size_t n,
                  size_t o) const;
    
    // calculate C = math(A,B)
    template <typename M>
    void operator=(M do_this_math);
    
    //return array size
    size_t size() const;
    
    //return array size
    size_t dims(size_t i) const;
    
}; // end of viewFArray

//class definitions for viewFArray

//~~~~constructors for viewFArray for 1D to 7D~~~~~~~

//no dimension
template <typename T>
ViewFArray<T>::ViewFArray(){}

//1D
template <typename T>
ViewFArray<T>::ViewFArray(T *array,
                          size_t dim0)
{
    dims_[0] = dim0;
    length_ = dim0;
	array_  = array;
}

//2D
template <typename T>
ViewFArray<T>::ViewFArray(T *array,
                          size_t dim0,
                          size_t dim1)
{
    dims_[0] = dim0;
    dims_[1] = dim1;
    length_ = dim0*dim1;
	array_  = array;
}

//3D
template <typename T>
ViewFArray<T>::ViewFArray(T *array,
                          size_t dim0,
                          size_t dim1,
                          size_t dim2)
{
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    length_ = dim0*dim1*dim2;
	array_  = array;
}

//4D
template <typename T>
ViewFArray<T>::ViewFArray(T *array,
                          size_t dim0,
                          size_t dim1,
                          size_t dim2,
                          size_t dim3)
{
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    length_ = dim0*dim1*dim2*dim3;
	array_  = array;
}

//5D
template <typename T>
ViewFArray<T>::ViewFArray(T *array,
                          size_t dim0,
                          size_t dim1,
                          size_t dim2,
                          size_t dim3,
                          size_t dim4)
{
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    length_ = dim0*dim1*dim2*dim3*dim4;
	array_  = array;
}

//6D
template <typename T>
ViewFArray<T>::ViewFArray(T *array,
                          size_t dim0,
                          size_t dim1,
                          size_t dim2,
                          size_t dim3,
                          size_t dim4,
                          size_t dim5)
{
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    dims_[5] = dim5;
    length_ = dim0*dim1*dim2*dim3*dim4*dim5;
	array_  = array;
}

//7D
template <typename T>
ViewFArray<T>::ViewFArray(T *array,
                          size_t dim0,
                          size_t dim1,
                          size_t dim2,
                          size_t dim3,
                          size_t dim4,
                          size_t dim5,
                          size_t dim6)
{
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    dims_[5] = dim5;
    dims_[6] = dim6;
    length_ = dim0*dim1*dim2*dim3*dim4*dim5*dim6;
    array_  = array;
}

//~~~~~~operator () overload 
//for dimensions 1D to 7D
//indices for array are from 0...N-1

//1D
template <typename T>
T& ViewFArray<T>::operator()(size_t i) const
{
	assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewFArray 1D!");
	return array_[i];
}

//2D
template <typename T>
T& ViewFArray<T>::operator()(size_t i,
                             size_t j) const
{
	assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewFArray 2D!");
	assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewFArray 2D!");
	return array_[i + j*dims_[0]];
}

//3D
template <typename T>
T& ViewFArray<T>::operator()(size_t i,
                             size_t j,
                             size_t k) const
{
	assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewFArray 3D!");
	assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewFArray 3D!");
	assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewFArray 3D!");
	return array_[i + j*dims_[0]
                    + k*dims_[0]*dims_[1]];
}

//4D
template <typename T>
T& ViewFArray<T>::operator()(size_t i,
                             size_t j,
                             size_t k,
                             size_t l) const
{
	assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewFArray 4D!");
	assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewFArray 4D!");
	assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewFArray 4D!");
	assert(l >= 0 && l < dims_[3] && "l is out of bounds in ViewFArray 4D!");
	return array_[i + j*dims_[0]
                    + k*dims_[0]*dims_[1]
                    + l*dims_[0]*dims_[1]*dims_[2]];
}

//5D
template <typename T>
T& ViewFArray<T>::operator()(size_t i,
                             size_t j,
                             size_t k,
                             size_t l,
                             size_t m) const
{
	assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewFArray 5D!");
	assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewFArray 5D!");
	assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewFArray 5D!");
	assert(l >= 0 && l < dims_[3] && "l is out of bounds in ViewFArray 5D!");
	assert(m >= 0 && m < dims_[4] && "m is out of bounds in ViewFArray 5D!");
	return array_[i + j*dims_[0]
                    + k*dims_[0]*dims_[1]
                    + l*dims_[0]*dims_[1]*dims_[2]
                    + m*dims_[0]*dims_[1]*dims_[2]*dims_[3]];
}

//6D
template <typename T>
T& ViewFArray<T>:: operator()(size_t i,
                              size_t j,
                              size_t k,
                              size_t l,
                              size_t m,
                              size_t n) const
{
	assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewFArray 6D!");
	assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewFArray 6D!");
	assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewFArray 6D!");
	assert(l >= 0 && l < dims_[3] && "l is out of bounds in ViewFArray 6D!");
	assert(m >= 0 && m < dims_[4] && "m is out of bounds in ViewFArray 6D!");
	assert(n >= 0 && n < dims_[5] && "n is out of bounds in ViewFArray 6D!");
	return array_[i + j*dims_[0]
                    + k*dims_[0]*dims_[1]
                    + l*dims_[0]*dims_[1]*dims_[2]
                    + m*dims_[0]*dims_[1]*dims_[2]*dims_[3]
                    + n*dims_[0]*dims_[1]*dims_[2]*dims_[3]*dims_[4]];
}

//7D
template <typename T>
T& ViewFArray<T>:: operator()(size_t i,
                              size_t j,
                              size_t k,
                              size_t l,
                              size_t m,
                              size_t n,
                              size_t o) const
{
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in ViewFArray 7D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in ViewFArray 7D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in ViewFArray 7D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in ViewFArray 7D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in ViewFArray 7D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in ViewFArray 7D!");
    assert(o >= 0 && o < dims_[6] && "n is out of bounds in ViewFArray 7D!");
    return array_[i + j*dims_[0]
                    + k*dims_[0]*dims_[1]
                    + l*dims_[0]*dims_[1]*dims_[2]
                    + m*dims_[0]*dims_[1]*dims_[2]*dims_[3]
                    + n*dims_[0]*dims_[1]*dims_[2]*dims_[3]*dims_[4]
                    + o*dims_[0]*dims_[1]*dims_[2]*dims_[3]*dims_[4]*dims_[5]];
}

// calculate this ViewFArray object = math(A,B)
template <typename T>
template <typename M>
void ViewFArray<T>::operator=(M do_this_math){
    do_this_math(*this); // pass in this ViewFArray object
}// end of math opperation

template <typename T>
inline size_t ViewFArray<T>::dims(size_t i) const {
    return dims_[i];
}

template <typename T>
inline size_t ViewFArray<T>::size() const {
    return length_;
}

//---end of ViewFArray class definitions---


//3. FMatrix
// indicies are [1:N]
template <typename T>
class FMatrix {
private:
    size_t dim1_, dim2_, dim3_, dim4_, dim5_, dim6_, dim7_;
    size_t length_; // Length of 1D array
    T* this_matrix_;

public:
    // Default constructor
    FMatrix ();

    //---1D to 7D matrix ---
    FMatrix (size_t some_dim1);

    FMatrix (size_t some_dim1,
             size_t some_dim2);

    FMatrix (size_t some_dim1,
             size_t some_dim2,
             size_t some_dim3);

    FMatrix (size_t some_dim1,
             size_t some_dim2,
             size_t some_dim3,
             size_t some_dim4);

    FMatrix (size_t some_dim1,
             size_t some_dim2,
             size_t some_dim3,
             size_t some_dim4,
             size_t some_dim5);

    FMatrix (size_t some_dim1,
             size_t some_dim2,
             size_t some_dim3,
             size_t some_dim4,
             size_t some_dim5,
             size_t some_dim6);


    FMatrix (size_t some_dim1,
             size_t some_dim2,
             size_t some_dim3,
             size_t some_dim4,
             size_t some_dim5,
             size_t some_dim6,
             size_t some_dim7);
    
    T& operator() (size_t i) const;
    
    T& operator() (size_t i,
                   size_t j) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l,
                   size_t m) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l,
                   size_t m,
                   size_t n) const;

    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l,
                   size_t m,
                   size_t n,
                   size_t o) const;
    
    
    // Overload copy assignment operator
    FMatrix& operator=(const FMatrix& temp);

    size_t size() const;

    //return pointer
    T* get_pointer() const;

    // Deconstructor
    ~FMatrix ();

}; // End of FMatrix

//---FMatrix class definitions---

//constructors
template <typename T>
FMatrix<T>::FMatrix(){
    this_matrix_ = NULL;
}

//1D
template <typename T>
FMatrix<T>::FMatrix(size_t some_dim1)
{
    dim1_ = some_dim1;
    length_ = dim1_;
    this_matrix_ = new T[length_];
}

//2D
template <typename T>
FMatrix<T>::FMatrix(size_t some_dim1,
                    size_t some_dim2)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    length_ = dim1_ * dim2_;
    this_matrix_ = new T[length_];
}

//3D
template <typename T>
FMatrix<T>::FMatrix(size_t some_dim1,
                    size_t some_dim2,
                    size_t some_dim3)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    length_ = dim1_ * dim2_ * dim3_;
    this_matrix_ = new T[length_];
}

//4D
template <typename T>
FMatrix<T>::FMatrix(size_t some_dim1,
                    size_t some_dim2,
                    size_t some_dim3,
                    size_t some_dim4)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    length_ = dim1_ * dim2_ * dim3_ * dim4_;
    this_matrix_ = new T[length_];
}

//5D
template <typename T>
FMatrix<T>::FMatrix(size_t some_dim1,
                    size_t some_dim2,
                    size_t some_dim3,
                    size_t some_dim4,
                    size_t some_dim5)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    length_ = dim1_ * dim2_ * dim3_ * dim4_ * dim5_;
    this_matrix_ = new T[length_];
}

//6D
template <typename T>
FMatrix<T>::FMatrix(size_t some_dim1,
                    size_t some_dim2,
                    size_t some_dim3,
                    size_t some_dim4,
                    size_t some_dim5,
                    size_t some_dim6)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    length_ = dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_;
    this_matrix_ = new T[length_];

}

//overload operators

//1D
template <typename T>
inline T& FMatrix<T>::operator() (size_t i) const
{
    assert(i >= 1 && i <= dim1_);
    return this_matrix_[i - 1];
}

//2D
template <typename T>
inline T& FMatrix<T>::operator() (size_t i,
                                  size_t j) const
{
    assert(i >= 1 && i <= dim1_);
    assert(j >= 1 && j <= dim2_);
    return this_matrix_[(i - 1) + ((j - 1) * dim1_)];
}

//3D
template <typename T>
inline T& FMatrix<T>::operator() (size_t i,
                                  size_t j,
                                  size_t k) const
{
    assert(i >= 1 && i <= dim1_);
    assert(j >= 1 && j <= dim2_);
    assert(k >= 1 && k <= dim3_);
    return this_matrix_[(i - 1) + ((j - 1) * dim1_) 
                                + ((k - 1) * dim1_ * dim2_)];
}

//4D
template <typename T>
inline T& FMatrix<T>::operator() (size_t i,
                                  size_t j,
                                  size_t k,
                                  size_t l) const
{
    assert(i >= 1 && i <= dim1_);
    assert(j >= 1 && j <= dim2_);
    assert(k >= 1 && k <= dim3_);
    assert(l >= 1 && l <= dim4_);
    return this_matrix_[(i - 1) + ((j - 1) * dim1_)  
                                + ((k - 1) * dim1_ * dim2_)  
                                + ((l - 1) * dim1_ * dim2_ * dim3_)];
}

//5D
template <typename T>
inline T& FMatrix<T>::operator() (size_t i,
                                  size_t j,
                                  size_t k,
                                  size_t l,
                                  size_t m) const
{
    assert(i >= 1 && i <= dim1_);
    assert(j >= 1 && j <= dim2_);
    assert(k >= 1 && k <= dim3_);
    assert(l >= 1 && l <= dim4_);
    assert(m >= 1 && m <= dim5_);
    return this_matrix_[(i - 1) + ((j - 1) * dim1_)  
                                + ((k - 1) * dim1_ * dim2_)  
                                + ((l - 1) * dim1_ * dim2_ * dim3_) 
                                + ((m - 1) * dim1_ * dim2_ * dim3_ * dim4_)];
}

//6D
template <typename T>
inline T& FMatrix<T>::operator() (size_t i,
                                  size_t j,
                                  size_t k,
                                  size_t l,
                                  size_t m,
                                  size_t n) const
{
    assert(i >= 1 && i <= dim1_);
    assert(j >= 1 && j <= dim2_);
    assert(k >= 1 && k <= dim3_);
    assert(l >= 1 && l <= dim4_);
    assert(m >= 1 && m <= dim5_);
    assert(n >= 1 && n <= dim6_);
    return this_matrix_[(i - 1) + ((j - 1) * dim1_)  
                                + ((k - 1) * dim1_ * dim2_)  
                                + ((l - 1) * dim1_ * dim2_ * dim3_)  
                                + ((m - 1) * dim1_ * dim2_ * dim3_ * dim4_)  
                                + ((n - 1) * dim1_ * dim2_ * dim3_ * dim4_ * dim5_)];
}

//7D
template <typename T>
inline T& FMatrix<T>::operator() (size_t i,
                                  size_t j,
                                  size_t k,
                                  size_t l,
                                  size_t m,
                                  size_t n,
                                  size_t o) const
{
    assert(i >= 1 && i <= dim1_);
    assert(j >= 1 && j <= dim2_);
    assert(k >= 1 && k <= dim3_);
    assert(l >= 1 && l <= dim4_);
    assert(m >= 1 && m <= dim5_);
    assert(n >= 1 && n <= dim6_);
    assert(o >= 1 && n <= dim7_);
    return this_matrix_[(i - 1) + ((j - 1) * dim1_)
                                + ((k - 1) * dim1_ * dim2_)
                                + ((l - 1) * dim1_ * dim2_ * dim3_)
                                + ((m - 1) * dim1_ * dim2_ * dim3_ * dim4_)
                                + ((n - 1) * dim1_ * dim2_ * dim3_ * dim4_ * dim5_)
                                + ((o - 1) * dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_)];
}


template <typename T>
inline FMatrix<T>& FMatrix<T>::operator= (const FMatrix& temp)
{
    // Do nothing if assignment is of the form x = x
    if (this != &temp) {
        dim1_ = temp.dim1_;
        dim2_ = temp.dim2_;
        dim3_ = temp.dim3_;
        dim4_ = temp.dim4_;
        dim5_ = temp.dim5_;
        dim6_ = temp.dim6_;
        dim7_ = temp.dim7_;
        length_ = temp.length_;
        this_matrix_ = new T[length_];
    }
    
    return *this;
}

template <typename T>
inline size_t FMatrix<T>::size() const {
    return length_;
}

template <typename T>
inline T* FMatrix<T>::get_pointer() const{
    return this_matrix_;
}

template <typename T>
FMatrix<T>::~FMatrix() {
    delete[] this_matrix_;
}

//----end of FMatrix class definitions----


//4. ViewFMatrix
//  indices are [1:N]
template <typename T>
class ViewFMatrix {

private:
    size_t dim1_, dim2_, dim3_, dim4_, dim5_, dim6_, dim7_;
    size_t length_; // Length of 1D array
    T * this_matrix_;
    
public:
    
    // Default constructor
    ViewFMatrix ();
    
    //--- 1D to 7D matrix ---

    ViewFMatrix(T *some_matrix,
                size_t some_dim1);
    
    ViewFMatrix(T *some_matrix,
                size_t some_dim1,
                size_t some_dim2);
    
    ViewFMatrix(T *some_matrix,
                size_t some_dim1,
                size_t some_dim2,
                size_t some_dim3);
    
    ViewFMatrix(T *some_matrix,
                size_t some_dim1,
                size_t some_dim2,
                size_t some_dim3,
                size_t some_dim4);
    
    ViewFMatrix (T *some_matrix,
                 size_t some_dim1,
                 size_t some_dim2,
                 size_t some_dim3,
                 size_t some_dim4,
                 size_t some_dim5);
    
    ViewFMatrix (T *some_matrix,
                 size_t some_dim1,
                 size_t some_dim2,
                 size_t some_dim3,
                 size_t some_dim4,
                 size_t some_dim5,
                 size_t some_dim6);
    
    ViewFMatrix (T *some_matrix,
                 size_t some_dim1,
                 size_t some_dim2,
                 size_t some_dim3,
                 size_t some_dim4,
                 size_t some_dim5,
                 size_t some_dim6,
                 size_t some_dim7);
    
    T& operator()(size_t i) const;
    
    T& operator()(size_t i,
                  size_t j) const;
    
    T& operator()(size_t i,
                  size_t j,
                  size_t k) const;
    
    T& operator()(size_t i,
                  size_t j,
                  size_t k,
                  size_t l) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l,
                   size_t m) const;
    
    T& operator()(size_t i, 
                  size_t j, 
                  size_t k, 
                  size_t l, 
                  size_t m, 
                  size_t n) const;

    T& operator()(size_t i,
                  size_t j,
                  size_t k,
                  size_t l,
                  size_t m,
                  size_t n,
                  size_t o) const;
    
    size_t size() const;
    
}; // end of ViewFMatrix

//constructors

//no dimension
template <typename T>
ViewFMatrix<T>::ViewFMatrix() {}

//1D
template <typename T>
ViewFMatrix<T>::ViewFMatrix(T *some_matrix,
                            size_t some_dim1)
{
    dim1_ = some_dim1;
    length_ = dim1_;
    this_matrix_ = some_matrix;
}

//2D
template <typename T>
ViewFMatrix<T>::ViewFMatrix(T *some_matrix,
                            size_t some_dim1,
                            size_t some_dim2)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    length_ = dim1_ * dim2_;
    this_matrix_ = some_matrix;
}

//3D
template <typename T>
ViewFMatrix<T>::ViewFMatrix (T *some_matrix,
                             size_t some_dim1,
                             size_t some_dim2,
                             size_t some_dim3)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    length_ = dim1_ * dim2_ * dim3_;
    this_matrix_ = some_matrix;
}

//4D
template <typename T>
ViewFMatrix<T>::ViewFMatrix(T *some_matrix,
                            size_t some_dim1,
                            size_t some_dim2,
                            size_t some_dim3,
                            size_t some_dim4)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    length_ = dim1_ * dim2_ * dim3_ * dim4_;
    this_matrix_ = some_matrix;
}

//5D
template <typename T>
ViewFMatrix<T>::ViewFMatrix(T *some_matrix,
                            size_t some_dim1,
                            size_t some_dim2,
                            size_t some_dim3,
                            size_t some_dim4,
                            size_t some_dim5)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    length_ = dim1_ * dim2_ * dim3_ * dim4_ * dim5_;
    this_matrix_ = some_matrix;
}

//6D
template <typename T>
ViewFMatrix<T>::ViewFMatrix(T *some_matrix,
                            size_t some_dim1,
                            size_t some_dim2,
                            size_t some_dim3,
                            size_t some_dim4,
                            size_t some_dim5,
                            size_t some_dim6)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    length_ = dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_;
    this_matrix_ = some_matrix;
}

//6D
template <typename T>
ViewFMatrix<T>::ViewFMatrix(T *some_matrix,
                            size_t some_dim1,
                            size_t some_dim2,
                            size_t some_dim3,
                            size_t some_dim4,
                            size_t some_dim5,
                            size_t some_dim6,
                            size_t some_dim7)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    dim7_ = some_dim7;
    length_ = dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_ * dim7_;
    this_matrix_ = some_matrix;
}


//overload operator ()

//1D
template <typename T>
inline T& ViewFMatrix<T>::operator()(size_t i) const
{
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in c_matrix 1D");  // die if >= dim1
        
    return this_matrix_[(i - 1)];
}

//2D
template <typename T>
inline T& ViewFMatrix<T>::operator()(size_t i, 
                                     size_t j) const
{
       
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in c_matrix 2D");  // die if >= dim1
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in c_matrix 2D");  // die if >= dim2
        
    return this_matrix_[(i - 1) + ((j - 1) * dim1_)];
}

//3D
template <typename T>
inline T& ViewFMatrix<T>::operator()(size_t i, 
                                     size_t j, 
                                     size_t k) const
{
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in c_matrix 3D");  // die if >= dim1
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in c_matrix 3D");  // die if >= dim2
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in c_matrix 3D");  // die if >= dim3
        
    return this_matrix_[(i - 1) + ((j - 1) * dim1_) 
                                + ((k - 1) * dim1_ * dim2_)];
}

//4D
template <typename T>
inline T& ViewFMatrix<T>::operator()(size_t i, 
                                     size_t j, 
                                     size_t k, 
                                     size_t l) const
{
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in c_matrix 4D");  // die if >= dim1
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in c_matrix 4D");  // die if >= dim2
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in c_matrix 4D");  // die if >= dim3
    assert(l >= 1 && l <= dim4_ && "l is out of bounds in c_matrix 4D");  // die if >= dim4
        
    return this_matrix_[(i - 1) + ((j - 1) * dim1_) 
                                + ((k - 1) * dim1_ * dim2_) 
                                + ((l - 1) * dim1_ * dim2_ * dim3_)];
}

//5D
template <typename T>
inline T& ViewFMatrix<T>::operator()(size_t i, 
                                     size_t j, 
                                     size_t k, 
                                     size_t l, 
                                     size_t m) const
{
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in c_matrix 5D");  // die if >= dim1
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in c_matrix 5D");  // die if >= dim2
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in c_matrix 5D");  // die if >= dim3
    assert(l >= 1 && l <= dim4_ && "l is out of bounds in c_matrix 5D");  // die if >= dim4
    assert(m >= 1 && m <= dim5_ && "m is out of bounds in c_matrix 5D");  // die if >= dim5
       
    return this_matrix_[(i - 1) + ((j - 1) * dim1_) 
                                + ((k - 1) * dim1_ * dim2_) 
                                + ((l - 1) * dim1_ * dim2_ * dim3_)
                                + ((m - 1) * dim1_ * dim2_ * dim3_ * dim4_)];
}

//6D
template <typename T>
inline T& ViewFMatrix<T>::operator()(size_t i,
                                     size_t j,
                                     size_t k,
                                     size_t l,
                                     size_t m,
                                     size_t n) const
{
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in c_matrix 6D");  // die if >= dim1
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in c_matrix 6D");  // die if >= dim2
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in c_matrix 6D");  // die if >= dim3
    assert(l >= 1 && l <= dim4_ && "l is out of bounds in c_matrix 6D");  // die if >= dim4
    assert(m >= 1 && m <= dim5_ && "m is out of bounds in c_matrix 6D");  // die if >= dim5
    assert(n >= 1 && n <= dim6_ && "n is out of bounds in c_matrix 6D");  // die if >= dim6
    return this_matrix_[(i - 1) + ((j - 1) * dim1_) 
                                + ((k - 1) * dim1_ * dim2_) 
                                + ((l - 1) * dim1_ * dim2_ * dim3_)
                                + ((m - 1) * dim1_ * dim2_ * dim3_ * dim4_)
                                + ((n - 1) * dim1_ * dim2_ * dim3_ * dim4_ * dim5_)];
}

//6D
template <typename T>
inline T& ViewFMatrix<T>::operator()(size_t i,
                                     size_t j,
                                     size_t k,
                                     size_t l,
                                     size_t m,
                                     size_t n,
                                     size_t o) const
{
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in c_matrix 7D");  // die if >= dim1
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in c_matrix 7D");  // die if >= dim2
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in c_matrix 7D");  // die if >= dim3
    assert(l >= 1 && l <= dim4_ && "l is out of bounds in c_matrix 7D");  // die if >= dim4
    assert(m >= 1 && m <= dim5_ && "m is out of bounds in c_matrix 7D");  // die if >= dim5
    assert(n >= 1 && n <= dim6_ && "n is out of bounds in c_matrix 7D");  // die if >= dim6
    assert(o >= 1 && o <= dim7_ && "o is out of bounds in c_matrix 7D");  // die if >= dim7
    
    return this_matrix_[(i - 1) + ((j - 1) * dim1_)
                                + ((k - 1) * dim1_ * dim2_)
                                + ((l - 1) * dim1_ * dim2_ * dim3_)
                                + ((m - 1) * dim1_ * dim2_ * dim3_ * dim4_)
                                + ((n - 1) * dim1_ * dim2_ * dim3_ * dim4_ * dim5_)
                                + ((o - 1) * dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_)];
}

//-----end ViewFMatrix-----


//5. CArray
// indicies are [0:N-1]
template <typename T>
class CArray {
private:
    size_t dim1_, dim2_, dim3_, dim4_, dim5_, dim6_, dim7_;
    size_t length_; // Length of 1D array
    T* this_array_;

public:
    // Default constructor
    CArray ();

    // --- 1D to 7D array ---
    
    CArray (size_t some_dim1);

    CArray (size_t some_dim1,
            size_t some_dim2);

    CArray (size_t some_dim1,
            size_t some_dim2,
            size_t some_dim3);

    CArray (size_t some_dim1,
            size_t some_dim2,
            size_t some_dim3,
            size_t some_dim4);

    CArray (size_t some_dim1,
            size_t some_dim2,
            size_t some_dim3,
            size_t some_dim4,
            size_t some_dim5);

    CArray (size_t some_dim1,
            size_t some_dim2,
            size_t some_dim3,
            size_t some_dim4,
            size_t some_dim5,
            size_t some_dim6);

    CArray (size_t some_dim1,
            size_t some_dim2,
            size_t some_dim3,
            size_t some_dim4,
            size_t some_dim5,
            size_t some_dim6,
            size_t some_dim7);
    
    // Overload operator()
    T& operator() (size_t i) const;
    
    T& operator() (size_t i,
                   size_t j) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l,
                   size_t m) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l,
                   size_t m,
                   size_t n) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l,
                   size_t m,
                   size_t n,
                   size_t o) const;
    
    // Overload copy assignment operator
    CArray& operator= (const CArray& temp); 

    size_t size() const;

    //return pointer
    T* get_pointer() const;

    // Deconstructor
    ~CArray ();

}; // End of CArray

//---carray class declarations---

//constructors

//no dim
template <typename T>
CArray<T>::CArray() {
    this_array_ = NULL;
}

//1D
template <typename T>
CArray<T>::CArray(size_t some_dim1)
{
    dim1_ = some_dim1;
    length_ = dim1_;
    this_array_ = new T[length_];
}

//2D
template <typename T>
CArray<T>::CArray(size_t some_dim1,
                  size_t some_dim2)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    length_ = dim1_ * dim2_;
    this_array_ = new T[length_];
}

//3D
template <typename T>
CArray<T>::CArray(size_t some_dim1,
                  size_t some_dim2,
                  size_t some_dim3)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    length_ = dim1_ * dim2_ * dim3_;
    this_array_ = new T[length_];
}

//4D
template <typename T>
CArray<T>::CArray(size_t some_dim1,
                  size_t some_dim2,
                  size_t some_dim3,
                  size_t some_dim4)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    length_ = dim1_ * dim2_ * dim3_ * dim4_;
    this_array_ = new T[length_];
}

//5D
template <typename T>
CArray<T>::CArray(size_t some_dim1,
                  size_t some_dim2,
                  size_t some_dim3,
                  size_t some_dim4,
                  size_t some_dim5) {
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    length_ = dim1_ * dim2_ * dim3_ * dim4_ * dim5_;
    this_array_ = new T[length_];
}

//6D
template <typename T>
CArray<T>::CArray(size_t some_dim1,
                  size_t some_dim2,
                  size_t some_dim3,
                  size_t some_dim4,
                  size_t some_dim5,
                  size_t some_dim6) {
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    length_ = dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_;
    this_array_ = new T[length_];
}

//7D
template <typename T>
CArray<T>::CArray(size_t some_dim1,
                  size_t some_dim2,
                  size_t some_dim3,
                  size_t some_dim4,
                  size_t some_dim5,
                  size_t some_dim6,
                  size_t some_dim7) {
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    dim7_ = some_dim7;
    length_ = dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_ * dim7_;
    this_array_ = new T[length_];
}



//overload () operator

//1D
template <typename T>
inline T& CArray<T>::operator() (size_t i) const {
    assert(i < dim1_ && "i is out of bounds in CArray 1D");  // die if >= dim1
    return this_array_[i];
}

//2D
template <typename T>
inline T& CArray<T>::operator() (size_t i,
                                 size_t j) const
{
    assert(i < dim1_ && "i is out of bounds in CArray 2D");  // die if >= dim1
    assert(j < dim2_ && "j is out of bounds in CArray 2D");  // die if >= dim2
    
    return this_array_[j + (i * dim2_)];
}

//3D
template <typename T>
inline T& CArray<T>::operator() (size_t i,
                                 size_t j,
                                 size_t k) const
{
    assert(i < dim1_ && "i is out of bounds in CArray 3D");  // die if >= dim1
    assert(j < dim2_ && "j is out of bounds in CArray 3D");  // die if >= dim2
    assert(k < dim3_ && "k is out of bounds in CArray 3D");  // die if >= dim3
    
    return this_array_[k + (j * dim3_) + (i * dim3_ * dim2_)];
}

//4D
template <typename T>
inline T& CArray<T>::operator() (size_t i,
                                 size_t j,
                                 size_t k,
                                 size_t l) const
{
    assert(i < dim1_ && "i is out of bounds in CArray 4D");  // die if >= dim1
    assert(j < dim2_ && "j is out of bounds in CArray 4D");  // die if >= dim2
    assert(k < dim3_ && "k is out of bounds in CArray 4D");  // die if >= dim3
    assert(l < dim4_ && "l is out of bounds in CArray 4D");  // die if >= dim4

    
    return this_array_[l + (k * dim4_) 
                         + (j * dim4_ * dim3_)  
                         + (i * dim4_ * dim3_ * dim2_)];
}

//5D
template <typename T>
inline T& CArray<T>::operator() (size_t i,
                                 size_t j,
                                 size_t k,
                                 size_t l,
                                 size_t m) const
{
    assert(i < dim1_ && "i is out of bounds in CArray 5D");  // die if >= dim1
    assert(j < dim2_ && "j is out of bounds in CArray 5D");  // die if >= dim2
    assert(k < dim3_ && "k is out of bounds in CArray 5D");  // die if >= dim3
    assert(l < dim4_ && "l is out of bounds in CArray 5D");  // die if >= dim4
    assert(m < dim5_ && "m is out of bounds in CArray 5D");  // die if >= dim5
    
    return this_array_[m + (l * dim5_) 
                         + (k * dim5_ * dim4_) 
                         + (j * dim5_ * dim4_ * dim3_) 
                         + (i * dim5_ * dim4_ * dim3_ * dim2_)];
}

//6D
template <typename T>
inline T& CArray<T>::operator() (size_t i,
                                 size_t j,
                                 size_t k,
                                 size_t l,
                                 size_t m,
                                 size_t n) const
{
    assert(i < dim1_ && "i is out of bounds in CArray 6D");  // die if >= dim1
    assert(j < dim2_ && "j is out of bounds in CArray 6D");  // die if >= dim2
    assert(k < dim3_ && "k is out of bounds in CArray 6D");  // die if >= dim3
    assert(l < dim4_ && "l is out of bounds in CArray 6D");  // die if >= dim4
    assert(m < dim5_ && "m is out of bounds in CArray 6D");  // die if >= dim5
    assert(n < dim6_ && "n is out of bounds in CArray 6D");  // die if >= dim6
    
    return this_array_[n + (m * dim6_) 
                         + (l * dim6_ * dim5_)  
                         + (k * dim6_ * dim5_ * dim4_) 
                         + (j * dim6_ * dim5_ * dim4_ * dim3_)  
                         + (i * dim6_ * dim5_ * dim4_ * dim3_ * dim2_)];

}

//7D
template <typename T>
inline T& CArray<T>::operator() (size_t i,
                                 size_t j,
                                 size_t k,
                                 size_t l,
                                 size_t m,
                                 size_t n,
                                 size_t o) const
{
    assert(i < dim1_ && "i is out of bounds in CArray 7D");  // die if >= dim1
    assert(j < dim2_ && "j is out of bounds in CArray 7D");  // die if >= dim2
    assert(k < dim3_ && "k is out of bounds in CArray 7D");  // die if >= dim3
    assert(l < dim4_ && "l is out of bounds in CArray 7D");  // die if >= dim4
    assert(m < dim5_ && "m is out of bounds in CArray 7D");  // die if >= dim5
    assert(n < dim6_ && "n is out of bounds in CArray 7D");  // die if >= dim6
    assert(o < dim7_ && "o is out of bounds in CArray 7D");  // die if >= dim7
    
    return this_array_[o + (n * dim7_)
                         + (m * dim7_ * dim6_)
                         + (l * dim7_ * dim6_ * dim5_)
                         + (k * dim7_ * dim6_ * dim5_ * dim4_)
                         + (j * dim7_ * dim6_ * dim5_ * dim4_ * dim3_)
                         + (i * dim7_ * dim6_ * dim5_ * dim4_ * dim3_ * dim2_)];
    
}


//overload = operator
template <typename T>
inline CArray<T>& CArray<T>::operator= (const CArray& temp)
{
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        dim1_ = temp.dim1_;
        dim2_ = temp.dim2_;
        dim3_ = temp.dim3_;
        dim4_ = temp.dim4_;
        dim5_ = temp.dim5_;
        dim6_ = temp.dim6_;
        dim7_ = temp.dim7_;
        length_ = temp.length_;
        this_array_ = new T[length_];
    }
    return *this;
}

//return size
template <typename T>
inline size_t CArray<T>::size() const {
    return length_;
}

template <typename T>
inline T* CArray<T>::get_pointer() const{
    return this_array_;
}

//destructor
template <typename T>
CArray<T>::~CArray() {
    delete[] this_array_;
}

//----endof carray class definitions----


//6. ViewCArray
// indicies are [0:N-1]
template <typename T>
class ViewCArray {

private:
    size_t dim1_, dim2_, dim3_, dim4_, dim5_, dim6_, dim7_;
    size_t length_; // Length of 1D array
    T * this_array_;
    
public:
    
    // Default constructor
    ViewCArray ();
    
    //--- 1D to 7D array ---
    ViewCArray(T *some_array,
               size_t some_dim1);

    ViewCArray(T *some_array,
               size_t some_dim1,
               size_t some_dim2);
    
    ViewCArray(T *some_array,
               size_t some_dim1,
               size_t some_dim2,
               size_t some_dim3);
    
    ViewCArray(T *some_array,
               size_t some_dim1,
               size_t some_dim2,
               size_t some_dim3,
               size_t some_dim4);
    
    ViewCArray (T *some_array,
                size_t some_dim1,
                size_t some_dim2,
                size_t some_dim3,
                size_t some_dim4,
                size_t some_dim5);

    ViewCArray (T *some_array,
                size_t some_dim1,
                size_t some_dim2,
                size_t some_dim3,
                size_t some_dim4,
                size_t some_dim5,
                size_t some_dim6);
 
    ViewCArray (T *some_array,
                size_t some_dim1,
                size_t some_dim2,
                size_t some_dim3,
                size_t some_dim4,
                size_t some_dim5,
                size_t some_dim6,
                size_t some_dim7);
    
    T& operator()(size_t i) const;
    
    T& operator()(size_t i,
                  size_t j) const;
    
    T& operator()(size_t i,
                  size_t j,
                  size_t k) const;
    
    T& operator()(size_t i,
                  size_t j,
                  size_t k,
                  size_t l) const;
    T& operator()(size_t i,
                  size_t j,
                  size_t k,
                  size_t l,
                  size_t m) const;
    
    T& operator()(size_t i,
                  size_t j,
                  size_t k,
                  size_t l,
                  size_t m,
                  size_t n) const;
    
    T& operator()(size_t i,
                  size_t j,
                  size_t k,
                  size_t l,
                  size_t m,
                  size_t n,
                  size_t o) const;
    
    size_t size() const;
    
}; // end of ViewCArray

//class definitions

//constructors

//no dim
template <typename T>
ViewCArray<T>::ViewCArray() {}

//1D
template <typename T>
ViewCArray<T>::ViewCArray(T *some_array,
                          size_t some_dim1)
{
    dim1_ = some_dim1;
    length_ = dim1_;
    this_array_ = some_array;
}

//2D
template <typename T>
ViewCArray<T>::ViewCArray(T *some_array,
                          size_t some_dim1,
                          size_t some_dim2)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    length_ = dim1_ * dim2_;
    this_array_ = some_array;
}

//3D
template <typename T>
ViewCArray<T>::ViewCArray (T *some_array,
                           size_t some_dim1,
                           size_t some_dim2,
                           size_t some_dim3)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    length_ = dim1_ * dim2_ * dim3_;
    this_array_ = some_array;
}

//4D
template <typename T>
ViewCArray<T>::ViewCArray(T *some_array,
                          size_t some_dim1,
                          size_t some_dim2,
                          size_t some_dim3,
                          size_t some_dim4)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_);
    this_array_ = some_array;
}

//5D
template <typename T>
ViewCArray<T>::ViewCArray(T *some_array,
                          size_t some_dim1,
                          size_t some_dim2,
                          size_t some_dim3,
                          size_t some_dim4,
                          size_t some_dim5)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_);
    this_array_ = some_array;
}

//6D
template <typename T>
ViewCArray<T>::ViewCArray(T *some_array,
                          size_t some_dim1,
                          size_t some_dim2,
                          size_t some_dim3,
                          size_t some_dim4,
                          size_t some_dim5,
                          size_t some_dim6)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    length_ = dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_;
    this_array_ = some_array;
}

//7D
template <typename T>
ViewCArray<T>::ViewCArray(T *some_array,
                          size_t some_dim1,
                          size_t some_dim2,
                          size_t some_dim3,
                          size_t some_dim4,
                          size_t some_dim5,
                          size_t some_dim6,
                          size_t some_dim7)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    dim7_ = some_dim7;
    length_ = dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_ * dim7_;
    this_array_ = some_array;
}

//overload () operator

//1D
template <typename T>
inline T& ViewCArray<T>::operator()(size_t i) const
{
    assert(i < dim1_ && "i is out of bounds in c_array 1D");  // die if >= dim1
    
    return this_array_[i];
}

//2D
template <typename T>
inline T& ViewCArray<T>::operator()(size_t i, 
                                    size_t j) const
{
   
    assert(i < dim1_ && "i is out of bounds in ViewCArray 2D");  // die if >= dim1
    assert(j < dim2_ && "j is out of bounds in ViewCArray 2D");  // die if >= dim2
    
    return this_array_[j + (i * dim2_)];
}

//3D
template <typename T>
inline T& ViewCArray<T>::operator()(size_t i, 
                                    size_t j, 
                                    size_t k) const
{
    assert(i < dim1_ && "i is out of bounds in ViewCArray 3D");  // die if >= dim1
    assert(j < dim2_ && "j is out of bounds in ViewCArray 3D");  // die if >= dim2
    assert(k < dim3_ && "k is out of bounds in ViewCArray 3D");  // die if >= dim3
    
    return this_array_[k + (j * dim3_) 
                         + (i * dim3_ * dim2_)];
}

//4D
template <typename T>
inline T& ViewCArray<T>::operator()(size_t i, 
                                    size_t j, 
                                    size_t k, 
                                    size_t l) const
{
    assert(i < dim1_ && "i is out of bounds in ViewCArray 4D");  // die if >= dim1
    assert(j < dim2_ && "j is out of bounds in ViewCArray 4D");  // die if >= dim2
    assert(k < dim3_ && "k is out of bounds in ViewCArray 4D");  // die if >= dim3
    assert(l < dim4_ && "l is out of bounds in ViewCArray 4D");  // die if >= dim4
    
    return this_array_[l + (k * dim4_) 
                         + (j * dim4_ * dim3_) 
                         + (i * dim4_ * dim3_ * dim2_)];
}

//5D
template <typename T>
inline T& ViewCArray<T>::operator()(size_t i, 
                                    size_t j, 
                                    size_t k, 
                                    size_t l, 
                                    size_t m) const
{
    assert(i < dim1_ && "i is out of bounds in ViewCArray 5D");  // die if >= dim1
    assert(j < dim2_ && "j is out of bounds in ViewCArray 5D");  // die if >= dim2
    assert(k < dim3_ && "k is out of bounds in ViewCArray 5D");  // die if >= dim3
    assert(l < dim4_ && "l is out of bounds in ViewCArray 5D");  // die if >= dim4
    assert(m < dim5_ && "m is out of bounds in ViewCArray 5D");  // die if >= dim5
    
    return this_array_[m + (l * dim5_) 
                         + (k * dim5_ * dim4_) 
                         + (j * dim5_ * dim4_ * dim3_)
                         + (i * dim5_ * dim4_ * dim3_ * dim2_)];
}

//6D
template <typename T>
inline T& ViewCArray<T>::operator()(size_t i,
                                    size_t j,
                                    size_t k,
                                    size_t l,
                                    size_t m,
                                    size_t n) const
{
    assert(i < dim1_ && "i is out of bounds in ViewCArray 6D");  // die if >= dim1
    assert(j < dim2_ && "j is out of bounds in ViewCArray 6D");  // die if >= dim2
    assert(k < dim3_ && "k is out of bounds in ViewCArray 6D");  // die if >= dim3
    assert(l < dim4_ && "l is out of bounds in ViewCArray 6D");  // die if >= dim4
    assert(m < dim5_ && "m is out of bounds in ViewCArray 6D");  // die if >= dim5
    assert(n < dim6_ && "n is out of bounds in ViewCArray 6D");  // die if >= dim6
    
    return this_array_[n + (m * dim6_) 
                         + (l * dim6_ * dim5_) 
                         + (k * dim6_ * dim5_ * dim4_)
                         + (j * dim6_ * dim5_ * dim4_ * dim3_) 
                         + (i * dim6_ * dim5_ * dim4_ * dim3_ * dim2_)];
}

//7D
template <typename T>
inline T& ViewCArray<T>::operator()(size_t i,
                                    size_t j,
                                    size_t k,
                                    size_t l,
                                    size_t m,
                                    size_t n,
                                    size_t o) const
{
    assert(i < dim1_ && "i is out of bounds in ViewCArray 7D");  // die if >= dim1
    assert(j < dim2_ && "j is out of bounds in ViewCArray 7D");  // die if >= dim2
    assert(k < dim3_ && "k is out of bounds in ViewCArray 7D");  // die if >= dim3
    assert(l < dim4_ && "l is out of bounds in ViewCArray 7D");  // die if >= dim4
    assert(m < dim5_ && "m is out of bounds in ViewCArray 7D");  // die if >= dim5
    assert(n < dim6_ && "n is out of bounds in ViewCArray 7D");  // die if >= dim6
    assert(o < dim7_ && "o is out of bounds in ViewCArray 7D");  // die if >= dim7
    
    return this_array_[o + (n * dim7_)
                         + (m * dim7_ * dim6_)
                         + (l * dim7_ * dim6_ * dim5_)
                         + (k * dim7_ * dim6_ * dim5_ * dim4_)
                         + (j * dim7_ * dim6_ * dim5_ * dim4_ * dim3_)
                         + (i * dim7_ * dim6_ * dim5_ * dim4_ * dim3_ * dim2_)];
}


//return size    
template <typename T>
inline size_t ViewCArray<T>::size() const {
    return length_;
}

//---end of ViewCArray class definitions----


//7. CMatrix
template <typename T>
class CMatrix {
        
private:
    size_t dim1_, dim2_, dim3_, dim4_, dim5_, dim6_, dim7_, length_;
    T * this_matrix;
            
public:
        
       // default constructor
       CMatrix();
    
       CMatrix(size_t some_dim1);
    
       CMatrix(size_t some_dim1,
               size_t some_dim2);
    
       CMatrix(size_t some_dim1,
               size_t some_dim2,
               size_t some_dim3);
    
       CMatrix(size_t some_dim1,
               size_t some_dim2,
               size_t some_dim3,
               size_t some_dim4);
    
       CMatrix(size_t some_dim1,
               size_t some_dim2,
               size_t some_dim3,
               size_t some_dim4,
               size_t some_dim5);
    
       CMatrix (size_t some_dim1,
                size_t some_dim2,
                size_t some_dim3,
                size_t some_dim4,
                size_t some_dim5,
                size_t some_dim6);
 
       CMatrix (size_t some_dim1,
                size_t some_dim2,
                size_t some_dim3,
                size_t some_dim4,
                size_t some_dim5,
                size_t some_dim6,
                size_t some_dim7);
    
       //overload operators to access data
       T& operator()(size_t i) const;
    
       T& operator()(size_t i,
                     size_t j) const;
    
       T& operator()(size_t i,
                     size_t j,
                     size_t k) const;
    
       T& operator()(size_t i,
                     size_t j,
                     size_t k,
                     size_t l) const;
    
       T& operator()(size_t i,
                     size_t j,
                     size_t k,
                     size_t l,
                     size_t m) const;
    
       T& operator()(size_t i,
                     size_t j,
                     size_t k,
                     size_t l,
                     size_t m,
                     size_t n) const;
    
       T& operator()(size_t i,
                     size_t j,
                     size_t k,
                     size_t l,
                     size_t m,
                     size_t n,
                     size_t o) const;
    
       //overload = operator
	   CMatrix& operator= (const CMatrix &temp);
    
       size_t size() const;

       //return pointer
       T* get_pointer() const;
    
       // deconstructor
       ~CMatrix( );
        
}; // end of CMatrix

// CMatrix class definitions

//constructors

//no dim

//1D
template <typename T>
CMatrix<T>::CMatrix() {
    this_matrix = NULL;
}

//1D
template <typename T>
CMatrix<T>::CMatrix(size_t some_dim1)
{
    dim1_ = some_dim1;
    length_ = dim1_;
    this_matrix = new T[length_];
}

//2D
template <typename T>
CMatrix<T>::CMatrix(size_t some_dim1,
                    size_t some_dim2)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    length_ = dim1_*dim2_;
    this_matrix = new T[length_];
}

//3D
template <typename T>
CMatrix<T>::CMatrix(size_t some_dim1,
                    size_t some_dim2,
                    size_t some_dim3)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    length_ = dim1_*dim2_*dim3_;
    this_matrix = new T[length_];
}

//4D
template <typename T>
CMatrix<T>::CMatrix(size_t some_dim1,
                    size_t some_dim2,
                    size_t some_dim3,
                    size_t some_dim4)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    length_ = dim1_*dim2_*dim3_*dim4_;
    this_matrix= new T[length_];
}   

//5D
template <typename T>
CMatrix<T>::CMatrix(size_t some_dim1,
                    size_t some_dim2,
                    size_t some_dim3,
                    size_t some_dim4,
                    size_t some_dim5)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    length_ = dim1_*dim2_*dim3_*dim4_*dim5_;
    this_matrix = new T[length_];
}

//6D
template <typename T>
CMatrix<T>::CMatrix(size_t some_dim1,
                    size_t some_dim2,
                    size_t some_dim3,
                    size_t some_dim4,
                    size_t some_dim5,
                    size_t some_dim6)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    length_ = dim1_*dim2_*dim3_*dim4_*dim5_*dim6_;
    this_matrix = new T[length_];
}

//7D
template <typename T>
CMatrix<T>::CMatrix(size_t some_dim1,
                    size_t some_dim2,
                    size_t some_dim3,
                    size_t some_dim4,
                    size_t some_dim5,
                    size_t some_dim6,
                    size_t some_dim7)
{
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    dim7_ = some_dim7;
    length_ = dim1_*dim2_*dim3_*dim4_*dim5_*dim6_*dim7_;
    this_matrix = new T[length_];
}

//overload () operator

//1D
template <typename T>
T& CMatrix<T>::operator()(size_t i) const
{
    assert( i < dim1_+1 && "i is out of bounds in CMatrix 1D!");
    return this_matrix[i-1];
}

//2D
template <typename T>
T& CMatrix<T>::operator()(size_t i,
                          size_t j) const
{
    assert( i < dim1_+1 && "i is out of bounds in CMatrix 2D!");
    assert( j < dim2_+1 && "j is out of bounds in CMatrix 2D!");
    return this_matrix[(j-1) + (i-1)*dim2_];
}

//3D
template <typename T>
T& CMatrix<T>::operator()(size_t i,
                          size_t j,
                          size_t k) const
{
    assert( i < dim1_+1 && "i is out of bounds in CMatrix 3D!");
    assert( j < dim2_+1 && "j is out of bounds in CMatrix 3D!");
    assert( k < dim3_+1 && "k is out of bounds in CMatrix 3D!");
    return this_matrix[(k-1) + (j-1)*dim3_
                             + (i-1)*dim3_*dim2_];
}

//4D
template <typename T>
T& CMatrix<T>::operator()(size_t i,
                          size_t j,
                          size_t k,
                          size_t l) const
{
    assert( i < dim1_+1 && "i is out of bounds in CMatrix 4D!");
    assert( j < dim2_+1 && "j is out of bounds in CMatrix 4D!");
    assert( k < dim3_+1 && "k is out of bounds in CMatrix 4D!");
    assert( l < dim4_+1 && "l is out of bounds in CMatrix 4D!");
    return this_matrix[ (l-1) + (k-1)*dim4_
                              + (j-1)*dim4_*dim3_
                              + (i-1)*dim4_*dim3_*dim2_];
}

//5D
template <typename T>
T& CMatrix<T>::operator()(size_t i,
                          size_t j,
                          size_t k,
                          size_t l,
                          size_t m) const
{
    assert( i < dim1_+1 && "i is out of bounds in CMatrix 5D!");
    assert( j < dim2_+1 && "j is out of bounds in CMatrix 5D!");
    assert( k < dim3_+1 && "k is out of bounds in CMatrix 5D!");
    assert( l < dim4_+1 && "l is out of bounds in CMatrix 5D!");
    assert( m < dim5_+1 && "m is out of bounds in CMatrix 5D!");
    return this_matrix[(m-1) + (l-1)*dim5_
                             + (k-1)*dim5_*dim4_
                             + (j-1)*dim5_*dim4_*dim3_
                             + (i-1)*dim5_*dim4_*dim3_*dim2_];
}

//6D
template <typename T>
T& CMatrix<T>::operator()(size_t i,
                          size_t j,
                          size_t k,
                          size_t l,
                          size_t m,
                          size_t n) const
{
    assert( i < dim1_+1 && "i is out of bounds in CMatrix 6D!");
    assert( j < dim2_+1 && "j is out of bounds in CMatrix 6D!");
    assert( k < dim3_+1 && "k is out of bounds in CMatrix 6D!");
    assert( l < dim4_+1 && "l is out of bounds in CMatrix 6D!");
    assert( m < dim5_+1 && "m is out of bounds in CMatrix 6D!");
    assert( n < dim6_+1 && "n is out of bounds in CMatrix 6D!");
    return this_matrix[ (n-1) + (m-1)*dim6_
                              + (l-1)*dim6_*dim5_
                              + (k-1)*dim6_*dim5_*dim4_
                              + (j-1)*dim6_*dim5_*dim4_*dim3_
                              + (i-1)*dim6_*dim5_*dim4_*dim3_*dim2_];
}

//7D
template <typename T>
T& CMatrix<T>::operator()(size_t i,
                          size_t j,
                          size_t k,
                          size_t l,
                          size_t m,
                          size_t n,
                          size_t o) const
{
    assert( i < dim1_+1 && "i is out of bounds in CMatrix 7D!");
    assert( j < dim2_+1 && "j is out of bounds in CMatrix 7D!");
    assert( k < dim3_+1 && "k is out of bounds in CMatrix 7D!");
    assert( l < dim4_+1 && "l is out of bounds in CMatrix 7D!");
    assert( m < dim5_+1 && "m is out of bounds in CMatrix 7D!");
    assert( n < dim6_+1 && "n is out of bounds in CMatrix 7D!");
    assert( o < dim7_+1 && "n is out of bounds in CMatrix 7D!");
    
    return this_matrix[(o-1) + (n-1)*dim7_
                             + (m-1)*dim7_*dim6_
                             + (l-1)*dim7_*dim6_*dim5_
                             + (k-1)*dim7_*dim6_*dim5_*dim4_
                             + (j-1)*dim7_*dim6_*dim5_*dim4_*dim3_
                             + (i-1)*dim7_*dim6_*dim5_*dim4_*dim3_*dim2_];
}

//overload = operator
//THIS = CMatrix<> temp
template <typename T>
CMatrix<T> &CMatrix<T>::operator= (const CMatrix &temp) {
	if(this != &temp) {
        dim1_ = temp.dim1_;
        dim2_ = temp.dim2_;
        dim3_ = temp.dim3_;
        dim4_ = temp.dim4_;
        dim5_ = temp.dim5_;
        dim6_ = temp.dim6_;
        dim7_ = temp.dim7_;
        length_ = temp.length_;
        this_matrix = new T[length_];
	}
  return *this;
}

template <typename T>
inline size_t CMatrix<T>::size() const {
    return length_;
}

template <typename T>
inline T* CMatrix<T>::get_pointer() const{
    return this_matrix;
}

// Destructor
template <typename T>
CMatrix<T>::~CMatrix(){
    delete[] this_matrix;
}

//----end of CMatrix class definitions----


//8. ViewCMatrix
//  indices [1:N]
template <typename T>
class ViewCMatrix {

private:
    size_t dim1_, dim2_, dim3_, dim4_, dim5_, dim6_, dim7_;
     T * this_matrix;
		    
public:
		    
    // default constructor
    ViewCMatrix();
		    
		    
    //--- 1D array ---	   	    
    // overloaded constructor
    ViewCMatrix (T *some_matrix,
                 size_t some_dim1);
    
    ViewCMatrix (T *some_matrix,
                 size_t some_dim1,
                 size_t some_dim2);

    ViewCMatrix (T *some_matrix,
		size_t some_dim1,
		size_t some_dim2,
		size_t some_dim3);

    ViewCMatrix (T *some_matrix,
		size_t some_dim1,
		size_t some_dim2,
		size_t some_dim3,
		size_t some_dim4);

    ViewCMatrix (T *some_matrix,
		size_t some_dim1,
		size_t some_dim2,
		size_t some_dim3,
		size_t some_dim4,
		size_t some_dim5);

    ViewCMatrix (T *some_matrix,
		   size_t some_dim1,
		   size_t some_dim2,
		   size_t some_dim3,
		   size_t some_dim4,
		   size_t some_dim5,
		   size_t some_dim6);

    ViewCMatrix (T *some_matrix,
                 size_t some_dim1,
                 size_t some_dim2,
                 size_t some_dim3,
                 size_t some_dim4,
                 size_t some_dim5,
                 size_t some_dim6,
                 size_t some_dim7);
    
    T& operator() (size_t i) const;
    
    T& operator() (size_t i,
                   size_t j) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l,
                   size_t m) const;
    
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l,
                   size_t m,
                   size_t n) const;
    T& operator() (size_t i,
                   size_t j,
                   size_t k,
                   size_t l,
                   size_t m,
                   size_t n,
                   size_t o) const;
		    
}; // end of ViewCMatrix

//class definitions

//constructors

//no dim
template <typename T>
ViewCMatrix<T>::ViewCMatrix(){}

//1D
template <typename T>
ViewCMatrix<T>::ViewCMatrix(T *some_matrix,
                            size_t some_dim1)
{
	dim1_ = some_dim1;
	this_matrix = some_matrix;
}

//2D
template <typename T>
ViewCMatrix<T>::ViewCMatrix(T *some_matrix,
                            size_t some_dim1,
                            size_t some_dim2)
{
	dim1_ = some_dim1;
	dim2_ = some_dim2;
	this_matrix = some_matrix;
}

//3D
template <typename T>
ViewCMatrix<T>::ViewCMatrix(T *some_matrix,
                            size_t some_dim1,
                            size_t some_dim2,
                            size_t some_dim3)
{
	dim1_ = some_dim1;
	dim2_ = some_dim2;
	dim3_ = some_dim3;
	this_matrix = some_matrix;
}

//4D
template <typename T>
ViewCMatrix<T>::ViewCMatrix(T *some_matrix,
                            size_t some_dim1,
                            size_t some_dim2,
                            size_t some_dim3,
                            size_t some_dim4)
{
	dim1_ = some_dim1;
	dim2_ = some_dim2;
	dim3_ = some_dim3;
	dim4_ = some_dim4;
	this_matrix = some_matrix;
}

//5D
template <typename T>
ViewCMatrix<T>::ViewCMatrix(T *some_matrix,
                            size_t some_dim1,
                            size_t some_dim2,
                            size_t some_dim3,
                            size_t some_dim4,
                            size_t some_dim5)
{
	dim1_ = some_dim1;
	dim2_ = some_dim2;
	dim3_ = some_dim3;
	dim4_ = some_dim4;
	dim5_ = some_dim5;
	this_matrix = some_matrix;
}

//6D
template <typename T>
ViewCMatrix<T>::ViewCMatrix(T *some_matrix,
                            size_t some_dim1,
                            size_t some_dim2,
                            size_t some_dim3,
                            size_t some_dim4,
                            size_t some_dim5,
                            size_t some_dim6) {
	dim1_ = some_dim1;
	dim2_ = some_dim2;
	dim3_ = some_dim3;
	dim4_ = some_dim4;
	dim5_ = some_dim5;
	dim6_ = some_dim6;
	this_matrix = some_matrix;
}

//7D
template <typename T>
ViewCMatrix<T>::ViewCMatrix(T *some_matrix,
                            size_t some_dim1,
                            size_t some_dim2,
                            size_t some_dim3,
                            size_t some_dim4,
                            size_t some_dim5,
                            size_t some_dim6,
                            size_t some_dim7) {
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    dim7_ = some_dim7;
    this_matrix = some_matrix;
}

//overload () operator

//1D
template <typename T>
T& ViewCMatrix<T>:: operator() (size_t i) const
{
	assert(i < dim1_+1 && "i is out of bounds for ViewCMatrix 1D!");
	return this_matrix[i-1];
}

//2D
template <typename T>
T& ViewCMatrix<T>::operator() (size_t i,
                               size_t j) const
{
	assert(i < dim1_+1 && "i is out of bounds for ViewCMatrix 2D!");
	assert(j < dim2_+1 && "j is out of bounds for ViewCMatrix 2D!");
	return this_matrix[(i-1)*dim2_ + (j-1)];
}

//3D
template <typename T>
T& ViewCMatrix<T>::operator () (size_t i,
                                size_t j,
                                size_t k) const
{
	assert(i < dim1_+1 && "i is out of bounds for ViewCMatrix 3D!");
	assert(j < dim2_+1 && "j is out of bounds for ViewCMatrix 3D!");
	assert(k < dim3_+1 && "k is out of bounds for ViewCMatrix 3D!");
	return this_matrix[(k-1) + (j-1)*dim3_
                             + (i-1)*dim3_*dim2_];
}

//4D
template <typename T>
T& ViewCMatrix<T>::operator()(size_t i,
                              size_t j,
                              size_t k,
                              size_t l) const
{
	assert(i < dim1_+1 && "i is out of bounds for ViewCMatrix 4D!");
	assert(j < dim2_+1 && "j is out of bounds for ViewCMatrix 4D!");
	assert(k < dim3_+1 && "k is out of bounds for ViewCMatrix 4D!");
	assert(l < dim4_+1 && "l is out of bounds for ViewCMatrix 4D!");
	return this_matrix[(l-1) + (k-1)*dim4_
                             + (j-1)*dim4_*dim3_
                             + (i-1)*dim4_*dim3_*dim2_];
}

//5D
template <typename T>
T& ViewCMatrix<T>::operator()(size_t i,
                              size_t j,
                              size_t k,
                              size_t l,
                              size_t m) const
{
	assert(i < dim1_+1 && "i is out of bounds for ViewCMatrix 5D!");
	assert(j < dim2_+1 && "j is out of bounds for ViewCMatrix 5D!");
	assert(k < dim3_+1 && "k is out of bounds for ViewCMatrix 5D!");
	assert(l < dim4_+1 && "l is out of bounds for ViewCMatrix 5D!");
	assert(m < dim5_+1 && "m is out of bounds for ViewCMatrix 5D!");
	return this_matrix[(m-1) + (l-1)*dim5_
                             + (k-1)*dim5_*dim4_
                             + (j-1)*dim5_*dim4_*dim3_
                             + (i-1)*dim5_*dim4_*dim3_*dim2_];
}

//6D
template <typename T>
T& ViewCMatrix<T>::operator()(size_t i,
                              size_t j,
                              size_t k,
                              size_t l,
                              size_t m,
                              size_t n) const
{
	assert(i < dim1_+1 && "i is out of bounds for ViewCMatrix 6D!");
	assert(j < dim2_+1 && "j is out of bounds for ViewCMatrix 6D!");
	assert(k < dim3_+1 && "k is out of bounds for ViewCMatrix 6D!");
	assert(l < dim4_+1 && "l is out of bounds for ViewCMatrix 6D!");
	assert(m < dim5_+1 && "m is out of bounds for ViewCMatrix 6D!");
	assert(n < dim6_+1 && "n is out of bounds for ViewCMatrix 6D!");
	return this_matrix[(n-1) + (m-1)*dim6_
                             + (l-1)*dim5_*dim6_
                             + (k-1)*dim6_*dim5_*dim4_
                             + (j-1)*dim6_*dim5_*dim4_*dim3_
                             + (i-1)*dim5_*dim6_*dim4_*dim3_*dim2_];
}

//7D
template <typename T>
T& ViewCMatrix<T>::operator()(size_t i,
                              size_t j,
                              size_t k,
                              size_t l,
                              size_t m,
                              size_t n,
                              size_t o) const
{
    assert(i < dim1_+1 && "i is out of bounds for ViewCMatrix 7D!");
    assert(j < dim2_+1 && "j is out of bounds for ViewCMatrix 7D!");
    assert(k < dim3_+1 && "k is out of bounds for ViewCMatrix 7D!");
    assert(l < dim4_+1 && "l is out of bounds for ViewCMatrix 7D!");
    assert(m < dim5_+1 && "m is out of bounds for ViewCMatrix 7D!");
    assert(n < dim6_+1 && "n is out of bounds for ViewCMatrix 7D!");
    assert(o < dim7_+1 && "o is out of bounds for ViewCMatrix 7D!");
    return this_matrix[(o-1) + (n-1)*dim7_
                             + (m-1)*dim7_*dim6_
                             + (l-1)*dim7_*dim5_*dim6_
                             + (k-1)*dim7_*dim6_*dim5_*dim4_
                             + (j-1)*dim7_*dim6_*dim5_*dim4_*dim3_
                             + (i-1)*dim7_*dim5_*dim6_*dim4_*dim3_*dim2_];
}


//----end of ViewCMatrix class definitions----

//9. RaggedRightArray
template <typename T>
class RaggedRightArray {
private:
    size_t *start_index_;
    T * array_;
    
    size_t dim1_, length_;
    size_t num_saved_; // the number saved in the 1D array
    
public:
    // Default constructor
    RaggedRightArray ();
    
    //--- 2D array access of a ragged right array ---
    
    // Overload constructor for a CArray
    RaggedRightArray (CArray<size_t> &strides_array);
    
    // Overload constructor for a ViewCArray
    RaggedRightArray (ViewCArray<size_t> &strides_array);
    
    // Overloaded constructor for a traditional array
    RaggedRightArray (size_t *strides_array, size_t some_dim1);
    
    // Overload constructor for a RaggedRightArray to
    // support a dynamically built stride_array
    RaggedRightArray (size_t some_dim1, size_t buffer);
    
    // A method to return the stride size
    size_t stride(size_t i) const;
    
    // A method to increase the number of column entries, i.e.,
    // the stride size. Used with the constructor for building
    // the stride_array dynamically.
    // DO NOT USE with the constructures with a strides_array
    void push_back(size_t i);
    
    // Overload operator() to access data as array(i,j)
    // where i=[0:N-1], j=[stride(i)]
    T& operator()(size_t i, size_t j) const;

    // method to return total size
    size_t size() const;

    //return pointer
    T* get_pointer() const;
    
    //get row starts array
    size_t* get_starts() const;

    RaggedRightArray& operator+= (const size_t i);

    RaggedRightArray& operator= (const RaggedRightArray &temp);

    // Destructor
    ~RaggedRightArray ( );
}; // End of RaggedRightArray

// Default constructor
template <typename T>
RaggedRightArray<T>::RaggedRightArray () {
    array_ = NULL;
}


// Overloaded constructor with CArray
template <typename T>
RaggedRightArray<T>::RaggedRightArray (CArray<size_t> &strides_array){
    // The length of the stride array is some_dim1;
    dim1_  = strides_array.size();
    
    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[(dim1_ + 1)];  // note the dim1+1
    start_index_[0] = 0; // the 1D array starts at 0
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array and set the starting 1D index
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += strides_array(i);
        start_index_[(i + 1)] = count;
    } // end for i
    length_ = count;
    
    array_ = new T[length_];
} // End constructor

// Overloaded constructor with a view c array
template <typename T>
RaggedRightArray<T>::RaggedRightArray (ViewCArray<size_t> &strides_array) {
    // The length of the stride array is some_dim1;
    dim1_  = strides_array.size();
    
    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[(dim1_ + 1)];  // note the dim1+1
    start_index_[0] = 0; // the 1D array starts at 0
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array and set the starting 1D index
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += strides_array(i);
        start_index_[(i + 1)] = count;
    } // end for i
    length_ = count;
    
    array_ = new T[length_];
} // End constructor

// Overloaded constructor with a regular cpp array
template <typename T>
RaggedRightArray<T>::RaggedRightArray (size_t *strides_array, size_t dim1){
    // The length of the stride array is some_dim1;
    dim1_ = dim1;
    
    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[(dim1_ + 1)];  // note the dim1+1
    start_index_[0] = 0; // the 1D array starts at 0
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array and set the starting 1D index
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += strides_array[i];
        start_index_[(i + 1)] = count;
    } // end for i
    length_ = count;
    
    array_ = new T[length_];
} // End constructor

// overloaded constructor for a dynamically built strides_array.
// buffer is the max number of columns needed
template <typename T>
RaggedRightArray<T>::RaggedRightArray (size_t some_dim1, size_t buffer){
    
    dim1_ = some_dim1;
    
    // create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[dim1_+1]();  // note the dim1+1
    //start_index_[0] = 0; // the 1D array starts at 0

    num_saved_ = 0;
    
    length_ = some_dim1*buffer;
    array_ = new T[some_dim1*buffer];
    
} // end constructor

// A method to return the stride size
template <typename T>
inline size_t RaggedRightArray<T>::stride(size_t i) const {
    // Ensure that i is within bounds
    assert(i < (dim1_ + 1) && "i is greater than dim1_ in RaggedRightArray");

    return start_index_[(i + 1)] - start_index_[i];
}

// A method to increase the stride size, in other words,
// this is used to build the stride array dynamically
// DO NOT USE with constructors that are given a stride array
template <typename T>
void RaggedRightArray<T>::push_back(size_t i){
    num_saved_ ++;
    start_index_[i+1] = num_saved_;
}

// Overload operator() to access data as array(i,j)
// where i=[0:N-1], j=[0:stride(i)]
template <typename T>
inline T& RaggedRightArray<T>::operator()(size_t i, size_t j) const {
    // get the 1D array index
    size_t start = start_index_[i];
    
    // asserts
    assert(i < dim1_ && "i is out of dim1 bounds in RaggedRightArray");  // die if >= dim1
    //assert(j < stride(i) && "j is out of stride bounds in RaggedRightArray");  // die if >= stride
    assert(j+start < length_ && "j+start is out of bounds in RaggedRightArray");  // die if >= 1D array length)
    
    return array_[j + start];
} // End operator()

//return size
template <typename T>
size_t RaggedRightArray<T>::size() const {
    return length_;
}

template <typename T>
RaggedRightArray<T> & RaggedRightArray<T>::operator+= (const size_t i) {
    this->num_saved_ ++;
    this->start_index_[i+1] = num_saved_;
    return *this;
}

//overload = operator
template <typename T>
RaggedRightArray<T> & RaggedRightArray<T>::operator= (const RaggedRightArray &temp) {

    if( this != &temp) {
        dim1_ = temp.dim1_;
        length_ = temp.length_;
        num_saved_ = temp.num_saved_;
        start_index_ = new size_t[dim1_ + 1];
        for (int j = 0; j < dim1_; j++) {
            start_index_[j] = temp.start_index_[j];  
        }
        array_ = new T[length_];
    }
	
    return *this;
}

template <typename T>
inline T* RaggedRightArray<T>::get_pointer() const{
    return array_;
}

template <typename T>
inline size_t* RaggedRightArray<T>::get_starts() const{
    return start_index_;
}

// Destructor
template <typename T>
RaggedRightArray<T>::~RaggedRightArray () {
    delete[] array_;
    delete[] start_index_;
}

//----end of RaggedRightArray class definitions----

//9. RaggedRightArrayofVectors
template <typename T>
class RaggedRightArrayofVectors {
private:
    size_t *start_index_;
    T * array_;
    
    size_t dim1_, length_, vector_dim_;
    size_t num_saved_; // the number saved in the 1D array
    
public:
    // Default constructor
    RaggedRightArrayofVectors ();
    
    //--- 3D array access of a ragged right array storing a vector of size vector_dim_ at each (i,j)---
    
    // Overload constructor for a CArray
    RaggedRightArrayofVectors (CArray<size_t> &strides_array, size_t vector_dim);
    
    // Overload constructor for a ViewCArray
    RaggedRightArrayofVectors (ViewCArray<size_t> &strides_array, size_t vector_dim);
    
    // Overloaded constructor for a traditional array
    RaggedRightArrayofVectors (size_t *strides_array, size_t some_dim1, size_t vector_dim);
    
    // Overload constructor for a RaggedRightArray to
    // support a dynamically built stride_array
    RaggedRightArrayofVectors (size_t some_dim1, size_t buffer, size_t vector_dim);
    
    // A method to return the stride size
    size_t stride(size_t i) const;

    // A method to return the vector dim
    size_t vector_dim() const;
    
    // A method to increase the number of column entries, i.e.,
    // the stride size. Used with the constructor for building
    // the stride_array dynamically.
    // DO NOT USE with the constructures with a strides_array
    void push_back(size_t i);
    
    // Overload operator() to access data as array(i,j)
    // where i=[0:N-1], j=[stride(i)], k=[0,vector_dim_]
    T& operator()(size_t i, size_t j, size_t k) const;

    // method to return total size
    size_t size() const;

    //return pointer
    T* get_pointer() const;
    
    //get row starts array
    size_t* get_starts() const;

    RaggedRightArrayofVectors& operator+= (const size_t i);

    RaggedRightArrayofVectors& operator= (const RaggedRightArrayofVectors &temp);

    // Destructor
    ~RaggedRightArrayofVectors ( );
}; // End of RaggedRightArray

// Default constructor
template <typename T>
RaggedRightArrayofVectors<T>::RaggedRightArrayofVectors () {
    array_ = NULL;
}


// Overloaded constructor with CArray
template <typename T>
RaggedRightArrayofVectors<T>::RaggedRightArrayofVectors (CArray<size_t> &strides_array, size_t vector_dim){
    // The length of the stride array is some_dim1;
    dim1_  = strides_array.size();
    vector_dim_ = vector_dim;
    
    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[(dim1_ + 1)];  // note the dim1+1
    start_index_[0] = 0; // the 1D array starts at 0
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array and set the starting 1D index
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += strides_array(i)*vector_dim_;
        start_index_[(i + 1)] = count;
    } // end for i
    length_ = count;
    
    array_ = new T[length_];
} // End constructor

// Overloaded constructor with a view c array
template <typename T>
RaggedRightArrayofVectors<T>::RaggedRightArrayofVectors (ViewCArray<size_t> &strides_array, size_t vector_dim) {
    // The length of the stride array is some_dim1;
    dim1_  = strides_array.size();
    vector_dim_ = vector_dim;
    
    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[(dim1_ + 1)];  // note the dim1+1
    start_index_[0] = 0; // the 1D array starts at 0
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array and set the starting 1D index
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += strides_array(i)*vector_dim_;
        start_index_[(i + 1)] = count;
    } // end for i
    length_ = count;
    
    array_ = new T[length_];
} // End constructor

// Overloaded constructor with a regular cpp array
template <typename T>
RaggedRightArrayofVectors<T>::RaggedRightArrayofVectors (size_t *strides_array, size_t dim1, size_t vector_dim){
    // The length of the stride array is some_dim1;
    dim1_ = dim1;
    vector_dim_ = vector_dim;

    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[(dim1_ + 1)];  // note the dim1+1
    start_index_[0] = 0; // the 1D array starts at 0
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array of vectors and set the starting 1D index
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += strides_array[i]*vector_dim_;
        start_index_[(i + 1)] = count;
    } // end for i
    length_ = count;
    
    array_ = new T[length_];
} // End constructor

// overloaded constructor for a dynamically built strides_array.
// buffer is the max number of columns needed
template <typename T>
RaggedRightArrayofVectors<T>::RaggedRightArrayofVectors (size_t some_dim1, size_t buffer, size_t vector_dim){
    
    dim1_ = some_dim1;
    vector_dim_ = vector_dim;

    // create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[dim1_+1]();  // note the dim1+1
    //start_index_[0] = 0; // the 1D array starts at 0

    num_saved_ = 0;
    
    length_ = some_dim1*buffer*vector_dim;
    array_ = new T[some_dim1*buffer];
    
} // end constructor

// A method to return the stride size
template <typename T>
inline size_t RaggedRightArrayofVectors<T>::stride(size_t i) const {
    // Ensure that i is within bounds
    assert(i < (dim1_ + 1) && "i is greater than dim1_ in RaggedRightArray");

    return (start_index_[(i + 1)] - start_index_[i])/vector_dim_;
}

// A method to increase the stride size, in other words,
// this is used to build the stride array dynamically
// DO NOT USE with constructors that are given a stride array
template <typename T>
void RaggedRightArrayofVectors<T>::push_back(size_t i){
    num_saved_ += vector_dim_;
    start_index_[i+1] = num_saved_;
}

// Overload operator() to access data as array(i,j,k)
// where i=[0:N-1], j=[0:stride(i)], k=[0:vector_dim_]
template <typename T>
inline T& RaggedRightArrayofVectors<T>::operator()(size_t i, size_t j, size_t k) const {
    // get the 1D array index
    size_t start = start_index_[i];
    
    // asserts
    assert(i < dim1_ && "i is out of dim1 bounds in RaggedRightArray");  // die if >= dim1
    //assert(j < stride(i) && "j is out of stride bounds in RaggedRightArray");  // die if >= stride
    assert(j*vector_dim_+start + k < length_ && "j+start is out of bounds in RaggedRightArray");  // die if >= 1D array length)
    
    return array_[j*vector_dim_ + start + k];
} // End operator()

//return size
template <typename T>
size_t RaggedRightArrayofVectors<T>::size() const {
    return length_;
}

template <typename T>
RaggedRightArrayofVectors<T> & RaggedRightArrayofVectors<T>::operator+= (const size_t i) {
    this->num_saved_ += vector_dim_;
    this->start_index_[i+1] = num_saved_;
    return *this;
}

//overload = operator
template <typename T>
RaggedRightArrayofVectors<T> & RaggedRightArrayofVectors<T>::operator= (const RaggedRightArrayofVectors &temp) {

    if( this != &temp) {
        dim1_ = temp.dim1_;
        vector_dim_ = temp.vector_dim_;
        length_ = temp.length_;
        num_saved_ = temp.num_saved_;
        start_index_ = new size_t[dim1_ + 1];
        for (int j = 0; j < dim1_; j++) {
            start_index_[j] = temp.start_index_[j];  
        }
        array_ = new T[length_];
    }
	
    return *this;
}

template <typename T>
inline T* RaggedRightArrayofVectors<T>::get_pointer() const{
    return array_;
}

template <typename T>
inline size_t* RaggedRightArrayofVectors<T>::get_starts() const{
    return start_index_;
}

// Destructor
template <typename T>
RaggedRightArrayofVectors<T>::~RaggedRightArrayofVectors () {
    delete[] array_;
    delete[] start_index_;
}

//----end of RaggedRightArrayofVectors class definitions----

//10. RaggedDownArray
template <typename T>
class RaggedDownArray { 
private:
    size_t *start_index_;
	T * array_;

	size_t dim2_;
    size_t length_;
    size_t num_saved_; // the number saved in the 1D array

public:
    //default constructor
    RaggedDownArray() ;

    //~~~~2D`~~~~
	//overload constructor with CArray
	RaggedDownArray(CArray<size_t> &strides_array);

	//overload with ViewCArray
	RaggedDownArray(ViewCArray <size_t> &strides_array);

	//overload with traditional array
	RaggedDownArray(size_t *strides_array, size_t dome_dim1);

    // Overload constructor for a RaggedDownArray to
    // support a dynamically built stride_array
    RaggedDownArray (size_t some_dim2, size_t buffer);
    
	//method to return stride size
	size_t stride(size_t j);

    // A method to increase the number of column entries, i.e.,
    // the stride size. Used with the constructor for building
    // the stride_array dynamically.
    // DO NOT USE with the constructures with a strides_array
    void push_back(size_t j);
    
	//overload () operator to access data as array (i,j)
	T& operator()(size_t i, size_t j);

    // method to return total size
    size_t size();

    //return pointer
    T* get_pointer() const;
    
    //get row starts array
    size_t* get_starts() const;

    //overload = operator
    RaggedDownArray& operator= (const RaggedDownArray &temp);

    //destructor
    ~RaggedDownArray();

}; //~~~~~end of RaggedDownArray class declarations~~~~~~~~	

//no dims
template <typename T>
RaggedDownArray<T>::RaggedDownArray() {
    array_ = NULL;
}

//overload constructor with CArray 
template <typename T>
RaggedDownArray<T>::RaggedDownArray( CArray <size_t> &strides_array) {
    // Length of stride array
    //dim2_ = strides_array.size();

    // Create and initialize startding indices
    start_index_ = new size_t[dim2_+1]; //theres a plus 1, because 
    start_index_[0] = 0; //1D array starts at 0

		
	//length of strides
	dim2_ = strides_array.size();

    // Loop to find total length of 1D array
    size_t count = 0;
    for(size_t j = 0; j < dim2_ ; j++) { 
        count += strides_array(j);
        start_index_[j+1] = count;
    } 
    length_ = count;

    array_ = new T[length_];

} // End constructor 

// Overload constructor with ViewCArray
template <typename T>
RaggedDownArray<T>::RaggedDownArray( ViewCArray <size_t> &strides_array) {
    // Length of strides
    //dim2_ = strides_array.size();

    //create array for holding start indices
    start_index_ = new size_t[dim2_+1];
    start_index_[0] = 0;

    size_t count = 0;
    // Loop over to get total length of 1D array
    for(size_t j = 0; j < dim2_ ;j++ ) {
        count += strides_array(j);
        start_index_[j+1] = count;
    }
    length_ = count;	
    array_ = new T[length_];

} // End constructor 

// Overload constructor with regualar array
template <typename T>
RaggedDownArray<T>::RaggedDownArray( size_t *strides_array, size_t dim2){
    // Length of stride array
    dim2_ = dim2;

    // Create and initialize starting index of entries
    start_index_ = new size_t[dim2_+1];
    start_index_[0] = 0;

    // Loop over to find length of 1D array
    // Represent ragged down array and set 1D index
    size_t count = 0;
    for(size_t j = 0; j < dim2_; j++) {
        count += strides_array[j];
        start_index_[j+1] = count;
	}

    length_ = count;	
    array_ = new T[length_];

} //end construnctor

// overloaded constructor for a dynamically built strides_array.
// buffer is the max number of columns needed
template <typename T>
RaggedDownArray<T>::RaggedDownArray (size_t some_dim2, size_t buffer){
    
    dim2_ = some_dim2;
    
    // create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[dim2_+1]();  // note the dim2+1
    //start_index_[0] = 0; // the 1D array starts at 0
    
    num_saved_ = 0;
    
    length_ = some_dim2*buffer;
    array_ = new T[some_dim2*buffer];
    
} // end constructor

// Check the stride size
template <typename T>
size_t RaggedDownArray<T>::stride(size_t j) {
    return start_index_[j+1] - start_index_[j];
}

// A method to increase the stride size, in other words,
// this is used to build the stride array dynamically
// DO NOT USE with constructors that are given a stride array
template <typename T>
void RaggedDownArray<T>::push_back(size_t j){
    num_saved_ ++;
    start_index_[j+1] = num_saved_;
}

//return size
template <typename T>
size_t RaggedDownArray<T>::size() {
    return length_;
}

// overload operator () to access data as an array(i,j)
// Note: i = 0:stride(j), j = 0:N-1
template <typename T>
T& RaggedDownArray<T>::operator()(size_t i, size_t j) {
    // Where is the array starting?
    // look at start index
    size_t start = start_index_[j]; 

    // Make sure we are within array bounds
    assert(i < stride(j) && "i is out of bounds in RaggedDownArray");
    assert(j < dim2_ && "j is out of dim2_ bounds in RaggedDownArray");
    assert(i+start < length_ && "i+start is out of bounds in RaggedDownArray");  // die if >= 1D array length)
    
    return array_[i + start];

} // End () operator

//overload = operator
template <typename T>
RaggedDownArray<T> & RaggedDownArray<T>::operator= (const RaggedDownArray &temp) {

    if( this != &temp) {
        dim2_ = temp.dim2_;
        length_ = temp.length_;
        num_saved_ = temp.num_saved_;
        start_index_ = new size_t[dim2_ + 1];
        for (int j = 0; j < dim2_; j++) {
            start_index_[j] = temp.start_index_[j];  
        }
        array_ = new T[length_];
    }
	
    return *this;
}

template <typename T>
inline T* RaggedDownArray<T>::get_pointer() const{
    return array_;
}


template <typename T>
inline size_t* RaggedDownArray<T>::get_starts() const{
    return start_index_;
}

// Destructor
template <typename T>
RaggedDownArray<T>::~RaggedDownArray() {
    delete[] array_;
    delete[] start_index_;

} // End destructor


//----end of RaggedDownArray----


//11. DynamicRaggedRightArray

template <typename T>
class DynamicRaggedRightArray {
private:
    size_t *stride_;
    T * array_;
    
    size_t dim1_;
    size_t dim2_;
    size_t length_;
    
public:
    // Default constructor
    DynamicRaggedRightArray ();
    
    //--- 2D array access of a ragged right array ---
    
    // overload constructor
    DynamicRaggedRightArray (size_t dim1, size_t dim2);
    
    // A method to return or set the stride size
    size_t& stride(size_t i) const;
    
    // A method to return the size
    size_t size() const;

    //return pointer
    T* get_pointer() const;
    
    // Overload operator() to access data as array(i,j),
    // where i=[0:N-1], j=[stride(i)]
    T& operator()(size_t i, size_t j) const;
    
    // Overload copy assignment operator
    DynamicRaggedRightArray& operator= (const DynamicRaggedRightArray &temp);
    
    // Destructor
    ~DynamicRaggedRightArray ();
};

//nothing
template <typename T>
DynamicRaggedRightArray<T>::DynamicRaggedRightArray () {
    array_ = NULL;
}

// Overloaded constructor
template <typename T>
DynamicRaggedRightArray<T>::DynamicRaggedRightArray (size_t dim1, size_t dim2) {
    // The dimensions of the array;
    dim1_  = dim1;
    dim2_  = dim2;
    length_ = dim1*dim2;
    
    // Create memory on the heap for the values
    array_ = new T[dim1*dim2];
    
    // Create memory for the stride size in each row
    stride_ = new size_t[dim1];
    
    // Initialize the stride
    for (int i=0; i<dim1_; i++){
        stride_[i] = 0;
    }
    
    // Start index is always = j + i*dim2
}

// A method to set the stride size for row i
template <typename T>
size_t& DynamicRaggedRightArray<T>::stride(size_t i) const {
    return stride_[i];
}

//return size
template <typename T>
size_t DynamicRaggedRightArray<T>::size() const{
    return length_;
}

// Overload operator() to access data as array(i,j),
// where i=[0:N-1], j=[0:stride(i)]
template <typename T>
inline T& DynamicRaggedRightArray<T>::operator()(size_t i, size_t j) const {
    // Asserts
    assert(i < dim1_ && "i is out of dim1 bounds in DynamicRaggedRight");  // die if >= dim1
    assert(j < dim2_ && "j is out of dim2 bounds in DynamicRaggedRight");  // die if >= dim2
    assert(j < stride_[i] && "j is out of stride bounds in DynamicRaggedRight");  // die if >= stride
    
    return array_[j + i*dim2_];
}

//overload = operator
template <typename T>
inline DynamicRaggedRightArray<T>& DynamicRaggedRightArray<T>::operator= (const DynamicRaggedRightArray &temp)
{
    
    if( this != &temp) {
        dim1_ = temp.dim1_;
        dim2_ = temp.dim2_;
        length_ = temp.length_;
        stride_ = new size_t[dim1_];
        for (int i = 0; i < dim1_; i++) {
            stride_[i] = temp.stride_[i];
        }
        array_ = new T[length_];
    }
    
    return *this;
}

template <typename T>
inline T* DynamicRaggedRightArray<T>::get_pointer() const{
    return array_;
}

// Destructor
template <typename T>
DynamicRaggedRightArray<T>::~DynamicRaggedRightArray() {
    delete[] array_;
    delete[] stride_;
}




//----end DynamicRaggedRightArray class definitions----


//12. DynamicRaggedDownArray

template <typename T>
class DynamicRaggedDownArray {
private:
    size_t *stride_;
    T * array_;
    
    size_t dim1_;
    size_t dim2_;
    size_t length_;
    
public:
    // Default constructor
    DynamicRaggedDownArray ();
    
    //--- 2D array access of a ragged right array ---
    
    // overload constructor
    DynamicRaggedDownArray (size_t dim1, size_t dim2);
    
    // A method to return or set the stride size
    size_t& stride(size_t j) const;
    
    // A method to return the size
    size_t size() const;
    
    // Overload operator() to access data as array(i,j),
    // where i=[stride(j)], j=[0:N-1]
    T& operator()(size_t i, size_t j) const;
    
    // Overload copy assignment operator
    DynamicRaggedDownArray& operator= (const DynamicRaggedDownArray &temp);

    //return pointer
    T* get_pointer() const;
    
    // Destructor
    ~DynamicRaggedDownArray ();
};

//nothing
template <typename T>
DynamicRaggedDownArray<T>::DynamicRaggedDownArray () {}

// Overloaded constructor
template <typename T>
DynamicRaggedDownArray<T>::DynamicRaggedDownArray (size_t dim1, size_t dim2) {
    // The dimensions of the array;
    dim1_  = dim1;
    dim2_  = dim2;
    length_ = dim1*dim2;
    
    // Create memory on the heap for the values
    array_ = new T[dim1*dim2];
    
    // Create memory for the stride size in each row
    stride_ = new size_t[dim2];
    
    // Initialize the stride
    for (int j=0; j<dim2_; j++){
        stride_[j] = 0;
    }
    
    // Start index is always = i + j*dim1
}

// A method to set the stride size for column j
template <typename T>
size_t& DynamicRaggedDownArray<T>::stride(size_t j) const {
    return stride_[j];
}

//return size
template <typename T>
size_t DynamicRaggedDownArray<T>::size() const{
    return length_;
}

// overload operator () to access data as an array(i,j)
// Note: i = 0:stride(j), j = 0:N-1

template <typename T>
inline T& DynamicRaggedDownArray<T>::operator()(size_t i, size_t j) const {
    // Asserts
    assert(i < dim1_ && "i is out of dim1 bounds in DynamicRaggedDownArray");  // die if >= dim1
    assert(j < dim2_ && "j is out of dim2 bounds in DynamicRaggedDownArray");  // die if >= dim2
    assert(i < stride_[j] && "i is out of stride bounds in DynamicRaggedDownArray");  // die if >= stride
    
    return array_[i + j*dim1_];
}

//overload = operator
template <typename T>
inline DynamicRaggedDownArray<T>& DynamicRaggedDownArray<T>::operator= (const DynamicRaggedDownArray &temp)
{
    
    if( this != &temp) {
        dim1_ = temp.dim1_;
        dim2_ = temp.dim2_;
        length_ = temp.length_;
        stride_ = new size_t[dim1_];
        for (int j = 0; j < dim2_; j++) {
            stride_[j] = temp.stride_[j];
        }
        array_ = new T[length_];
    }
    
    return *this;
}

template <typename T>
inline T* DynamicRaggedDownArray<T>::get_pointer() const{
    return array_;
}

// Destructor
template <typename T>
DynamicRaggedDownArray<T>::~DynamicRaggedDownArray() {
    delete[] array_;
    delete[] stride_;
}

//----end of DynamicRaggedDownArray class definitions-----




//13. SparseRowArray
template <typename T>
class SparseRowArray {
private:
    size_t *start_index_;
    size_t *column_index_;
    
    T * array_;
    
    size_t dim1_, length_;
    
public:
    // Default constructor
    SparseRowArray ();
    
    //--- 2D array access of a ragged right array ---
    
    // Overload constructor for a CArray
    SparseRowArray (CArray<size_t> &strides_array);
    
    // Overload constructor for a ViewCArray
    SparseRowArray (ViewCArray<size_t> &strides_array);
    
    // Overloaded constructor for a traditional array
    SparseRowArray (size_t *strides_array, size_t some_dim1);
    
    // A method to return the stride size
    size_t stride(size_t i) const;
    
    // A method to return the column index as array.column_index(i,j)
    size_t& column_index(size_t i, size_t j) const;
    
    // A method to access data as array.value(i,j),
    // where i=[0:N-1], j=[stride(i)]
    T& value(size_t i, size_t j) const;

    // A method to return the total size of the array
    size_t size() const;

    //return pointer
    T* get_pointer() const;

    //get row starts array
    size_t* get_starts() const;
    
    // Destructor
    ~SparseRowArray ();
}; 

//Default Constructor
template <typename T>
SparseRowArray<T>::SparseRowArray (){
    array_ = NULL;
}
// Overloaded constructor
template <typename T>
SparseRowArray<T>::SparseRowArray (CArray<size_t> &strides_array) {
    // The length of the stride array is some_dim1;
    dim1_  = strides_array.size();
    
    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[dim1_+1];  // note the dim1+1
    start_index_[0] = 0; // the 1D array starts at 0
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array and set the starting 1D index
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += strides_array(i);
        start_index_[i+1] = count;
    } // end for i
    
    length_ = count;
    array_ = new T[count];
    column_index_ = new size_t[count];
} 


// Overloaded constructor
template <typename T>
SparseRowArray<T>::SparseRowArray (ViewCArray<size_t> &strides_array) {
    // The length of the stride array is some_dim1;
    dim1_  = strides_array.size();
    
    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[dim1_+1];  // note the dim1+1
    start_index_[0] = 0; // the 1D array starts at 0
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array and set the starting 1D index
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += strides_array(i);
        start_index_[i+1] = count;
    } // end for i
    
    length_ = count;
    array_ = new T[count];
    column_index_ = new size_t[count];
} 

// Overloaded constructor
template <typename T>
SparseRowArray<T>::SparseRowArray (size_t *strides_array, size_t dim1) {
    // The length of the stride array is some_dim1;
    dim1_ = dim1;
    
    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = new size_t[dim1_+1];  // note the dim1+1
    start_index_[0] = 0; // the 1D array starts at 0
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array and set the starting 1D index
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += strides_array[i];
        start_index_[i+1] = count;
    } // end for i
    
    length_ = count;
    array_ = new T[count];
    column_index_ = new size_t[count];
} 


// A method to return the stride size
template <typename T>
size_t SparseRowArray<T>::stride(size_t i) const {
    return start_index_[i+1] - start_index_[i];
}

// A method to return the column index
template <typename T>
size_t& SparseRowArray<T>::column_index(size_t i, size_t j) const {
    // Get the 1D array index
    size_t start = start_index_[i];
    
    // Asserts
    assert(i < dim1_ && "i is out of dim1 bounds in SparseRowArray");  // die if >= dim1
    assert(j < stride(i) && "j is out of stride bounds in SparseRowArray");  // die if >= stride
    
    return column_index_[j + start];
}

// Access data as array.value(i,j), 
// where i=[0:N-1], j=[0:stride(i)]
template <typename T>
inline T& SparseRowArray<T>::value(size_t i, size_t j) const {
    // Get the 1D array index
    size_t start = start_index_[i];
    
    // Asserts
    assert(i < dim1_ && "i is out of dim1 bounds in sparseRowArray");  // die if >= dim1
    assert(j < stride(i) && "j is out of stride bounds in sparseRowArray");  // die if >= stride
    
    return array_[j + start];
} 

//return size
template <typename T>
size_t SparseRowArray<T>::size() const{
    return length_;
}

template <typename T>
inline T* SparseRowArray<T>::get_pointer() const{
    return array_;
}

template <typename T>
inline size_t* SparseRowArray<T>::get_starts() const{
    return start_index_;
}

// Destructor
template <typename T>
SparseRowArray<T>::~SparseRowArray() {
    delete[] array_;
    delete[] start_index_;
    delete[] column_index_;
}

//---- end of SparseRowArray class definitions-----



//14. SparseColArray
template <typename T>
class SparseColArray {

private:
	size_t *start_index_;
	size_t *row_index_;
	T * array_;

	size_t dim2_, length_;

public:

	//default constructor
	SparseColArray ();

	//constructor with CArray
	SparseColArray(CArray<size_t> &strides_array);

	//constructor with ViewCArray
	SparseColArray(ViewCArray<size_t> &strides_array);

	//constructor with regular array
	SparseColArray(size_t *strides_array, size_t some_dim1);

	//method return stride size
	size_t stride(size_t j) const;

	//method return row index ass array.row_index(i,j)
	size_t& row_index(size_t i, size_t j) const;

	//method access data as an array
	T& value(size_t i, size_t j) const;

    // A method to return the total size of the array
    size_t size() const;

    //return pointer
    T* get_pointer() const;

    //get row starts array
    size_t* get_starts() const;

	//destructor
	~SparseColArray();
};

//Default Constructor
template <typename T>
SparseColArray<T>::SparseColArray (){
    array_ = NULL;
}
//overload constructor with CArray
template <typename T>
SparseColArray<T>::SparseColArray(CArray<size_t> &strides_array) {

	dim2_ = strides_array.size();

	start_index_ = new size_t[dim2_+1];
	start_index_[0] = 0;

	//loop over to find total length of the 1D array
	size_t count = 0;
	for(size_t j = 0; j < dim2_; j++) {
	  count+= strides_array(j);
	  start_index_[j+1] = count;
	}
    
    length_ = count;
	array_ = new T[count];
	row_index_ = new T[count];

} //end constructor with CArray


//overload constructor with ViewCArray
template <typename T>
SparseColArray<T>::SparseColArray(ViewCArray<size_t> &strides_array) {

	dim2_ = strides_array.size();

	//create and initialize starting index of 1D array
	start_index_ = new size_t[dim2_+1];
	start_index_[0] = 0;

	//loop over to find total length of 1D array
	size_t count = 0;
	for(size_t j = 0; j < dim2_ ; j++) {
	  count += strides_array(j);
	  start_index_[j+1] = count;
	}
    
    length_ = count;
	array_ = new T[count];
	row_index_ = new T[count];

} //end constructor

//overload constructor with traditional array
template <typename T>
SparseColArray<T>::SparseColArray(size_t *strides_array, size_t dim2) {

	dim2_ = dim2;

	//create and initialize the starting index 
	start_index_ = new size_t[dim2_ +1];
	start_index_[0] = 0;

	//loop over to find the total length of the 1D array
	size_t count = 0;
	for(size_t j = 0; j < dim2_; j++) {
	  count += strides_array[j];
	  start_index_[j+1] = count;
	}
    
    length_ = count;
	array_ = new T[count];
	row_index_ = new T[count];

} //end constructor

//method to return stride size
template <typename T>
size_t SparseColArray<T>::stride(size_t j) const{
	return start_index_[j+1] - start_index_[j];
}

//acces data ass arrow.row_index(i,j)
// where i = 0:stride(j), j = 0:N-1
template <typename T>
size_t& SparseColArray<T>::row_index(size_t i, size_t j) const {

	//get 1D array index
	size_t start = start_index_[j];

	//asserts to make sure we are in bounds
	assert(i < stride(j) && "i is out of stride bounnds in SparseColArray!");
	assert(j < dim2_ && "j is out of dim1 bounds in SparseColArray");

	return row_index_[i + start];

} //end row index method	


//access values as array.value(i,j)
// where i = 0:stride(j), j = 0:N-1
template <typename T>
T& SparseColArray<T>::value(size_t i, size_t j) const {

	size_t start = start_index_[j];

	//asserts
	assert(i < stride(j) && "i is out of stride boundns in SparseColArray");
	assert(j < dim2_ && "j is out of dim1 bounds in SparseColArray");

	return array_[i + start];

}

//return size
template <typename T>
size_t SparseColArray<T>::size() const{
    return length_;
}

template <typename T>
inline T* SparseColArray<T>::get_pointer() const{
    return array_;
}

template <typename T>
inline size_t* SparseColArray<T>::get_starts() const{
    return start_index_;
}

//destructor
template <typename T>
SparseColArray<T>::~SparseColArray() {
	delete [] array_;
	delete [] start_index_;
	delete [] row_index_;
}

//----end SparseColArray----

//=======================================================================
//	end of standard MATAR data-types
//========================================================================

/*! \brief Kokkos version of the serial FArray class.
 *
 *  This is the Kokkos version of the serial FArray class. 
 *  Its usage is analagous to that of the serial FArray class, and it is to be
 *  used in Kokkos-specific code.
 */
#ifdef HAVE_KOKKOS
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class FArrayKokkos {

    using TArray1D = Kokkos::View<T*, Layout, ExecSpace, MemoryTraits>;
    
private:

    /*!
     * Private variable that specifies the length of the first dimension
     * of the FArrayKokkos object.
     */
    size_t dim1_;
    /*!
     * Private variable that specifies the length of the second dimension
     * of the FArrayKokkos object.
     */
    size_t dim2_;
    size_t dim3_;
    size_t dim4_;
    size_t dim5_;
    size_t dim6_;
    size_t dim7_;
    size_t length_;
    TArray1D this_array_; 

public:

    /*!
     * \brief Default constructor
     */
    FArrayKokkos();

    /*!
     * \brief An overloaded constructor used to construct an 1D FArrayKokkos
              object.

        \param some_dim1 the length of the first dimension
     */
    FArrayKokkos(size_t some_dim1);

    /*!
     * \brief An overloaded constructor used to construct a 2D FArrayKokkos
              object.

        \param some_dim1 the length of the first dimension
        \param some_dim2 the length of the second dimension
     */
    FArrayKokkos(size_t some_dim1, size_t some_dim2);

    /*!
     * \brief An overloaded constructor used to construct a 3D FArrayKokkos
              object.

        \param some_dim1 the length of the first dimension
        \param some_dim2 the length of the second dimension
        \param some_dim3 the length of the third dimension
     */
    FArrayKokkos(size_t some_dim1, size_t some_dim2, size_t some_dim3);

    FArrayKokkos(size_t some_dim1, size_t some_dim2, size_t some_dim3, 
                 size_t some_dim4);

    FArrayKokkos(size_t some_dim1, size_t some_dim2, size_t some_dim3, 
                 size_t some_dim4, size_t some_dim5); 

    FArrayKokkos(size_t some_dim1, size_t sone_dim2, size_t some_dim3, 
                 size_t some_dim4, size_t some_dim5, size_t some_dim6);

    FArrayKokkos(size_t some_dim1, size_t sone_dim2, size_t some_dim3,
                 size_t some_dim4, size_t some_dim5, size_t some_dim6,
                 size_t some_dim7);
    
    // Overload operator() to acces data
    // from 1D to 6D
    
    KOKKOS_FUNCTION
    T& operator()(size_t i) const;
    
    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j) const;

    KOKKOS_FUNCTION
    T& operator() (size_t i, size_t j, size_t k) const;

    KOKKOS_FUNCTION
    T& operator() (size_t i, size_t j, size_t k,
                   size_t l) const;

    KOKKOS_FUNCTION
    T& operator() (size_t i, size_t j, size_t k,
                   size_t l, size_t m) const;

    KOKKOS_FUNCTION
    T& operator() (size_t i, size_t j, size_t k,
                   size_t l, size_t m, size_t n) const;

    KOKKOS_FUNCTION
    T& operator() (size_t i, size_t j, size_t k,
                   size_t l, size_t m, size_t n, size_t o) const;

    // Overload = operator
    FArrayKokkos& operator= (const FArrayKokkos<T,Layout,ExecSpace,MemoryTraits> &temp);

    KOKKOS_FUNCTION
    size_t size();

    size_t extent();

    T* pointer();
    
    //return kokkos view
    TArray1D get_kokkos_view();

    // Destructor
    KOKKOS_FUNCTION
    ~FArrayKokkos();    

}; //end of FArrayKokkos declarations

// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::FArrayKokkos() {}

// Overloaded 1D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::FArrayKokkos(size_t some_dim1){
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    length_ = dim1_;
    this_array_ = TArray1D("this_array_", length_);
}

// Overloaded 2D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::FArrayKokkos(size_t some_dim1, size_t some_dim2) {

    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    length_ = (dim1_ * dim2_);
    this_array_ = TArray1D("this_array_", length_);
}

// Overloaded 3D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::FArrayKokkos(size_t some_dim1, size_t some_dim2, 
                              size_t some_dim3) {

    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    length_ = (dim1_ * dim2_ * dim3_);
    this_array_ = TArray1D("this_array_", length_);
}

// Overloaded 4D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::FArrayKokkos(size_t some_dim1, size_t some_dim2, 
                              size_t some_dim3, size_t some_dim4) {

    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_);
    this_array_ = TArray1D("this_array_", length_);
}

// Overloaded 5D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::FArrayKokkos(size_t some_dim1, size_t some_dim2, 
                              size_t some_dim3, size_t some_dim4, 
                              size_t some_dim5) {

    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_);
    this_array_ = TArray1D("this_array_", length_);
}

// Overloaded 6D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::FArrayKokkos(size_t some_dim1, size_t some_dim2, 
                              size_t some_dim3, size_t some_dim4, 
                              size_t some_dim5, size_t some_dim6) {

    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_);
    this_array_ = TArray1D("this_array_", length_);
}

// Overloaded 7D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::FArrayKokkos(size_t some_dim1, size_t some_dim2,
                              size_t some_dim3, size_t some_dim4,
                              size_t some_dim5, size_t some_dim6,
                              size_t some_dim7) {
    
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    dim7_ = some_dim7;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_ * dim7_);
    this_array_ = TArray1D("this_array_", length_);
}

// Definitions of overload operator()
// for 1D to 7D
// Note: the indices for array all start at 0

// 1D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()( size_t i) const {
    assert( i < dim1_ && "i is out of bounds in FArrayKokkos 1D!");
    return this_array_(i);
}

// 2D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j) const {
    assert( i < dim1_ && "i is out of bounds in FArrayKokkos 2D!");
    assert( j < dim2_ && "j is out of bounds in FArrayKokkos 2D!");
    return this_array_(i + (j * dim1_));
}

// 3D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k) const {
    assert( i < dim1_ && "i is out of bounds in FArrayKokkos 3D!");
    assert( j < dim2_ && "j is out of bounds in FArrayKokkos 3D!");
    assert( k < dim3_ && "k is out of bounds in FArrayKokkos 3D!");
    return this_array_(i + (j * dim1_) 
                         + (k * dim1_ * dim2_));
}

// 4D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l) const {
    assert( i < dim1_ && "i is out of bounds in FArrayKokkos 4D!");
    assert( j < dim2_ && "j is out of bounds in FArrayKokkos 4D!");
    assert( k < dim3_ && "k is out of bounds in FArrayKokkos 4D!");
    assert( l < dim4_ && "l is out of bounds in FArrayKokkos 4D!");
    return this_array_(i + (j * dim1_) 
                         + (k * dim1_ * dim2_) 
                         + (l * dim1_ * dim2_ * dim3_));
}

// 5D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l, 
                               size_t m) const {
    assert( i < dim1_ && "i is out of bounds in FArrayKokkos 5D!");
    assert( j < dim2_ && "j is out of bounds in FArrayKokkos 5D!");
    assert( k < dim3_ && "k is out of bounds in FArrayKokkos 5D!");
    assert( l < dim4_ && "l is out of bounds in FArrayKokkos 5D!");
    assert( m < dim5_ && "m is out of bounds in FArrayKokkos 5D!");
    return this_array_(i + (j * dim1_) 
                         + (k * dim1_ * dim2_) 
                         + (l * dim1_ * dim2_ * dim3_) 
                         + (m * dim1_ * dim2_ * dim3_ * dim4_));
}

// 6D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l, 
                               size_t m, size_t n) const {
    assert( i < dim1_ && "i is out of bounds in FArrayKokkos 6D!");
    assert( j < dim2_ && "j is out of bounds in FArrayKokkos 6D!");
    assert( k < dim3_ && "k is out of bounds in FArrayKokkos 6D!");
    assert( l < dim4_ && "l is out of bounds in FArrayKokkos 6D!");
    assert( m < dim5_ && "m is out of bounds in FArrayKokkos 6D!");
    assert( n < dim6_ && "n is out of bounds in FArrayKokkos 6D!");
    return this_array_(i + (j * dim1_) 
                         + (k * dim1_ * dim2_) 
                         + (l * dim1_ * dim2_ * dim3_) 
                         + (m * dim1_ * dim2_ * dim3_ * dim4_) 
                         + (n * dim1_ * dim2_ * dim3_ * dim4_ * dim5_));
}

// 7D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n, size_t o) const {
    assert( i < dim1_ && "i is out of bounds in FArrayKokkos 7D!");
    assert( j < dim2_ && "j is out of bounds in FArrayKokkos 7D!");
    assert( k < dim3_ && "k is out of bounds in FArrayKokkos 7D!");
    assert( l < dim4_ && "l is out of bounds in FArrayKokkos 7D!");
    assert( m < dim5_ && "m is out of bounds in FArrayKokkos 7D!");
    assert( n < dim6_ && "n is out of bounds in FArrayKokkos 7D!");
    assert( o < dim7_ && "o is out of bounds in FArrayKokkos 7D!");
    return this_array_(i + (j * dim1_)
                         + (k * dim1_ * dim2_)
                         + (l * dim1_ * dim2_ * dim3_)
                         + (m * dim1_ * dim2_ * dim3_ * dim4_)
                         + (n * dim1_ * dim2_ * dim3_ * dim4_ * dim5_)
                         + (o * dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_));
}

// Overload = operator
// for object assingment THIS = FArrayKokkos<> TEMP(n,m,,,,)
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>& FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator= (const FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>& temp) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    if (this != &temp) {
      dim1_ = temp.dim1_;
      dim2_ = temp.dim2_;
      dim3_ = temp.dim3_;
      dim4_ = temp.dim4_;
      dim5_ = temp.dim5_;
      dim6_ = temp.dim6_;
      dim7_ = temp.dim7_;
      length_ = temp.length_;
      this_array_ = TArray1D("this_array_", length_);
    }
    return *this;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
size_t FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::size() {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
size_t FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::extent() {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T* FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::pointer() {
    return this_array_.data();
}

//return the stored Kokkos view
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
Kokkos::View<T*, Layout, ExecSpace, MemoryTraits> FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_view() {
    return this_array_;
}

// Destructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
FArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::~FArrayKokkos() {}

////////////////////////////////////////////////////////////////////////////////
// End of FArrayKokkos
////////////////////////////////////////////////////////////////////////////////

/*! \brief Kokkos version of the serial ViewFArray class.
 *
 */
template <typename T>
class ViewFArrayKokkos {

private: 
    size_t dim1_;
    size_t dim2_;
    size_t dim3_;
    size_t dim4_;
    size_t dim5_;
    size_t dim6_;
    size_t dim7_;
    size_t length_;
    T* this_array_;

public:
    ViewFArrayKokkos();

    ViewFArrayKokkos(T* some_array, size_t dim1);
    
    ViewFArrayKokkos(T* some_array, size_t dim1, size_t dim2);
    
    ViewFArrayKokkos(T* some_array, size_t dim1, size_t dim2, size_t dim3);
    
    ViewFArrayKokkos(T* some_array, size_t dim1, size_t dim2, size_t dim3, 
                     size_t dim4);
    
    ViewFArrayKokkos(T* some_array, size_t dim1, size_t dim2, size_t dim3, 
                     size_t dim4, size_t dim5);
    
    ViewFArrayKokkos(T* some_array, size_t dim1, size_t dim2, size_t dim3, 
                     size_t dim4, size_t dim5, size_t dim6);

    ViewFArrayKokkos(T* some_array, size_t dim1, size_t dim2, size_t dim3,
                     size_t dim4, size_t dim5, size_t dim6, size_t dim7);
    
    KOKKOS_FUNCTION
    T& operator()(size_t i) const; 

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k,
                  size_t l) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k,
                  size_t l, size_t m) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k,
                  size_t l, size_t m, size_t n) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k,
                  size_t l, size_t m, size_t n, size_t o) const;

    
    KOKKOS_FUNCTION
    size_t size();

    size_t extent();

    KOKKOS_FUNCTION
    ~ViewFArrayKokkos();

}; // End of ViewFArrayKokkos declarations

// Default constructor
template <typename T>
ViewFArrayKokkos<T>::ViewFArrayKokkos() {}

// Overloaded 1D constructor
template <typename T>
ViewFArrayKokkos<T>::ViewFArrayKokkos(T *some_array, size_t dim1) {
    dim1_ = dim1;
    length_ = dim1_;
    this_array_ = some_array;
}

// Overloaded 2D constructor
template <typename T>
ViewFArrayKokkos<T>::ViewFArrayKokkos(T *some_array, size_t dim1, size_t dim2) {
    dim1_ = dim1;
    dim2_ = dim2;
    length_ = (dim1_ * dim2_);
    this_array_ = some_array;
}

// Overloaded 3D constructor
template <typename T>
ViewFArrayKokkos<T>::ViewFArrayKokkos(T *some_array, size_t dim1, size_t dim2, 
                                      size_t dim3) {
    dim1_ = dim1;
    dim2_ = dim2;
    dim3_ = dim3;
    length_ = (dim1_ * dim2_ * dim3_);
    this_array_ = some_array;
}

// Overloaded 4D constructor
template <typename T>
ViewFArrayKokkos<T>::ViewFArrayKokkos(T *some_array, size_t dim1, size_t dim2, 
                                      size_t dim3, size_t dim4) {
    dim1_ = dim1;
    dim2_ = dim2;
    dim3_ = dim3;
    dim4_ = dim4;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_);
    this_array_ = some_array;
}

// Overloaded 5D constructor
template <typename T>
ViewFArrayKokkos<T>::ViewFArrayKokkos(T *some_array, size_t dim1, size_t dim2, 
                                      size_t dim3, size_t dim4, size_t dim5) {
    dim1_ = dim1;
    dim2_ = dim2;
    dim3_ = dim3;
    dim4_ = dim4;
    dim5_ = dim5;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_);
    this_array_ = some_array;
}

// Overloaded 6D constructor
template <typename T>
ViewFArrayKokkos<T>::ViewFArrayKokkos(T *some_array, size_t dim1, size_t dim2, 
                                      size_t dim3, size_t dim4, size_t dim5, 
                                      size_t dim6) {
    dim1_ = dim1;
    dim2_ = dim2;
    dim3_ = dim3;
    dim4_ = dim4;
    dim5_ = dim5;
    dim6_ = dim6;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_);
    this_array_ = some_array;
}

// Overloaded 7D constructor
template <typename T>
ViewFArrayKokkos<T>::ViewFArrayKokkos(T *some_array, size_t dim1, size_t dim2,
                                      size_t dim3, size_t dim4, size_t dim5,
                                      size_t dim6, size_t dim7) {
    dim1_ = dim1;
    dim2_ = dim2;
    dim3_ = dim3;
    dim4_ = dim4;
    dim5_ = dim5;
    dim6_ = dim6;
    dim7_ = dim7;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_ * dim7_);
    this_array_ = some_array;
}

// Overloaded operator() for 1D array access
template <typename T>
KOKKOS_FUNCTION
T& ViewFArrayKokkos<T>::operator()(size_t i) const {
    assert( i < dim1_ && "i is out of bounds in ViewFArrayKokkos 1D!");
    return this_array_[i];
}

//2D
template <typename T>
KOKKOS_FUNCTION
T& ViewFArrayKokkos<T>::operator()(size_t i, size_t j) const {
    assert(i < dim1_ && "i is out of bounds in ViewFArrayKokkos 2D!");
    assert(j < dim2_ && "j is out of bounds in ViewFArrayKokkos 2D!");
    return this_array_[i + (j * dim1_)];
}

//3D
template <typename T>
KOKKOS_FUNCTION
T& ViewFArrayKokkos<T>::operator()(size_t i, size_t j, size_t k) const {
    assert(i < dim1_ && "i is out of bounds in ViewFArrayKokkos 3D!");
    assert(j < dim2_ && "j is out of bounds in ViewFArrayKokkos 3D!");
    assert(k < dim3_ && "k is out of bounds in ViewFArrayKokkos 3D!");
    return this_array_[i + (j * dim1_) 
                         + (k * dim1_ * dim2_)];
}

//4D
template <typename T>
KOKKOS_FUNCTION
T& ViewFArrayKokkos<T>::operator()(size_t i, size_t j, size_t k, 
                                   size_t l) const {
    assert(i < dim1_ && "i is out of bounds in ViewFArrayKokkos 4D!");
    assert(j < dim2_ && "j is out of bounds in ViewFArrayKokkos 4D!");
    assert(k < dim3_ && "k is out of bounds in ViewFArrayKokkos 4D!");
    assert(l < dim4_ && "l is out of bounds in ViewFArrayKokkos 4D!");
    return this_array_[i + (j * dim1_) 
                         + (k * dim1_ * dim2_) 
                         + (l * dim1_ * dim2_ *dim3_)];
}

//5D
template <typename T>
KOKKOS_FUNCTION
T& ViewFArrayKokkos<T>::operator()(size_t i, size_t j, size_t k, 
                                   size_t l, size_t m) const {
    assert(i < dim1_ && "i is out of bounds in ViewFArrayKokkos 5D!");
    assert(j < dim2_ && "j is out of bounds in ViewFArrayKokkos 5D!");
    assert(k < dim3_ && "k is out of bounds in ViewFArrayKokkos 5D!");
    assert(l < dim4_ && "l is out of bounds in ViewFArrayKokkos 5D!");
    assert(m < dim5_ && "m is out of bounds in ViewFArrayKokkos 5D!");
    return this_array_[i + (j * dim1_) 
                         + (k * dim1_ * dim2_) 
                         + (l * dim1_ * dim2_ * dim3_) 
                         + (m * dim1_ * dim2_ * dim3_ * dim4_)];
}

//6D
template <typename T>
KOKKOS_FUNCTION
T& ViewFArrayKokkos<T>::operator()(size_t i, size_t j, size_t k, 
                                   size_t l, size_t m, size_t n) const {
    assert(i < dim1_ && "i is out of bounds in ViewFArrayKokkos 6D!");
    assert(j < dim2_ && "j is out of bounds in ViewFArrayKokkos 6D!");
    assert(k < dim3_ && "k is out of bounds in ViewFArrayKokkos 6D!");
    assert(l < dim4_ && "l is out of bounds in ViewFArrayKokkos 6D!");
    assert(m < dim5_ && "m is out of bounds in ViewFArrayKokkos 6D!");
    assert(n < dim6_ && "n is out of bounds in ViewFArrayKokkos 6D!");
    return this_array_[i + (j * dim1_) 
                         + (k * dim1_ * dim2_) 
                         + (l * dim1_ * dim2_ * dim3_) 
                         + (m * dim1_ * dim2_ * dim3_ * dim4_)
                         + (n * dim1_ * dim2_ * dim3_ * dim4_ * dim5_)];
}

//7D
template <typename T>
KOKKOS_FUNCTION
T& ViewFArrayKokkos<T>::operator()(size_t i, size_t j, size_t k,
                                   size_t l, size_t m, size_t n,
                                   size_t o) const {
    assert(i < dim1_ && "i is out of bounds in ViewFArrayKokkos 7D!");
    assert(j < dim2_ && "j is out of bounds in ViewFArrayKokkos 7D!");
    assert(k < dim3_ && "k is out of bounds in ViewFArrayKokkos 7D!");
    assert(l < dim4_ && "l is out of bounds in ViewFArrayKokkos 7D!");
    assert(m < dim5_ && "m is out of bounds in ViewFArrayKokkos 7D!");
    assert(n < dim6_ && "n is out of bounds in ViewFArrayKokkos 7D!");
    assert(o < dim7_ && "o is out of bounds in ViewFArrayKokkos 7D!");
    return this_array_[i + (j * dim1_)
                         + (k * dim1_ * dim2_)
                         + (l * dim1_ * dim2_ * dim3_)
                         + (m * dim1_ * dim2_ * dim3_ * dim4_)
                         + (n * dim1_ * dim2_ * dim3_ * dim4_ * dim5_)
                         + (o * dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_)];
}

template <typename T>
KOKKOS_FUNCTION
size_t ViewFArrayKokkos<T>::size() {
    return length_;
}

template <typename T>
size_t ViewFArrayKokkos<T>::extent() {
    return length_;
}

template <typename T>
KOKKOS_FUNCTION
ViewFArrayKokkos<T>::~ViewFArrayKokkos() {}

////////////////////////////////////////////////////////////////////////////////
// End of ViewFArrayKokkos
////////////////////////////////////////////////////////////////////////////////

/*! \brief Kokkos version of the serial FMatrix class.
 *
 */
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class FMatrixKokkos {

    using TArray1D = Kokkos::View<T*, Layout, ExecSpace, MemoryTraits>;
    
private:

    size_t dim1_;
    size_t dim2_;
    size_t dim3_;
    size_t dim4_;
    size_t dim5_;
    size_t dim6_;
    size_t dim7_;
    size_t length_; 
    TArray1D this_matrix_; 

public:
    FMatrixKokkos();

    FMatrixKokkos(size_t some_dim1);

    FMatrixKokkos(size_t some_dim1, size_t some_dim2);

    FMatrixKokkos(size_t some_dim1, size_t some_dim2, size_t some_dim3);

    FMatrixKokkos(size_t some_dim1, size_t some_dim2, size_t some_dim3, 
                  size_t some_dim4);

    FMatrixKokkos(size_t some_dim1, size_t some_dim2, size_t some_dim3,
                  size_t some_dim4, size_t some_dim5);

    FMatrixKokkos(size_t some_dim1, size_t some_dim2, size_t some_dim3,
                  size_t some_dim4, size_t some_dim5, size_t some_dim6);

    FMatrixKokkos(size_t some_dim1, size_t some_dim2, size_t some_dim3,
                  size_t some_dim4, size_t some_dim5, size_t some_dim6,
                  size_t some_dim7);
    
    KOKKOS_FUNCTION
    T& operator()(size_t i) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m, 
                  size_t n) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m,
                  size_t n, size_t o) const;
    
    FMatrixKokkos& operator=(const FMatrixKokkos& temp);

    KOKKOS_FUNCTION
    size_t size();

    size_t extent();

    T* pointer();
    
    //return kokkos view
    TArray1D get_kokkos_view();

    KOKKOS_FUNCTION
    ~FMatrixKokkos();

}; // End of FMatrixKokkos

// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::FMatrixKokkos() {}

// Overloaded 1D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::FMatrixKokkos(size_t some_dim1) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    length_ = dim1_;
    this_matrix_ = TArray1D("this_matrix_", length_);
}

// Overloaded 2D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::FMatrixKokkos(size_t some_dim1, size_t some_dim2) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    length_ = (dim1_ * dim2_);
    this_matrix_ = TArray1D("this_matrix_", length_);
}

// Overloaded 3D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::FMatrixKokkos(size_t some_dim1, size_t some_dim2, 
                                size_t some_dim3) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    length_ = (dim1_ * dim2_ * dim3_);
    this_matrix_ = TArray1D("this_matrix_", length_);
}

// Overloaded 4D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::FMatrixKokkos(size_t some_dim1, size_t some_dim2, 
                                size_t some_dim3, size_t some_dim4) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_);
    this_matrix_ = TArray1D("this_matrix_", length_);
}

// Overloaded 5D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::FMatrixKokkos(size_t some_dim1, size_t some_dim2, 
                                size_t some_dim3, size_t some_dim4, 
                                size_t some_dim5) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_);
    this_matrix_ = TArray1D("this_matrix_", length_);
}

// Overloaded 5D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::FMatrixKokkos(size_t some_dim1, size_t some_dim2, 
                                size_t some_dim3, size_t some_dim4, 
                                size_t some_dim5, size_t some_dim6) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_);
    this_matrix_ = TArray1D("this_matrix_", length_);
}

// Overloaded 5D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::FMatrixKokkos(size_t some_dim1, size_t some_dim2,
                                size_t some_dim3, size_t some_dim4,
                                size_t some_dim5, size_t some_dim6,
                                size_t some_dim7) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    dim7_ = some_dim7;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_ * dim7_);
    this_matrix_ = TArray1D("this_matrix_", length_);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in FMatrixKokkos in 1D!");
    return this_matrix_((i - 1));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in FMatrixKokkos in 2D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in FMatrixKokkos in 2D!");
    return this_matrix_((i - 1) + ((j - 1) * dim1_));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in FMatrixKokkos in 3D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in FMatrixKokkos in 3D!");
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in FMatrixKokkos in 3D!");
    return this_matrix_((i - 1) + ((j - 1) * dim1_) 
                                + ((k - 1) * dim1_ * dim2_));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in FMatrixKokkos in 4D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in FMatrixKokkos in 4D!");
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in FMatrixKokkos in 4D!");
    assert(l >= 1 && l <= dim4_ && "l is out of bounds in FMatrixKokkos in 4D!");
    return this_matrix_((i - 1) + ((j - 1) * dim1_)  
                                + ((k - 1) * dim1_ * dim2_)  
                                + ((l - 1) * dim1_ * dim2_ * dim3_));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                                size_t m) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in FMatrixKokkos in 5D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in FMatrixKokkos in 5D!");
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in FMatrixKokkos in 5D!");
    assert(l >= 1 && l <= dim4_ && "l is out of bounds in FMatrixKokkos in 5D!");
    assert(m >= 1 && m <= dim5_ && "m is out of bounds in FMatrixKokkos in 5D!");
    return this_matrix_((i - 1) + ((j - 1) * dim1_)  
                                + ((k - 1) * dim1_ * dim2_)  
                                + ((l - 1) * dim1_ * dim2_ * dim3_) 
                                + ((m - 1) * dim1_ * dim2_ * dim3_ * dim4_));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                                size_t m, size_t n) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in FMatrixKokkos in 6D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in FMatrixKokkos in 6D!");
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in FMatrixKokkos in 6D!");
    assert(l >= 1 && l <= dim4_ && "l is out of bounds in FMatrixKokkos in 6D!");
    assert(m >= 1 && m <= dim5_ && "m is out of bounds in FMatrixKokkos in 6D!");
    assert(n >= 1 && n <= dim6_ && "n is out of bounds in FMatrixKokkos in 6D!");
    return this_matrix_[(i - 1) + ((j - 1) * dim1_)  
                                + ((k - 1) * dim1_ * dim2_)  
                                + ((l - 1) * dim1_ * dim2_ * dim3_)  
                                + ((m - 1) * dim1_ * dim2_ * dim3_ * dim4_)  
                                + ((n - 1) * dim1_ * dim2_ * dim3_ * dim4_ * dim5_)];
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                                size_t m, size_t n, size_t o) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in FMatrixKokkos in 7D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in FMatrixKokkos in 7D!");
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in FMatrixKokkos in 7D!");
    assert(l >= 1 && l <= dim4_ && "l is out of bounds in FMatrixKokkos in 7D!");
    assert(m >= 1 && m <= dim5_ && "m is out of bounds in FMatrixKokkos in 7D!");
    assert(n >= 1 && n <= dim6_ && "n is out of bounds in FMatrixKokkos in 7D!");
    assert(o >= 1 && o <= dim7_ && "o is out of bounds in FMatrixKokkos in 7D!");
    return this_matrix_[(i - 1) + ((j - 1) * dim1_)
                                + ((k - 1) * dim1_ * dim2_)
                                + ((l - 1) * dim1_ * dim2_ * dim3_)
                                + ((m - 1) * dim1_ * dim2_ * dim3_ * dim4_)
                                + ((n - 1) * dim1_ * dim2_ * dim3_ * dim4_ * dim5_)
                                + ((o - 1) * dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_)];
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>& FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::operator=(const FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>& temp) {
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        dim1_ = temp.dim1_;
        dim2_ = temp.dim2_;
        dim3_ = temp.dim3_;
        dim4_ = temp.dim4_;
        dim5_ = temp.dim5_;
        dim6_ = temp.dim6_;
        dim7_ = temp.dim7_;
        length_ = temp.length_;
        this_matrix_ = TArray1D("this_matrix_", length_);
    }
    return *this;
}



template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
size_t FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::size() {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
size_t FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::extent() {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T* FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::pointer() {
    return this_matrix_.data();
}

//return the stored Kokkos view
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
Kokkos::View<T*, Layout, ExecSpace, MemoryTraits> FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_view() {
    return this_matrix_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
FMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::~FMatrixKokkos() {}

////////////////////////////////////////////////////////////////////////////////
// End of FMatrixKokkos
////////////////////////////////////////////////////////////////////////////////

/*! \brief Kokkos version of the serial ViewFMatrix class.
 * 
 */
template <typename T>
class ViewFMatrixKokkos {

private:

    size_t dim1_;
    size_t dim2_;
    size_t dim3_;
    size_t dim4_;
    size_t dim5_;
    size_t dim6_;
    size_t dim7_;
    size_t length_; 
    T* this_matrix_;
    
public:
    
    ViewFMatrixKokkos();
    
    ViewFMatrixKokkos(T* some_matrix, size_t some_dim1);

    ViewFMatrixKokkos(T* some_matrix, size_t some_dim1, size_t some_dim2);

    ViewFMatrixKokkos(T* some_matrix, size_t some_dim1, size_t some_dim2,
                      size_t some_dim3);

    ViewFMatrixKokkos(T* some_matrix, size_t some_dim1, size_t some_dim2,
                      size_t some_dim3, size_t some_dim4);

    ViewFMatrixKokkos(T* some_matrix, size_t some_dim1, size_t some_dim2,
                      size_t some_dim3, size_t some_dim4, size_t some_dim5);

    ViewFMatrixKokkos(T* some_matrix, size_t some_dim1, size_t some_dim2, 
                      size_t some_dim3, size_t some_dim4, size_t some_dim5,
                      size_t some_dim6);
    
    ViewFMatrixKokkos(T* some_matrix, size_t some_dim1, size_t some_dim2,
                      size_t some_dim3, size_t some_dim4, size_t some_dim5,
                      size_t some_dim6, size_t some_dim7);
    
    KOKKOS_FUNCTION
    T& operator()(size_t i) const;
    
    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j) const;
    
    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m) const;
        
    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m, 
                  size_t n) const;
 
    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m,
                  size_t n, size_t o) const;
    
    KOKKOS_FUNCTION
    size_t size();

    size_t extent();

    KOKKOS_FUNCTION
    ~ViewFMatrixKokkos();
    
}; // end of ViewFMatrixKokkos

// Default constructor
template <typename T>
ViewFMatrixKokkos<T>::ViewFMatrixKokkos() {}

// Overloaded 1D constructor
template <typename T>
ViewFMatrixKokkos<T>::ViewFMatrixKokkos(T* some_matrix, size_t some_dim1) {
    dim1_ = some_dim1;
    length_ = dim1_;
    this_matrix_ = some_matrix;
}

// Overloaded 2D constructor
template <typename T>
ViewFMatrixKokkos<T>::ViewFMatrixKokkos(T* some_matrix, size_t some_dim1,
                                        size_t some_dim2) {
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    length_ = (dim1_ * dim2_);
    this_matrix_ = some_matrix;
}

// Overloaded 3D constructor
template <typename T>
ViewFMatrixKokkos<T>::ViewFMatrixKokkos(T* some_matrix, size_t some_dim1,
                                        size_t some_dim2, size_t some_dim3) {
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    length_ = (dim1_ * dim2_ * dim3_);
    this_matrix_ = some_matrix;
}

// Overloaded 4D constructor
template <typename T>
ViewFMatrixKokkos<T>::ViewFMatrixKokkos(T* some_matrix, size_t some_dim1,
                                        size_t some_dim2, size_t some_dim3,
                                        size_t some_dim4) {
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_);
    this_matrix_ = some_matrix;
}

// Overloaded 5D constructor
template <typename T>
ViewFMatrixKokkos<T>::ViewFMatrixKokkos(T* some_matrix, size_t some_dim1,
                                        size_t some_dim2, size_t some_dim3,
                                        size_t some_dim4, size_t some_dim5) {
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_);
    this_matrix_ = some_matrix;
}

// Overloaded 6D constructor
template <typename T>
ViewFMatrixKokkos<T>::ViewFMatrixKokkos(T* some_matrix, size_t some_dim1,
                                        size_t some_dim2, size_t some_dim3,
                                        size_t some_dim4, size_t some_dim5,
                                        size_t some_dim6) {
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_);
    this_matrix_ = some_matrix;
}

// Overloaded 6D constructor
template <typename T>
ViewFMatrixKokkos<T>::ViewFMatrixKokkos(T* some_matrix, size_t some_dim1,
                                        size_t some_dim2, size_t some_dim3,
                                        size_t some_dim4, size_t some_dim5,
                                        size_t some_dim6, size_t some_dim7) {
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    dim7_ = some_dim7;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_ * dim7_);
    this_matrix_ = some_matrix;
}


template <typename T>
KOKKOS_FUNCTION
T& ViewFMatrixKokkos<T>::operator()(size_t i) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in ViewFMatrixKokkos 1D!"); 
    return this_matrix_[(i - 1)];
}

template <typename T>
KOKKOS_FUNCTION
T& ViewFMatrixKokkos<T>::operator()(size_t i, size_t j) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in ViewFMatrixKokkos 2D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in ViewFMatrixKokkos 2D!");  
    return this_matrix_[(i - 1) + ((j - 1) * dim1_)];
}

template <typename T>
KOKKOS_FUNCTION
T& ViewFMatrixKokkos<T>::operator()(size_t i, size_t j, size_t k) const
{
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in ViewFMatrixKokkos 3D!");  
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in ViewFMatrixKokkos 3D!");  
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in ViewFMatrixKokkos 3D!"); 
    
    return this_matrix_[(i - 1) + ((j - 1) * dim1_) 
                                + ((k - 1) * dim1_ * dim2_)];
}

template <typename T>
KOKKOS_FUNCTION
T& ViewFMatrixKokkos<T>::operator()(size_t i, size_t j, size_t k, 
                                    size_t l) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in ViewFMatrixKokkos 4D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in ViewFMatrixKokkos 4D!");
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in ViewFMatrixKokkos 4D!");
    assert(l >= 1 && l <= dim4_ && "l is out of bounds in ViewFMatrixKokkos 4D!");
    return this_matrix_[(i - 1) + ((j - 1) * dim1_) 
                                + ((k - 1) * dim1_ * dim2_)
                                + ((l - 1) * dim1_ * dim2_ * dim3_)];
}

template <typename T>
KOKKOS_FUNCTION
T& ViewFMatrixKokkos<T>::operator()(size_t i, size_t j, size_t k, size_t l, 
                                    size_t m) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in ViewFMatrixKokkos 5D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in ViewFMatrixKokkos 5D!");
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in ViewFMatrixKokkos 5D!");
    assert(l >= 1 && l <= dim4_ && "l is out of bounds in ViewFMatrixKokkos 5D!");
    assert(m >= 1 && m <= dim5_ && "m is out of bounds in ViewFMatrixKokkos 5D!");
    return this_matrix_[(i - 1) + ((j - 1) * dim1_) 
                                + ((k - 1) * dim1_ * dim2_) 
                                + ((l - 1) * dim1_ * dim2_ * dim3_)
                                + ((m - 1) * dim1_ * dim2_ * dim3_ * dim4_)];
}

template <typename T>
KOKKOS_FUNCTION
T& ViewFMatrixKokkos<T>::operator()(size_t i, size_t j, size_t k, size_t l, 
                                    size_t m, size_t n) const
{
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in ViewFMatrixKokkos 6D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in ViewFMatrixKokkos 6D!");
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in ViewFMatrixKokkos 6D!");
    assert(l >= 1 && l <= dim4_ && "l is out of bounds in ViewFMatrixKokkos 6D!");
    assert(m >= 1 && m <= dim5_ && "m is out of bounds in ViewFMatrixKokkos 6D!");
    assert(n >= 1 && n <= dim6_ && "n is out of bounds in ViewFMatrixKokkos 6D!");
    return this_matrix_[(i - 1) + ((j - 1) * dim1_) 
                                + ((k - 1) * dim1_ * dim2_) 
                                + ((l - 1) * dim1_ * dim2_ * dim3_)
                                + ((m - 1) * dim1_ * dim2_ * dim3_ * dim4_)
                                + ((n - 1) * dim1_ * dim2_ * dim3_ * dim4_ * dim5_)];
}

template <typename T>
KOKKOS_FUNCTION
T& ViewFMatrixKokkos<T>::operator()(size_t i, size_t j, size_t k, size_t l,
                                    size_t m, size_t n, size_t o) const
{
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in ViewFMatrixKokkos 7D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in ViewFMatrixKokkos 7D!");
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in ViewFMatrixKokkos 7D!");
    assert(l >= 1 && l <= dim4_ && "l is out of bounds in ViewFMatrixKokkos 7D!");
    assert(m >= 1 && m <= dim5_ && "m is out of bounds in ViewFMatrixKokkos 7D!");
    assert(n >= 1 && n <= dim6_ && "n is out of bounds in ViewFMatrixKokkos 7D!");
    assert(o >= 1 && o <= dim7_ && "o is out of bounds in ViewFMatrixKokkos 7D!");
    return this_matrix_[(i - 1) + ((j - 1) * dim1_)
                                + ((k - 1) * dim1_ * dim2_)
                                + ((l - 1) * dim1_ * dim2_ * dim3_)
                                + ((m - 1) * dim1_ * dim2_ * dim3_ * dim4_)
                                + ((n - 1) * dim1_ * dim2_ * dim3_ * dim4_ * dim5_)
                                + ((o - 1) * dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_)];
}

template <typename T>
KOKKOS_FUNCTION
size_t ViewFMatrixKokkos<T>::size() {
    return length_;
}

template <typename T>
size_t ViewFMatrixKokkos<T>::extent() {
    return length_;
}

template <typename T>
KOKKOS_FUNCTION
ViewFMatrixKokkos<T>::~ViewFMatrixKokkos() {}

////////////////////////////////////////////////////////////////////////////////
// End of ViewFMatrixKokkos
////////////////////////////////////////////////////////////////////////////////

/*! \brief Kokkos version of the serial CArray class.
 *
 */
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class CArrayKokkos {

    using TArray1D = Kokkos::View<T*, Layout, ExecSpace, MemoryTraits>;
    
private:
    size_t dim1_;
    size_t dim2_;
    size_t dim3_;
    size_t dim4_;
    size_t dim5_;
    size_t dim6_;
    size_t dim7_;
    size_t length_;
    TArray1D this_array_; 

public:
    CArrayKokkos();
    
    CArrayKokkos(size_t some_dim1);

    CArrayKokkos(size_t some_dim1, size_t some_dim2);

    CArrayKokkos (size_t some_dim1, size_t some_dim2, size_t some_dim3);

    CArrayKokkos(size_t some_dim1, size_t some_dim2, size_t some_dim3, 
                 size_t some_dim4);

    CArrayKokkos(size_t some_dim1, size_t some_dim2, size_t some_dim3,
                 size_t some_dim4, size_t some_dim5);

    CArrayKokkos(size_t some_dim1, size_t some_dim2, size_t some_dim3,
                 size_t some_dim4, size_t some_dim5, size_t some_dim6);

    CArrayKokkos(size_t some_dim1, size_t some_dim2, size_t some_dim3,
                 size_t some_dim4, size_t some_dim5, size_t some_dim6,
                 size_t some_dim7);
    
    KOKKOS_FUNCTION
    T& operator()(size_t i) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m, 
                  size_t n) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m,
                  size_t n, size_t o) const;
    
    CArrayKokkos& operator=(const CArrayKokkos& temp);

    // GPU Method
    // Method that returns size
    KOKKOS_FUNCTION
    size_t size();

    // Host Method
    // Method that returns size
    size_t extent();

    // Methods returns the raw pointer (most likely GPU) of the Kokkos View
    T* pointer();
    
    //return the view
    TArray1D get_kokkos_view();

    // Deconstructor
    KOKKOS_FUNCTION
    ~CArrayKokkos ();
}; // End of CArrayKokkos

// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::CArrayKokkos() {}

// Overloaded 1D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::CArrayKokkos(size_t some_dim1) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    length_ = dim1_;
    this_array_ = TArray1D("this_array_", length_);
}

// Overloaded 2D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::CArrayKokkos(size_t some_dim1, size_t some_dim2) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    length_ = (dim1_ * dim2_);
    this_array_ = TArray1D("this_array_", length_);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::CArrayKokkos(size_t some_dim1, size_t some_dim2, 
                              size_t some_dim3) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    length_ = (dim1_ * dim2_ * dim3_);
    this_array_ = TArray1D("this_array_", length_);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::CArrayKokkos(size_t some_dim1, size_t some_dim2, 
                              size_t some_dim3, size_t some_dim4) {
    using TArray1D = Kokkos::View<T *,Layout,ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_);
    this_array_ = TArray1D("this_array_", length_);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::CArrayKokkos(size_t some_dim1, size_t some_dim2, 
                              size_t some_dim3, size_t some_dim4, 
                              size_t some_dim5) {

    using TArray1D = Kokkos::View<T *,Layout,ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_);
    this_array_ = TArray1D("this_array_", length_);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::CArrayKokkos(size_t some_dim1, size_t some_dim2, 
                              size_t some_dim3, size_t some_dim4, 
                              size_t some_dim5, size_t some_dim6) {
    using TArray1D = Kokkos::View<T *,Layout,ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_);
    this_array_ = TArray1D("this_array_", length_);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::CArrayKokkos(size_t some_dim1, size_t some_dim2,
                              size_t some_dim3, size_t some_dim4,
                              size_t some_dim5, size_t some_dim6,
                              size_t some_dim7) {
    using TArray1D = Kokkos::View<T *,Layout,ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    dim7_ = some_dim7;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_ * dim7_);
    this_array_ = TArray1D("this_array_", length_);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i) const {
    assert(i < dim1_ && "i is out of bounds in CArrayKokkos 1D!");
    return this_array_(i);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j) const {
    assert(i < dim1_ && "i is out of bounds in CArrayKokkos 2D!");
    assert(j < dim2_ && "j is out of bounds in CArrayKokkos 2D!");
    return this_array_(j + (i * dim2_));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k) const {
    assert(i < dim1_ && "i is out of bounds in CArrayKokkos 3D!");
    assert(j < dim2_ && "j is out of bounds in CArrayKokkos 3D!");
    assert(k < dim3_ && "k is out of bounds in CArrayKokkos 3D!");
    return this_array_(k + (j * dim3_) 
                         + (i * dim3_ * dim2_));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l) const {
    assert(i < dim1_ && "i is out of bounds in CArrayKokkos 4D!");
    assert(j < dim2_ && "j is out of bounds in CArrayKokkos 4D!");
    assert(k < dim3_ && "k is out of bounds in CArrayKokkos 4D!");
    assert(l < dim4_ && "l is out of bounds in CArrayKokkos 4D!");
    return this_array_(l + (k * dim4_) 
                         + (j * dim4_ * dim3_)  
                         + (i * dim4_ * dim3_ * dim2_));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m) const {
    assert(i < dim1_ && "i is out of bounds in CArrayKokkos 5D!");
    assert(j < dim2_ && "j is out of bounds in CArrayKokkos 5D!");
    assert(k < dim3_ && "k is out of bounds in CArrayKokkos 5D!");
    assert(l < dim4_ && "l is out of bounds in CArrayKokkos 5D!");
    assert(m < dim5_ && "m is out of bounds in CArrayKokkos 5D!");
    return this_array_(m + (l * dim5_) 
                         + (k * dim5_ * dim4_) 
                         + (j * dim5_ * dim4_ * dim3_) 
                         + (i * dim5_ * dim4_ * dim3_ * dim2_));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n) const {
    assert(i < dim1_ && "i is out of bounds in CArrayKokkos 6D!");
    assert(j < dim2_ && "j is out of bounds in CArrayKokkos 6D!");
    assert(k < dim3_ && "k is out of bounds in CArrayKokkos 6D!");
    assert(l < dim4_ && "l is out of bounds in CArrayKokkos 6D!");
    assert(m < dim5_ && "m is out of bounds in CArrayKokkos 6D!");
    assert(n < dim6_ && "n is out of bounds in CArrayKokkos 6D!");
    return this_array_(n + (m * dim6_) 
                         + (l * dim6_ * dim5_)  
                         + (k * dim6_ * dim5_ * dim4_) 
                         + (j * dim6_ * dim5_ * dim4_ * dim3_)  
                         + (i * dim6_ * dim5_ * dim4_ * dim3_ * dim2_));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n, size_t o) const {
    assert(i < dim1_ && "i is out of bounds in CArrayKokkos 7D!");
    assert(j < dim2_ && "j is out of bounds in CArrayKokkos 7D!");
    assert(k < dim3_ && "k is out of bounds in CArrayKokkos 7D!");
    assert(l < dim4_ && "l is out of bounds in CArrayKokkos 7D!");
    assert(m < dim5_ && "m is out of bounds in CArrayKokkos 7D!");
    assert(n < dim6_ && "n is out of bounds in CArrayKokkos 7D!");
    assert(o < dim7_ && "o is out of bounds in CArrayKokkos 7D!");
    return this_array_(o + (n * dim7_)
                         + (m * dim7_ * dim6_)
                         + (l * dim7_ * dim6_ * dim5_)
                         + (k * dim7_ * dim6_ * dim5_ * dim4_)
                         + (j * dim7_ * dim6_ * dim5_ * dim4_ * dim3_)
                         + (i * dim7_ * dim6_ * dim5_ * dim4_ * dim3_ * dim2_));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>& CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator= (const CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>& temp) {
    using TArray1D = Kokkos::View<T *,Layout,ExecSpace>;
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        dim1_ = temp.dim1_;
        dim2_ = temp.dim2_;
        dim3_ = temp.dim3_;
        dim4_ = temp.dim4_;
        dim5_ = temp.dim5_;
        dim6_ = temp.dim6_;
        dim7_ = temp.dim7_;
        length_ = temp.length_;
        this_array_ = TArray1D("this_array_", length_);
    }
    
    return *this;
}

// Return size
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
size_t CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::size() {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
size_t CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::extent() {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T* CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::pointer() {
    return this_array_.data();
}

//return the stored Kokkos view
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
Kokkos::View<T*, Layout, ExecSpace, MemoryTraits> CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_view() {
    return this_array_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
CArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::~CArrayKokkos() {}

////////////////////////////////////////////////////////////////////////////////
// End of CArrayKokkos
////////////////////////////////////////////////////////////////////////////////

/*! \brief Kokkos version of the serial ViewCArray class.
 *
 */
template <typename T>
class ViewCArrayKokkos {

private:
    size_t dim1_;
    size_t dim2_;
    size_t dim3_;
    size_t dim4_;
    size_t dim5_;
    size_t dim6_;
    size_t dim7_;
    size_t length_;  // Length of 1D array
    T* this_array_;
    
public:
    ViewCArrayKokkos();

    ViewCArrayKokkos(T* some_array, size_t some_dim1);

    ViewCArrayKokkos(T* some_array, size_t some_dim1, size_t some_dim2);

    ViewCArrayKokkos(T* some_array, size_t some_dim1, size_t some_dim2,
                     size_t some_dim3);

    ViewCArrayKokkos(T* some_array, size_t some_dim1, size_t some_dim2,
                     size_t some_dim3, size_t some_dim4);

    ViewCArrayKokkos(T* some_array, size_t some_dim1, size_t some_dim2,
                     size_t some_dim3, size_t some_dim4, size_t some_dim5);

    ViewCArrayKokkos(T* some_array, size_t some_dim1, size_t some_dim2,
                     size_t some_dim3, size_t some_dim4, size_t some_dim5,
                     size_t some_dim6);
    
    ViewCArrayKokkos(T* some_array, size_t some_dim1, size_t some_dim2,
                     size_t some_dim3, size_t some_dim4, size_t some_dim5,
                     size_t some_dim6, size_t some_dim7);
    
    KOKKOS_FUNCTION
    T& operator()(size_t i) const;
    
    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j) const;
    
    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l) const;
        
    KOKKOS_FUNCTION
    T& operator() (size_t i, size_t j, size_t k, size_t l, size_t m) const;
        
    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m, 
                  size_t n) const;
 
    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m,
                  size_t n, size_t o) const;
    
    KOKKOS_FUNCTION
    size_t size();

    size_t extent();

    KOKKOS_FUNCTION
    ~ViewCArrayKokkos();
    
}; // end of ViewCArrayKokkos

// Default constructor
template <typename T>
ViewCArrayKokkos<T>::ViewCArrayKokkos() {}

// Overloaded 1D constructor
template <typename T>
ViewCArrayKokkos<T>::ViewCArrayKokkos(T* some_array, size_t some_dim1) {
    dim1_ = some_dim1;
    length_ = dim1_;
    this_array_ = some_array;
}

// Overloaded 2D constructor
template <typename T>
ViewCArrayKokkos<T>::ViewCArrayKokkos(T* some_array, size_t some_dim1, 
                                      size_t some_dim2) {
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    length_ = (dim1_ * dim2_);
    this_array_ = some_array;
}

// Overloaded 3D constructor
template <typename T>
ViewCArrayKokkos<T>::ViewCArrayKokkos(T* some_array, size_t some_dim1,
                                      size_t some_dim2, size_t some_dim3) {
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    length_ = (dim1_ * dim2_ * dim3_);
    this_array_ = some_array;
}

// Overloaded 4D constructor
template <typename T>
ViewCArrayKokkos<T>::ViewCArrayKokkos(T* some_array, size_t some_dim1,
                                      size_t some_dim2, size_t some_dim3,
                                      size_t some_dim4) {
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_);
    this_array_ = some_array;
}

// Overloaded 5D constructor
template <typename T>
ViewCArrayKokkos<T>::ViewCArrayKokkos(T* some_array, size_t some_dim1,
                                      size_t some_dim2, size_t some_dim3,
                                      size_t some_dim4, size_t some_dim5) {
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_);
    this_array_ = some_array;
}

// Overloaded 6D constructor
template <typename T>
ViewCArrayKokkos<T>::ViewCArrayKokkos(T* some_array, size_t some_dim1,
                                      size_t some_dim2, size_t some_dim3,
                                      size_t some_dim4, size_t some_dim5,
                                      size_t some_dim6) {
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_);
    this_array_ = some_array;
}

// Overloaded 7D constructor
template <typename T>
ViewCArrayKokkos<T>::ViewCArrayKokkos(T* some_array, size_t some_dim1,
                                      size_t some_dim2, size_t some_dim3,
                                      size_t some_dim4, size_t some_dim5,
                                      size_t some_dim6, size_t some_dim7) {
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    dim7_ = some_dim7;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_ * dim7_);
    this_array_ = some_array;
}


template <typename T>
KOKKOS_FUNCTION
T& ViewCArrayKokkos<T>::operator()(size_t i) const {
    assert(i < dim1_ && "i is out of bounds in ViewCArrayKokkos 1D!");
    return this_array_[i];
}

template <typename T>
KOKKOS_FUNCTION
T& ViewCArrayKokkos<T>::operator()(size_t i, size_t j) const {
    assert(i < dim1_ && "i is out of bounds in ViewCArrayKokkos 2D!");
    assert(j < dim2_ && "j is out of bounds in ViewCArrayKokkos 2D!");  
    return this_array_[j + (i * dim2_)];
}

template <typename T>
KOKKOS_FUNCTION
T& ViewCArrayKokkos<T>::operator()(size_t i, size_t j, size_t k) const {
    assert(i < dim1_ && "i is out of bounds in ViewCArrayKokkos 3D!");
    assert(j < dim2_ && "j is out of bounds in ViewCArrayKokkos 3D!");
    assert(k < dim3_ && "k is out of bounds in ViewCArrayKokkos 3D!");
    return this_array_[k + (j * dim3_) 
                         + (i * dim3_ * dim2_)];
}

template <typename T>
KOKKOS_FUNCTION
T& ViewCArrayKokkos<T>::operator()(size_t i, size_t j, size_t k, 
                                   size_t l) const {
    assert(i < dim1_ && "i is out of bounds in ViewCArrayKokkos 4D!");
    assert(j < dim2_ && "j is out of bounds in ViewCArrayKokkos 4D!");
    assert(k < dim3_ && "k is out of bounds in ViewCArrayKokkos 4D!");
    assert(l < dim4_ && "l is out of bounds in ViewCArrayKokkos 4D!");
    return this_array_[l + (k * dim4_) 
                         + (j * dim4_ * dim3_) 
                         + (i * dim4_ * dim3_ * dim2_)];
}

template <typename T>
KOKKOS_FUNCTION
T& ViewCArrayKokkos<T>::operator()(size_t i, size_t j, size_t k, size_t l, 
                                   size_t m) const {
    assert(i < dim1_ && "i is out of bounds in ViewCArrayKokkos 5D!");
    assert(j < dim2_ && "j is out of bounds in ViewCArrayKokkos 5D!");
    assert(k < dim3_ && "k is out of bounds in ViewCArrayKokkos 5D!");
    assert(l < dim4_ && "l is out of bounds in ViewCArrayKokkos 5D!");
    assert(m < dim5_ && "m is out of bounds in ViewCArrayKokkos 5D!");
    return this_array_[m + (l * dim5_) 
                         + (k * dim5_ * dim4_) 
                         + (j * dim5_ * dim4_ * dim3_)
                         + (i * dim5_ * dim4_ * dim3_ * dim2_)];
}

template <typename T>
KOKKOS_FUNCTION
T& ViewCArrayKokkos<T>::operator()(size_t i, size_t j, size_t k, size_t l, 
                                   size_t m, size_t n) const {
    assert(i < dim1_ && "i is out of bounds in ViewCArrayKokkos 6D!");
    assert(j < dim2_ && "j is out of bounds in ViewCArrayKokkos 6D!");
    assert(k < dim3_ && "k is out of bounds in ViewCArrayKokkos 6D!");
    assert(l < dim4_ && "l is out of bounds in ViewCArrayKokkos 6D!");
    assert(m < dim5_ && "m is out of bounds in ViewCArrayKokkos 6D!");
    assert(n < dim6_ && "n is out of bounds in ViewCArrayKokkos 6D!");
    return this_array_[n + (m * dim6_) 
                         + (l * dim6_ * dim5_) 
                         + (k * dim6_ * dim5_ * dim4_)
                         + (j * dim6_ * dim5_ * dim4_ * dim3_) 
                         + (i * dim6_ * dim5_ * dim4_ * dim3_ * dim2_)];
}

template <typename T>
KOKKOS_FUNCTION
T& ViewCArrayKokkos<T>::operator()(size_t i, size_t j, size_t k, size_t l,
                                   size_t m, size_t n, size_t o) const {
    assert(i < dim1_ && "i is out of bounds in ViewCArrayKokkos 7D!");
    assert(j < dim2_ && "j is out of bounds in ViewCArrayKokkos 7D!");
    assert(k < dim3_ && "k is out of bounds in ViewCArrayKokkos 7D!");
    assert(l < dim4_ && "l is out of bounds in ViewCArrayKokkos 7D!");
    assert(m < dim5_ && "m is out of bounds in ViewCArrayKokkos 7D!");
    assert(n < dim6_ && "n is out of bounds in ViewCArrayKokkos 7D!");
    assert(o < dim7_ && "o is out of bounds in ViewCArrayKokkos 7D!");
    return this_array_[o + (n * dim7_)
                         + (m * dim7_ * dim6_)
                         + (l * dim7_ * dim6_ * dim5_)
                         + (k * dim7_ * dim6_ * dim5_ * dim4_)
                         + (j * dim7_ * dim6_ * dim5_ * dim4_ * dim3_)
                         + (i * dim7_ * dim6_ * dim5_ * dim4_ * dim3_ * dim2_)];
}

template <typename T>
KOKKOS_FUNCTION
size_t ViewCArrayKokkos<T>::size() {
    return length_;
}

template <typename T>
size_t ViewCArrayKokkos<T>::extent() {
    return length_;
}

template <typename T>
KOKKOS_FUNCTION
ViewCArrayKokkos<T>::~ViewCArrayKokkos() {}

////////////////////////////////////////////////////////////////////////////////
// End of ViewCArrayKokkos
////////////////////////////////////////////////////////////////////////////////

/*! \brief Kokkos version of the serial CMatrix class.
 *
 */
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class CMatrixKokkos {

    using TArray1D = Kokkos::View<T*, Layout, ExecSpace, MemoryTraits>;
    
private:
    size_t dim1_;
    size_t dim2_;
    size_t dim3_;
    size_t dim4_;
    size_t dim5_;
    size_t dim6_;
    size_t dim7_;
    size_t length_;
    TArray1D this_matrix_; 

public:
    CMatrixKokkos();

    CMatrixKokkos(size_t some_dim1);

    CMatrixKokkos(size_t some_dim1, size_t some_dim2);

    CMatrixKokkos(size_t some_dim1, size_t some_dim2, size_t some_dim3);    

    CMatrixKokkos(size_t some_dim1, size_t some_dim2, size_t some_dim3, 
                  size_t some_dim4);

    CMatrixKokkos(size_t some_dim1, size_t some_dim2, size_t some_dim3, 
                  size_t some_dim4, size_t some_dim5);

    CMatrixKokkos(size_t some_dim1, size_t some_dim2, size_t some_dim3, 
                  size_t some_dim4, size_t some_dim5, size_t some_dim6);

    CMatrixKokkos(size_t some_dim1, size_t some_dim2, size_t some_dim3,
                  size_t some_dim4, size_t some_dim5, size_t some_dim6,
                  size_t some_dim7);
    
    KOKKOS_FUNCTION
    T& operator()(size_t i) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m, 
                  size_t n) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m,
                  size_t n, size_t o) const;
    
    CMatrixKokkos& operator=(const CMatrixKokkos &temp);

    KOKKOS_FUNCTION
    size_t size();

    size_t extent();

    T* pointer();

    //return the view
    TArray1D get_kokkos_view();

    KOKKOS_FUNCTION
    ~CMatrixKokkos();

}; // End of CMatrixKokkos

// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::CMatrixKokkos() {}

// Overloaded 1D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::CMatrixKokkos(size_t some_dim1) { 
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    length_ = dim1_;
    this_matrix_ = TArray1D("this_matrix_", length_);
}

// Overloaded 2D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::CMatrixKokkos(size_t some_dim1, size_t some_dim2) { 
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    length_ = (dim1_ * dim2_);
    this_matrix_ = TArray1D("this_matrix_", length_);
}

// Overloaded 3D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::CMatrixKokkos(size_t some_dim1, size_t some_dim2, 
                                size_t some_dim3) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    length_ = (dim1_ * dim2_ * dim3_);
    this_matrix_ = TArray1D("this_matrix_", length_);
}

// Overloaded 4D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::CMatrixKokkos(size_t some_dim1, size_t some_dim2, 
                                size_t some_dim3, size_t some_dim4) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_);
    this_matrix_ = TArray1D("this_matrix_", length_);
}

// Overloaded 5D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::CMatrixKokkos(size_t some_dim1, size_t some_dim2, 
                                size_t some_dim3, size_t some_dim4, 
                                size_t some_dim5) {

    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_);
    this_matrix_ = TArray1D("this_matrix_", length_);
}

// Overloaded 6D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::CMatrixKokkos(size_t some_dim1, size_t some_dim2, 
                                size_t some_dim3, size_t some_dim4, 
                                size_t some_dim5, size_t some_dim6) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_);
    this_matrix_ = TArray1D("this_matrix_", length_);
}

// Overloaded 7D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::CMatrixKokkos(size_t some_dim1, size_t some_dim2,
                                size_t some_dim3, size_t some_dim4,
                                size_t some_dim5, size_t some_dim6,
                                size_t some_dim7) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    dim7_ = some_dim7;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_ * dim7_);
    this_matrix_ = TArray1D("this_matrix_", length_);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in CMatrixKokkos 1D!");
    return this_matrix_((i - 1));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in CMatrixKokkos 2D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in CMatrixKokkos 2D!");
    return this_matrix_((j - 1) + ((i - 1) * dim2_));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in CMatrixKokkos 3D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in CMatrixKokkos 3D!");
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in CMatrixKokkos 3D!");
    return this_matrix_((k - 1) + ((j - 1) * dim3_) 
                                + ((i - 1) * dim3_ * dim2_));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in CMatrixKokkos 4D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in CMatrixKokkos 4D!");
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in CMatrixKokkos 4D!");
    assert(l >= 1 && l <= dim4_ && "l is out of bounds in CMatrixKokkos 4D!");
    return this_matrix_((l - 1) + ((k - 1) * dim4_) 
                                + ((j - 1) * dim4_ * dim3_) 
                                + ((i - 1) * dim4_ * dim3_ * dim2_));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l, 
                                size_t m) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in CMatrixKokkos 5D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in CMatrixKokkos 5D!");
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in CMatrixKokkos 5D!");
    assert(l >= 1 && l <= dim4_ && "l is out of bounds in CMatrixKokkos 5D!");
    assert(m >= 1 && m <= dim5_ && "m is out of bounds in CMatrixKokkos 5D!");
    return this_matrix_((m - 1) + ((l - 1) * dim5_) 
                                + ((k - 1) * dim5_ * dim4_) 
                                + ((j - 1) * dim5_ * dim4_ * dim3_) 
                                + ((i - 1) * dim5_ * dim4_ * dim3_ * dim2_));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l, 
                                size_t m, size_t n) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in CMatrixKokkos 6D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in CMatrixKokkos 6D!");
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in CMatrixKokkos 6D!");
    assert(l >= 1 && l <= dim4_ && "l is out of bounds in CMatrixKokkos 6D!");
    assert(m >= 1 && m <= dim5_ && "m is out of bounds in CMatrixKokkos 6D!");
    assert(n >= 1 && n <= dim6_ && "n is out of bounds in CMatrixKokkos 6D!");
    return this_matrix_((n - 1) + ((m - 1) * dim6_) 
                                + ((l - 1) * dim6_ * dim5_) 
                                + ((k - 1) * dim6_ * dim5_ * dim4_) 
                                + ((j - 1) * dim6_ * dim5_ * dim4_ * dim3_) 
                                + ((i - 1) * dim6_ * dim5_ * dim4_ * dim3_ * dim2_));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                                size_t m, size_t n, size_t o) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in CMatrixKokkos 7D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in CMatrixKokkos 7D!");
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in CMatrixKokkos 7D!");
    assert(l >= 1 && l <= dim4_ && "l is out of bounds in CMatrixKokkos 7D!");
    assert(m >= 1 && m <= dim5_ && "m is out of bounds in CMatrixKokkos 7D!");
    assert(n >= 1 && n <= dim6_ && "n is out of bounds in CMatrixKokkos 7D!");
    assert(o >= 1 && o <= dim7_ && "o is out of bounds in CMatrixKokkos 7D!");
    return this_matrix_((o-1) + ((n - 1) * dim7_)
                              + ((m - 1) * dim7_ * dim6_)
                              + ((l - 1) * dim7_ * dim6_ * dim5_)
                              + ((k - 1) * dim7_ * dim6_ * dim5_ * dim4_)
                              + ((j - 1) * dim7_ * dim6_ * dim5_ * dim4_ * dim3_)
                              + ((i - 1) * dim7_ * dim6_ * dim5_ * dim4_ * dim3_ * dim2_));
}

// Overload = operator
// for object assignment THIS = CMatrixKokkos <> temp
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits> & CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::operator=(const CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits> &temp) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;

    if( this != &temp) {
        dim1_ = temp.dim1_;
        dim2_ = temp.dim2_;
        dim3_ = temp.dim3_;
        dim4_ = temp.dim4_;
        dim5_ = temp.dim5_;
        dim6_ = temp.dim6_;
        dim7_ = temp.dim7_;
        length_ = temp.length_;
        this_matrix_ = TArray1D("this_matrix_", length_);
    }
    
    return *this;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
size_t CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::size() {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
size_t CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::extent() {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T* CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::pointer() {
    return this_matrix_.data();
}

//return the stored Kokkos view
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
Kokkos::View<T*, Layout, ExecSpace, MemoryTraits> CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_view() {
    return this_matrix_;
}

// Deconstructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
CMatrixKokkos<T,Layout,ExecSpace,MemoryTraits>::~CMatrixKokkos() {}

////////////////////////////////////////////////////////////////////////////////
// End of CMatrixKokkos
////////////////////////////////////////////////////////////////////////////////

/*! \brief Kokkos version of the serial ViewCMatrix class.
 *
 */
template <typename T>
class ViewCMatrixKokkos {

private:
    size_t dim1_;
    size_t dim2_;
    size_t dim3_;
    size_t dim4_;
    size_t dim5_;
    size_t dim6_;
    size_t dim7_;
    size_t length_;
    T* this_matrix_;

public:
    ViewCMatrixKokkos();

    ViewCMatrixKokkos(T* some_matrix, size_t dim1);

    ViewCMatrixKokkos(T* some_matrix, size_t dim1, size_t dim2);

    ViewCMatrixKokkos(T* some_matrix, size_t dim1, size_t dim2, size_t dim3);

    ViewCMatrixKokkos(T* some_matrix, size_t dim1, size_t dim2, size_t dim3, 
                      size_t dim4);

    ViewCMatrixKokkos(T* some_matrix, size_t dim1, size_t dim2, size_t dim3, 
                      size_t dim4, size_t dim5);

    ViewCMatrixKokkos(T* some_matrix, size_t dim1, size_t dim2, size_t dim3,
                      size_t dim4, size_t dim5, size_t dim6);

    ViewCMatrixKokkos(T* some_matrix, size_t dim1, size_t dim2, size_t dim3,
                      size_t dim4, size_t dim5, size_t dim6, size_t dim7);
    
    KOKKOS_FUNCTION
    T& operator()(size_t i) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j , size_t k) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k , size_t l) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m, size_t n, size_t o) const;

    KOKKOS_FUNCTION
    size_t size();

    size_t extent();

    KOKKOS_FUNCTION
    ~ViewCMatrixKokkos();

}; // End of ViewCMatrixKokkos

// Default constructor
template <typename T>
ViewCMatrixKokkos<T>::ViewCMatrixKokkos(){ }

// Overloaded 1D constructor
template <typename T>
ViewCMatrixKokkos<T>::ViewCMatrixKokkos(T* some_matrix, size_t dim1) {
    dim1_ = dim1;
    length_ = dim1_;
    this_matrix_ = some_matrix;
}

// Overloaded 2D constructor
template <typename T>
ViewCMatrixKokkos<T>::ViewCMatrixKokkos(T* some_matrix, size_t dim1, 
                                        size_t dim2) {
    dim1_ = dim1;
    dim2_ = dim2;
    length_ = (dim1_ * dim2_);
    this_matrix_ = some_matrix;
}

// Overloaded 3D constructor
template <typename T>
ViewCMatrixKokkos<T>::ViewCMatrixKokkos(T* some_matrix, size_t dim1, size_t dim2,
                                        size_t dim3) {
    dim1_ = dim1;
    dim2_ = dim2;
    dim3_ = dim3;
    length_ = (dim1_ * dim2_ * dim3_);
    this_matrix_ = some_matrix;
}

// Overloaded 4D constructor
template <typename T>
ViewCMatrixKokkos<T>::ViewCMatrixKokkos(T* some_matrix, size_t dim1, size_t dim2,
                                        size_t dim3, size_t dim4) {
    dim1_ = dim1;
    dim2_ = dim2;
    dim3_ = dim3;
    dim4_ = dim4;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_);
    this_matrix_ = some_matrix;
}

// Overloaded 5D constructor
template <typename T>
ViewCMatrixKokkos<T>::ViewCMatrixKokkos(T* some_matrix, size_t dim1, size_t dim2,
                                        size_t dim3, size_t dim4, size_t dim5) {
    dim1_ = dim1;
    dim2_ = dim2;
    dim3_ = dim3;
    dim4_ = dim4;
    dim5_ = dim5;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_);
    this_matrix_ = some_matrix;
}

// Overloaded 6D constructor
template <typename T>
ViewCMatrixKokkos<T>::ViewCMatrixKokkos(T* some_matrix, size_t dim1, size_t dim2,
                                        size_t dim3, size_t dim4, size_t dim5,
                                        size_t dim6) {
    dim1_ = dim1;
    dim2_ = dim2;
    dim3_ = dim3;
    dim4_ = dim4;
    dim5_ = dim5;
    dim6_ = dim6;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_);
    this_matrix_ = some_matrix;
}

// Overloaded 7D constructor
template <typename T>
ViewCMatrixKokkos<T>::ViewCMatrixKokkos(T* some_matrix, size_t dim1, size_t dim2,
                                        size_t dim3, size_t dim4, size_t dim5,
                                        size_t dim6, size_t dim7) {
    dim1_ = dim1;
    dim2_ = dim2;
    dim3_ = dim3;
    dim4_ = dim4;
    dim5_ = dim5;
    dim6_ = dim6;
    dim7_ = dim7;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_ * dim7_);
    this_matrix_ = some_matrix;
}

template <typename T>
KOKKOS_FUNCTION
T& ViewCMatrixKokkos<T>::operator()(size_t i) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in ViewCMatrixKokkos 1D!");
    return this_matrix_[(i - 1)];
}

template <typename T>
KOKKOS_FUNCTION
T& ViewCMatrixKokkos<T>::operator()(size_t i, size_t j) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in ViewCMatrixKokkos 2D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in ViewCMatrixKokkos 2D!");
    return this_matrix_[(j - 1) + ((i - 1) * dim2_)];
}

template <typename T>
KOKKOS_FUNCTION
T& ViewCMatrixKokkos<T>::operator()(size_t i, size_t j, size_t k) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in ViewCMatrixKokkos 3D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in ViewCMatrixKokkos 3D!");
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in ViewCMatrixKokkos 3D!");
    return this_matrix_[(k - 1) + ((j - 1) * dim3_) 
                                + ((i - 1) * dim3_ * dim2_)];
}

template <typename T>
KOKKOS_FUNCTION
T& ViewCMatrixKokkos<T>::operator()(size_t i, size_t j , size_t k, size_t l) const { 
    assert(i >= 1 && i <= dim1_ && "i is out of bounds in ViewCMatrixKokkos 4D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds in ViewCMatrixKokkos 4D!");
    assert(k >= 1 && k <= dim3_ && "k is out of bounds in ViewCMatrixKokkos 4D!");
    assert(l >= 1 && l <= dim4_ && "l is out of bounds in ViewCMatrixKokkos 4D!");
    return this_matrix_[(l - 1) + ((k - 1) * dim4_) 
                                + ((j - 1) * dim4_ * dim3_) 
                                + ((i - 1) * dim4_ * dim3_ * dim2_)];
}

template <typename T>
KOKKOS_FUNCTION
T& ViewCMatrixKokkos<T>::operator()(size_t i, size_t j, size_t k, size_t l, 
                                    size_t m) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds for ViewCMatrixKokkos 5D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds for ViewCMatrixKokkos 5D!");
    assert(k >= 1 && k <= dim3_ && "k is out of bounds for ViewCMatrixKokkos 5D!");
    assert(l >= 1 && l <= dim4_ && "l is out of bounds for ViewCMatrixKokkos 5D!");
    assert(m >= 1 && m <= dim5_ && "m is out of bounds for ViewCMatrixKokkos 5D!");
    return this_matrix_[(m - 1) + ((l - 1) * dim5_)
                                + ((k - 1) * dim5_ * dim4_)
                                + ((j - 1) * dim5_ * dim4_ * dim3_)
                                + ((i - 1) * dim5_ * dim4_ * dim3_ * dim2_)];
}

template <typename T>
KOKKOS_FUNCTION
T& ViewCMatrixKokkos<T>::operator()(size_t i, size_t j, size_t k, size_t l, 
                                    size_t m, size_t n) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds for ViewCMatrixKokkos 6D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds for ViewCMatrixKokkos 6D!");
    assert(k >= 1 && k <= dim3_ && "k is out of bounds for ViewCMatrixKokkos 6D!");
    assert(l >= 1 && l <= dim4_ && "l is out of bounds for ViewCMatrixKokkos 6D!");
    assert(m >= 1 && m <= dim5_ && "m is out of bounds for ViewCMatrixKokkos 6D!");
    assert(n >= 1 && n <= dim6_ && "n is out of bounds for ViewCMatrixKokkos 6D!");
    return this_matrix_[(n - 1) + ((m - 1) * dim6_)
                                + ((l - 1) * dim6_ * dim5_)
                                + ((k - 1) * dim6_ * dim5_ * dim4_)
                                + ((j - 1) * dim6_ * dim5_ * dim4_ * dim3_)
                                + ((i - 1) * dim6_ * dim5_ * dim4_ * dim3_ * dim2_)];
}

template <typename T>
KOKKOS_FUNCTION
T& ViewCMatrixKokkos<T>::operator()(size_t i, size_t j, size_t k, size_t l,
                                    size_t m, size_t n, size_t o) const {
    assert(i >= 1 && i <= dim1_ && "i is out of bounds for ViewCMatrixKokkos 7D!");
    assert(j >= 1 && j <= dim2_ && "j is out of bounds for ViewCMatrixKokkos 7D!");
    assert(k >= 1 && k <= dim3_ && "k is out of bounds for ViewCMatrixKokkos 7D!");
    assert(l >= 1 && l <= dim4_ && "l is out of bounds for ViewCMatrixKokkos 7D!");
    assert(m >= 1 && m <= dim5_ && "m is out of bounds for ViewCMatrixKokkos 7D!");
    assert(n >= 1 && n <= dim6_ && "n is out of bounds for ViewCMatrixKokkos 7D!");
    assert(o >= 1 && o <= dim7_ && "o is out of bounds for ViewCMatrixKokkos 7D!");
    return this_matrix_[o + ((n - 1) * dim7_)
                          + ((m - 1) * dim7_ * dim6_)
                          + ((l - 1) * dim7_ * dim6_ * dim5_)
                          + ((k - 1) * dim7_ * dim6_ * dim5_ * dim4_)
                          + ((j - 1) * dim7_ * dim6_ * dim5_ * dim4_ * dim3_)
                          + ((i - 1) * dim7_ * dim6_ * dim5_ * dim4_ * dim3_ * dim2_)];
}


template <typename T>
KOKKOS_FUNCTION
size_t ViewCMatrixKokkos<T>::size() {
    return length_;
}

template <typename T>
size_t ViewCMatrixKokkos<T>::extent() {
    return length_;
}

template <typename T>
KOKKOS_FUNCTION
ViewCMatrixKokkos<T>::~ViewCMatrixKokkos() {}

////////////////////////////////////////////////////////////////////////////////
// End of ViewCMatrixKokkos
////////////////////////////////////////////////////////////////////////////////

/*! \brief Kokkos version of the serial RaggedRightArray class.
 *
 */
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class RaggedRightArrayKokkos {

    using TArray1D = Kokkos::View<T*, Layout, ExecSpace, MemoryTraits>;
    
private:
    SArray1D start_index_;
    TArray1D array_; 
    
    size_t dim1_;
    size_t length_;
    size_t num_saved_;

    // THIS WILL BE A GPU POINTER!
    size_t* mystrides_;
    
public:
    // Default constructor
    RaggedRightArrayKokkos();
    
    //--- 2D array access of a ragged right array ---
    
    // Overload constructor for a CArray
    RaggedRightArrayKokkos(CArrayKokkos<size_t> &strides_array);
    
    // Overload constructor for a ViewCArray
    RaggedRightArrayKokkos(ViewCArray<size_t> &strides_array);
    
    // Overloaded constructor for a traditional array
    RaggedRightArrayKokkos(size_t* strides_array, size_t some_dim1);

    // Overload constructor for a RaggedRightArray to
    // support a dynamically built stride_array
    RaggedRightArrayKokkos (size_t some_dim1, size_t buffer);
    
    // A method to return the stride size
    KOKKOS_FUNCTION
    size_t stride(size_t i) const;
    
    // A method to increase the number of column entries, i.e.,
    // the stride size. Used with the constructor for building
    // the stride_array dynamically.
    // DO NOT USE with the constructures with a strides_array
    KOKKOS_FUNCTION
    size_t& build_stride(const size_t i) const;

    void stride_finalize() const;
    
    // Overload operator() to access data as array(i,j)
    // where i=[0:N-1], j=[stride(i)]
    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j) const;

    // method to return total size
    size_t size();

    T* pointer();

    //return the view
    TArray1D get_kokkos_view();

    RaggedRightArrayKokkos& operator= (const RaggedRightArrayKokkos &temp);

    //functors for kokkos execution policies
    //sets final 1D array size
    class finalize_stride_functor{
        finalize_stride_functor(){}
        void operator()(const int index, int& update, bool final) const {
          // Load old value in case we update it before accumulating
            const size_t count = start_index_(index+1);
            update += count;
            if (final) {
                start_index_((index+1)) = update;
            }   
        }
    };
    //initializes start(0); not sure if this is useful but copying from the LAMBDA implementation.
    class assignment_init_functor{
        assignment_init_functor(){}
        void operator()(const int index) const {
          start_index_(0) = 0;
        }
    };
    
    //used in the assignment operator overload
    class assignment_scan_functor{
        RaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>* mytemp;
        assignment_scan_functor(const RaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits> &temp){
          mytemp = &temp;
        }
        void operator()(const int index, int& update, bool final) const {
          // Load old value in case we update it before accumulating
            const size_t count = mytemp->mystrides_[index];
            update += count;
            if (final) {
                start_index_((index+1)) = update;
            }   
        }
    };

    class templen_functor{
        SArray1D* mytemplen;
        templen_functor(SArray1D &templen){
            mytemplen = &templen;
        }
        void operator()(const int index) const {
          (*mytemplen)(0) = start_index_(dim1_);
        }
    };

    // Destructor
    KOKKOS_FUNCTION
    ~RaggedRightArrayKokkos ( );
}; // End of RaggedRightArray

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
RaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::RaggedRightArrayKokkos() {}

// Overloaded constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
RaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::RaggedRightArrayKokkos(CArrayKokkos<size_t> &strides_array) {
    mystrides_ = strides_array.pointer();
    dim1_ = strides_array.extent();
} // End constructor

// Overloaded constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
RaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::RaggedRightArrayKokkos(ViewCArray<size_t> &strides_array) {
} // End constructor

// Overloaded constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
RaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::RaggedRightArrayKokkos(size_t* strides_array, 
                                                  size_t some_dim1) {
    mystrides_ = strides_array;
    dim1_ = some_dim1;
} // End constructor

// overloaded constructor for a dynamically built strides_array.
// buffer is the max number of columns needed
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
RaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::RaggedRightArrayKokkos (size_t some_dim1, size_t buffer) 
{
    dim1_ = some_dim1;

    // create and initialize the starting index of the entries in the 1D array
    //start_index_ = new size_t[dim1_+1]();  // note the dim1+1
    //start_index_[0] = 0; // the 1D array starts at 0

    num_saved_ = 0;
    
    length_ = some_dim1*buffer;
    SArray1D tempstrides = SArray1D("tempstrides", dim1_ + 1);
    mystrides_ = tempstrides.data();
    
} // end constructor

// A method to return the stride size
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
size_t RaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::stride(size_t i) const {
    // Ensure that i is within bounds
    assert(i < (dim1_ + 1) && "i is greater than dim1_ in RaggedRightArray");
    return start_index_((i + 1)) - start_index_(i);
}

// Method to build the stride (non-Kokkos push back)
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
size_t& RaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::build_stride(const size_t i) const {
    return start_index_(i+1);
}

// Method to finalize stride
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void RaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::stride_finalize() const {
    
    #ifdef HAVE_CLASS_LAMBDA
    Kokkos::parallel_scan("StartValues", dim1_, KOKKOS_CLASS_LAMBDA(const int i, int& update, const bool final) {
            // Load old value in case we update it before accumulating
            const size_t count = start_index_(i+1);
            update += count;
            if (final) {
                start_index_((i+1)) = update;
            }       

        });
    #else
    finalize_stride_functor execution_functor;
    Kokkos::parallel_scan("StartValues", dim1_,execution_functor);
    #endif
    Kokkos::fence();
}


// Overload operator() to access data as array(i,j)
// where i=[0:N-1], j=[0:stride(i)]
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& RaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j) const {
    // Get the 1D array index
    size_t start = start_index_(i);
    
    // asserts
    assert(i < dim1_ && "i is out of dim1 bounds in RaggedRightArrayKokkos");  // die if >= dim1
    assert(j < stride(i) && "j is out of stride bounds in RaggedRightArrayKokkos");  // die if >= stride
    
    return array_(j + start);
} // End operator()

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T* RaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::pointer() {
    return array_.data();
}


template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
RaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits> & RaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator= (const RaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits> &temp) {

  if (this != &temp) {
      /*
    SArray1D tempdim = SArray1D("tempdim", 1);
    auto h_tempdim = HostMirror(tempdim);
    Kokkos::parallel_for("StrideDim", 1, KOKKOS_CLASS_LAMBDA(const int&) {
            tempdim(0)  = strides_array.size();
            //dim1_  = strides_array.size();
        });
    Kokkos::fence();
    deep_copy(h_tempdim, tempdim);
    dim1_ = h_tempdim(0);
    */
    dim1_ = temp.dim1_;
    num_saved_ = temp.num_saved_;
    
    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = SArray1D("start_index_", dim1_ + 1);
    //start_index_(0) = 0; // the 1D array starts at 0
    #ifdef HAVE_CLASS_LAMBDA
    Kokkos::parallel_for("StartFirst", 1, KOKKOS_CLASS_LAMBDA(const int&) {
            start_index_(0) = 0;
        });
    #else
    assignment_init_functor init_execution_functor;
    Kokkos::parallel_for("StartFirst", 1, init_execution_functor);
    #endif
    Kokkos::fence();
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array and set the starting 1D index
    #ifdef HAVE_CLASS_LAMBDA
    Kokkos::parallel_scan("StartValues", dim1_, KOKKOS_CLASS_LAMBDA(const int i, double& update, const bool final) {
            // Load old value in case we update it before accumulating
            const size_t count = temp.mystrides_[i];
            update += count;
            if (final) {
                start_index_((i+1)) = update;
            }       

        });
    #else
    assignment_scan_functor scan_execution_functor(temp);
    Kokkos::parallel_scan("StartValues", dim1_, scan_execution_functor);
    #endif
    Kokkos::fence();

    /*
    size_t * h_start_index = new size_t [dim1_+1];
    h_start_index[0] = 0;
    size_t * herenow = new size_t [2];
    herenow[0] = 1;
    herenow[1] = 2;
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += herenow[i];
        h_start_index[(i + 1)] = count;
        printf("%d) Start check %ld\n", i, h_start_index[i]);
    } // end for i
    */

    SArray1D templen = SArray1D("templen", 1);
    auto h_templen = Kokkos::create_mirror_view(templen);
    #ifdef HAVE_CLASS_LAMBDA
    Kokkos::parallel_for("ArrayLength", 1, KOKKOS_CLASS_LAMBDA(const int&) {
            templen(0) = start_index_(dim1_);
            //length_ = start_index_(dim1_);
        });
    #else
    templen_functor templen_execution_functor(templen);
    Kokkos::parallel_for("ArrayLength", 1, templen_execution_functor);
    #endif
    Kokkos::fence();
    Kokkos::deep_copy(h_templen, templen);
    if (h_templen(0) != 0)
        length_ = h_templen(0);
    else
        length_ = temp.length_;


    //printf("Length %ld\n", length_);

    //Kokkos::parallel_for("StartCheck", dim1_+1, KOKKOS_CLASS_LAMBDA(const int i) {
    //        printf("%d) Start %ld\n", i, start_index_(i));
    //    });
    //Kokkos::fence();
    
    array_ = TArray1D("array_", length_);

    /*
        dim1_ = temp.dim1_;
        length_ = temp.length_;
        start_index_ = SArray1D("start_index_", dim1_ + 1);
        Kokkos::parallel_for("EqualOperator", dim1_+1, KOKKOS_CLASS_LAMBDA(const int j) {
                start_index_(j) = temp.start_index_(j);  
            });
        //for (int j = 0; j < dim1_; j++) {
        //    start_index_(j) = temp.start_index_(j);  
        //}
        array_ = TArray1D("array_", length_);
    */
  }
    
    return *this;
}

//return the stored Kokkos view
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
Kokkos::View<T*, Layout, ExecSpace, MemoryTraits> RaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_view() {
    return array_;
}

// Destructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
RaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::~RaggedRightArrayKokkos() { }

////////////////////////////////////////////////////////////////////////////////
// End of RaggedRightArrayKokkos
////////////////////////////////////////////////////////////////////////////////

/*! \brief Kokkos version of the serial RaggedDownArray class.
 *
 */
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class RaggedDownArrayKokkos {

    using TArray1D = Kokkos::View<T*, Layout, ExecSpace, MemoryTraits>;
    
private:
    SArray1D start_index_;
    TArray1D array_; 
    
    size_t dim2_;
    size_t length_;

    // THIS WILL BE A GPU POINTER!
    size_t* mystrides_;
    
public:
    // Default constructor
    RaggedDownArrayKokkos();
    
    //--- 2D array access of a ragged right array ---
    
    // Overload constructor for a CArray
    RaggedDownArrayKokkos(CArrayKokkos<size_t> &strides_array);
    
    // Overload constructor for a ViewCArray
    RaggedDownArrayKokkos(ViewCArray<size_t> &strides_array);
    
    // Overloaded constructor for a traditional array
    RaggedDownArrayKokkos(size_t* strides_array, size_t some_dim2);

    // A method to return the stride size
    KOKKOS_FUNCTION
    size_t stride(size_t j) const;
    
    // Overload operator() to access data as array(i,j)
    // where i=[0:N-1], j=[stride(i)]
    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j) const;

    T* pointer();

    //return the view
    TArray1D get_kokkos_view();

    RaggedDownArrayKokkos& operator= (const RaggedDownArrayKokkos &temp);
    
    //kokkos policy functors
    //initializes start(0); not sure if this is useful but copying from the LAMBDA implementation.
    class assignment_init_functor{
        assignment_init_functor(){}
        void operator()(const int index) const {
          start_index_(0) = 0;
        }
    };
    
    //used in the assignment operator overload
    class assignment_scan_functor{
        RaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>* mytemp;
        assignment_scan_functor(const RaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits> &temp){
          mytemp = &temp;
        }
        void operator()(const int index, int& update, bool final) const {
          // Load old value in case we update it before accumulating
            const size_t count = mytemp->mystrides_[index];
            update += count;
            if (final) {
                start_index_((index+1)) = update;
            }   
        }
    };

    class templen_functor{
        SArray1D* mytemplen;
        templen_functor(SArray1D &templen){
            mytemplen = &templen;
        }
        void operator()(const int index) const {
          (*mytemplen)(0) = start_index_(dim2_);
        }
    };
    
    class stride_check_functor{
        stride_check_functor(){}
        void operator()(const int index) const {
          printf("%d) Start %ld\n", index, start_index_(index));
        }
    };
    

    // Destructor
    KOKKOS_FUNCTION
    ~RaggedDownArrayKokkos ( );
}; // End of RaggedDownArray

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
RaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::RaggedDownArrayKokkos() {}

// Overloaded constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
RaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::RaggedDownArrayKokkos(CArrayKokkos<size_t> &strides_array) {
    mystrides_ = strides_array.pointer();
    dim2_ = strides_array.extent();
} // End constructor

// Overloaded constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
RaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::RaggedDownArrayKokkos(ViewCArray<size_t> &strides_array) {
} // End constructor

// Overloaded constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
RaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::RaggedDownArrayKokkos(size_t* strides_array, 
                                                  size_t some_dim2) {
    mystrides_ = strides_array;
    dim2_ = some_dim2;
} // End constructor

// A method to return the stride size
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
size_t RaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::stride(size_t j) const {
    // Ensure that j is within bounds
    assert(j < (dim2_ + 1) && "j is greater than dim1_ in RaggedDownArray");

    return start_index_((j + 1)) - start_index_(j);
}

// Overload operator() to access data as array(i,j)
// where i=[0:N-1], j=[0:stride(i)]
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& RaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j) const {
    // Get the 1D array index
    size_t start = start_index_(j);
    
    // asserts
    assert(i < stride(j) && "i is out of stride bounds in RaggedDownArrayKokkos");  // die if >= stride
    assert(j < dim2_ && "j is out of dim1 bounds in RaggedDownArrayKokkos");  // die if >= dim1
    
    return array_(i + start);
} // End operator()

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
RaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits> & RaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator= (const RaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits> &temp) {

  if (this != &temp) {
      /*
    SArray1D tempdim = SArray1D("tempdim", 1);
    auto h_tempdim = HostMirror(tempdim);
    Kokkos::parallel_for("StrideDim", 1, KOKKOS_CLASS_LAMBDA(const int&) {
            tempdim(0)  = strides_array.size();
            //dim1_  = strides_array.size();
        });
    Kokkos::fence();
    deep_copy(h_tempdim, tempdim);
    dim1_ = h_tempdim(0);
    */
    dim2_ = temp.dim2_;
    
    // Create and initialize the starting index of the entries in the 1D array
    start_index_ = SArray1D("start_index_", dim2_ + 1);
    //start_index_(0) = 0; // the 1D array starts at 0
    #ifdef HAVE_CLASS_LAMBDA
    Kokkos::parallel_for("StartFirst", 1, KOKKOS_CLASS_LAMBDA(const int&) {
            start_index_(0) = 0;
        });
    #else
    assignment_init_functor init_execution_functor;
    Kokkos::parallel_for("StartFirst", 1, init_execution_functor);
    #endif
    Kokkos::fence();
    
    // Loop over to find the total length of the 1D array to
    // represent the ragged-right array and set the starting 1D index
    #ifdef HAVE_CLASS_LAMBDA
    Kokkos::parallel_scan("StartValues", dim2_, KOKKOS_CLASS_LAMBDA(const int j, double& update, const bool final) {
            // Load old value in case we update it before accumulating
            const size_t count = temp.mystrides_[j];
            update += count;
            if (final) {
                start_index_((j+1)) = update;
            }       

        });
    #else
    assignment_scan_functor scan_execution_functor(temp);
    Kokkos::parallel_scan("StartValues", dim2_, scan_execution_functor);
    #endif
    Kokkos::fence();

    /*
    size_t * h_start_index = new size_t [dim1_+1];
    h_start_index[0] = 0;
    size_t * herenow = new size_t [2];
    herenow[0] = 1;
    herenow[1] = 2;
    size_t count = 0;
    for (size_t i = 0; i < dim1_; i++){
        count += herenow[i];
        h_start_index[(i + 1)] = count;
        printf("%d) Start check %ld\n", i, h_start_index[i]);
    } // end for i
    */

    SArray1D templen = SArray1D("templen", 1);
    auto h_templen = Kokkos::create_mirror_view(templen);
    #ifdef HAVE_CLASS_LAMBDA
    Kokkos::parallel_for("ArrayLength", 1, KOKKOS_CLASS_LAMBDA(const int&) {
            templen(0) = start_index_(dim2_);
            //length_ = start_index_(dim2_);
        });
    #else
    templen_functor templen_execution_functor(templen);
    Kokkos::parallel_for("ArrayLength", 1, templen_execution_functor);
    #endif
    Kokkos::fence();
    deep_copy(h_templen, templen);
    length_ = h_templen(0);

    printf("Length %ld\n", length_);
    
    #ifdef HAVE_CLASS_LAMBDA
    Kokkos::parallel_for("StartCheck", dim2_+1, KOKKOS_CLASS_LAMBDA(const int j) {
            printf("%d) Start %ld\n", j, start_index_(j));
        });
    #else
    stride_check_functor check_execution_functor;
    Kokkos::parallel_for("StartCheck", dim2_+1, check_execution_functor);
    #endif
    Kokkos::fence();
    
    array_ = TArray1D("array_", length_);

    /*
        dim1_ = temp.dim1_;
        length_ = temp.length_;
        start_index_ = SArray1D("start_index_", dim1_ + 1);
        Kokkos::parallel_for("EqualOperator", dim1_+1, KOKKOS_CLASS_LAMBDA(const int j) {
                start_index_(j) = temp.start_index_(j);  
            });
        //for (int j = 0; j < dim1_; j++) {
        //    start_index_(j) = temp.start_index_(j);  
        //}
        array_ = TArray1D("array_", length_);
    */
  }
    
    return *this;
}

//return the stored Kokkos view
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
Kokkos::View<T*, Layout, ExecSpace, MemoryTraits> RaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_view() {
    return array_;
}

// Destructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
RaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::~RaggedDownArrayKokkos() { }

////////////////////////////////////////////////////////////////////////////////
// End of RaggedDownArrayKokkos
////////////////////////////////////////////////////////////////////////////////

//11. DynamicRaggedRightArray
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class DynamicRaggedRightArrayKokkos {

    using TArray1D = Kokkos::View<T*, Layout, ExecSpace, MemoryTraits>;
    
private:
    // THIS WILL BE A GPU POINTER!
    SArray1D stride_;
    TArray1D array_; 
    
    size_t dim1_;
    size_t dim2_;
    size_t length_;
    
public:
    // Default constructor
    DynamicRaggedRightArrayKokkos ();
    
    //--- 2D array access of a ragged right array ---
    
    // overload constructor
    DynamicRaggedRightArrayKokkos (size_t dim1, size_t dim2);
    
    // A method to return or set the stride size
    KOKKOS_FUNCTION
    size_t& stride(size_t i) const;
    
    // A method to return the size
    KOKKOS_FUNCTION
    size_t size() const;

    //return the view
    TArray1D get_kokkos_view();
    
    // Overload operator() to access data as array(i,j),
    // where i=[0:N-1], j=[stride(i)]
    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j) const;
    
    // Overload copy assignment operator
    DynamicRaggedRightArrayKokkos& operator= (const DynamicRaggedRightArrayKokkos &temp);
    
    //kokkos policy functors
    //set strides to 0 functor
    class stride_zero_functor{
        stride_zero_functor(){}
        void operator()(const int index) const {
          stride_(index) = 0;
        }
    };
    
    // Destructor
    ~DynamicRaggedRightArrayKokkos ();
};

//nothing
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
DynamicRaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::DynamicRaggedRightArrayKokkos () {}

// Overloaded constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
DynamicRaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::DynamicRaggedRightArrayKokkos (size_t dim1, size_t dim2) {
    // The dimensions of the array;
    dim1_  = dim1;
    dim2_  = dim2;
    length_ = dim1*dim2;
}

// A method to set the stride size for row i
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
size_t& DynamicRaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::stride(size_t i) const {
    return stride_(i);
}

//return size
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
size_t DynamicRaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::size() const{
    return length_;
}

// Overload operator() to access data as array(i,j),
// where i=[0:N-1], j=[0:stride(i)]
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
inline T& DynamicRaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j) const {
    // Asserts
    assert(i < dim1_ && "i is out of dim1 bounds in DynamicRaggedRight");  // die if >= dim1
    assert(j < dim2_ && "j is out of dim2 bounds in DynamicRaggedRight");  // die if >= dim2
    // Cannot assert on Kokkos View
    //assert(j < stride_[i] && "j is out of stride bounds in DynamicRaggedRight");  // die if >= stride
    
    return array_(j + i*dim2_);
}

//overload = operator
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
inline DynamicRaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>&
       DynamicRaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator= (const DynamicRaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits> &temp)
{
    
    if( this != &temp) {
        dim1_ = temp.dim1_;
        dim2_ = temp.dim2_;
        length_ = temp.length_;
        stride_ = SArray1D("stride_", dim1_);
        array_ = TArray1D("array_", length_);
        #ifdef HAVE_CLASS_LAMBDA 
        Kokkos::parallel_for("StrideZeroOut", dim1_, KOKKOS_CLASS_LAMBDA(const int i) {
            stride_(i) = 0;
        });
        #else
        stride_zero_functor execution_functor;
        Kokkos::parallel_for("StrideZeroOut", dim1_, execution_functor);
        #endif
    }
    
    return *this;
}

//return the stored Kokkos view
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
Kokkos::View<T*, Layout, ExecSpace, MemoryTraits> DynamicRaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_view() {
    return array_;
}

// Destructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
DynamicRaggedRightArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::~DynamicRaggedRightArrayKokkos() {
}




//----end DynamicRaggedRightArray class definitions----


//12. DynamicRaggedDownArray

template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class DynamicRaggedDownArrayKokkos {

    using TArray1D = Kokkos::View<T*, Layout, ExecSpace, MemoryTraits>;

private:
    SArray1D stride_;
    TArray1D array_; 
    
    size_t dim1_;
    size_t dim2_;
    size_t length_;
    
public:
    // Default constructor
    DynamicRaggedDownArrayKokkos ();
    
    //--- 2D array access of a ragged right array ---
    
    // overload constructor
    DynamicRaggedDownArrayKokkos (size_t dim1, size_t dim2);
    
    // A method to return or set the stride size
    KOKKOS_FUNCTION
    size_t& stride(size_t j) const;
    
    // A method to return the size
    KOKKOS_FUNCTION
    size_t size() const;

    //return the view
    TArray1D get_kokkos_view();
    
    // Overload operator() to access data as array(i,j),
    // where i=[stride(j)], j=[0:N-1]
    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j) const;
    
    // Overload copy assignment operator
    DynamicRaggedDownArrayKokkos& operator= (const DynamicRaggedDownArrayKokkos &temp);

    //kokkos policy functors
    //set strides to 0 functor
    class stride_zero_functor{
        stride_zero_functor(){}
        void operator()(const int index) const {
          stride_(index) = 0;
        }
    };
    
    // Destructor
    ~DynamicRaggedDownArrayKokkos ();
};

//nothing
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
DynamicRaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::DynamicRaggedDownArrayKokkos () {}

// Overloaded constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
DynamicRaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::DynamicRaggedDownArrayKokkos (size_t dim1, size_t dim2) {
    // The dimensions of the array;
    dim1_  = dim1;
    dim2_  = dim2;
    length_ = dim1*dim2;
}

// A method to set the stride size for column j
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
size_t& DynamicRaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::stride(size_t j) const {
    return stride_(j);
}

//return size
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
size_t DynamicRaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::size() const{
    return length_;
}

// overload operator () to access data as an array(i,j)
// Note: i = 0:stride(j), j = 0:N-1

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_FUNCTION
T& DynamicRaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j) const {
    // Asserts
    assert(i < dim1_ && "i is out of dim1 bounds in DynamicRaggedDownArrayKokkos");  // die if >= dim1
    assert(j < dim2_ && "j is out of dim2 bounds in DynamicRaggedDownArrayKokkos");  // die if >= dim2
    // Can't do this assert with a Kokkos View
    //assert(i < stride_[j] && "i is out of stride bounds in DynamicRaggedDownArrayKokkos");  // die if >= stride
    
    return array_(i + j*dim1_);
}

//overload = operator
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
DynamicRaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits>&
  DynamicRaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator= (const DynamicRaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits> &temp)
{
    
    if( this != &temp) {
        dim1_ = temp.dim1_;
        dim2_ = temp.dim2_;
        length_ = temp.length_;
        stride_ = SArray1D("stride_", dim2_);
        array_ = TArray1D("array_", length_);
        #ifdef HAVE_CLASS_LAMBDA
        Kokkos::parallel_for("StrideZeroOut", dim2_, KOKKOS_CLASS_LAMBDA(const int j) {
            stride_(j) = 0;
        });
        #else
        stride_zero_functor execution_functor;
        Kokkos::parallel_for("StrideZeroOut", dim2_, execution_functor);
        #endif
    }
    
    return *this;
}

//return the stored Kokkos view
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
Kokkos::View<T*, Layout, ExecSpace, MemoryTraits> DynamicRaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_view() {
    return array_;
}

// Destructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
DynamicRaggedDownArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::~DynamicRaggedDownArrayKokkos() {
}


/////////////////////////
//// CArrayKokkosPtr ////
/////////////////////////
template <typename T>
class CArrayKokkosPtr {

    // this is always unmanaged
    using TArray1DHost = Kokkos::View<T*, Layout, HostSpace, MemoryUnmanaged>;
    // this is manage
    using TArray1D     = Kokkos::View<T*, Layout, ExecSpace>;
    
private:
    size_t dim1_;
    size_t dim2_;
    size_t dim3_;
    size_t dim4_;
    size_t dim5_;
    size_t dim6_;
    size_t dim7_;
    size_t length_;
    TArray1D this_array_; 
    TArray1DHost this_array_host_; 
    T * temp_inp_array_;
    //typename Kokkos::View<T*, Layout, ExecSpace>::HostMirror  h_this_array_;

public:
    CArrayKokkosPtr();
    
    CArrayKokkosPtr(T * inp_array, size_t some_dim1);

    CArrayKokkosPtr(T * inp_array, size_t some_dim1, size_t some_dim2);

    CArrayKokkosPtr(T * inp_array, size_t some_dim1, size_t some_dim2, size_t some_dim3);

    CArrayKokkosPtr(T * inp_array, size_t some_dim1, size_t some_dim2, size_t some_dim3, 
                 size_t some_dim4);

    CArrayKokkosPtr(T * inp_array, size_t some_dim1, size_t some_dim2, size_t some_dim3,
                 size_t some_dim4, size_t some_dim5);

    CArrayKokkosPtr(T * inp_array, size_t some_dim1, size_t some_dim2, size_t some_dim3,
                 size_t some_dim4, size_t some_dim5, size_t some_dim6);

    CArrayKokkosPtr(T * inp_array, size_t some_dim1, size_t some_dim2, size_t some_dim3,
                 size_t some_dim4, size_t some_dim5, size_t some_dim6,
                 size_t some_dim7);
    
    KOKKOS_FUNCTION
    T& operator()(size_t i) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m, 
                  size_t n) const;

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m,
                  size_t n, size_t o) const;
    
    CArrayKokkosPtr& operator=(const CArrayKokkosPtr& temp);

    // GPU Method
    // Method that returns size
    KOKKOS_FUNCTION
    size_t size();

    // Host Method
    // Method that returns size
    size_t extent();

    // Methods returns the raw pointer (most likely GPU) of the Kokkos View
    T* pointer();

    // Deconstructor
    KOKKOS_FUNCTION
    ~CArrayKokkosPtr ();
}; // End of CArrayKokkosPtr


// Default constructor
template <typename T>
CArrayKokkosPtr<T>::CArrayKokkosPtr() {}

// Overloaded 1D constructor
template <typename T>
CArrayKokkosPtr<T>::CArrayKokkosPtr(T * inp_array, size_t some_dim1) {
    using TArray1DHost = Kokkos::View<T*, Layout, HostSpace, MemoryUnmanaged>;
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    length_ = dim1_;
    // Create a 1D host view of the external allocation
    this_array_host_ = TArray1DHost(inp_array, length_);
    // Assign temp point to inp_array pointer that is passed in
    temp_inp_array_ = inp_array;
    // Create a device copy of that host view
    this_array_ = create_mirror_view_and_copy(ExecSpace(), this_array_host_);
}

// Overloaded 2D constructor
template <typename T>
CArrayKokkosPtr<T>::CArrayKokkosPtr(T * inp_array, size_t some_dim1, size_t some_dim2) {
    using TArray1DHost = Kokkos::View<T*, Layout, HostSpace, MemoryUnmanaged>;
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    //using TArray1Dtemp = TArray1D::HostMirror;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    length_ = (dim1_ * dim2_);
    // Create a 1D host view of the external allocation
    this_array_host_ = TArray1DHost(inp_array, length_);
    // Assign temp point to inp_array pointer that is passed in
    temp_inp_array_ = inp_array;
    // Create a device copy of that host view
    this_array_ = create_mirror_view_and_copy(ExecSpace(), this_array_host_);
}

template <typename T>
CArrayKokkosPtr<T>::CArrayKokkosPtr(T * inp_array, size_t some_dim1, size_t some_dim2, 
                              size_t some_dim3) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    length_ = (dim1_ * dim2_ * dim3_);
    // Create a 1D host view of the external allocation
    this_array_host_ = TArray1DHost(inp_array, length_);
    // Assign temp point to inp_array pointer that is passed in
    temp_inp_array_ = inp_array;
    // Create a device copy of that host view
    this_array_ = create_mirror_view_and_copy(ExecSpace(), this_array_host_);
}

template <typename T>
CArrayKokkosPtr<T>::CArrayKokkosPtr(T * inp_array, size_t some_dim1, size_t some_dim2, 
                              size_t some_dim3, size_t some_dim4) {
    using TArray1D = Kokkos::View<T *,Layout,ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_);
    // Create a 1D host view of the external allocation
    this_array_host_ = TArray1DHost(inp_array, length_);
    // Assign temp point to inp_array pointer that is passed in
    temp_inp_array_ = inp_array;
    // Create a device copy of that host view
    this_array_ = create_mirror_view_and_copy(ExecSpace(), this_array_host_);
}

template <typename T>
CArrayKokkosPtr<T>::CArrayKokkosPtr(T * inp_array, size_t some_dim1, size_t some_dim2, 
                              size_t some_dim3, size_t some_dim4, 
                              size_t some_dim5) {

    using TArray1D = Kokkos::View<T *,Layout,ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_);
    // Create a 1D host view of the external allocation
    this_array_host_ = TArray1DHost(inp_array, length_);
    // Assign temp point to inp_array pointer that is passed in
    temp_inp_array_ = inp_array;
    // Create a device copy of that host view
    this_array_ = create_mirror_view_and_copy(ExecSpace(), this_array_host_);
}

template <typename T>
CArrayKokkosPtr<T>::CArrayKokkosPtr(T * inp_array, size_t some_dim1, size_t some_dim2, 
                              size_t some_dim3, size_t some_dim4, 
                              size_t some_dim5, size_t some_dim6) {
    using TArray1D = Kokkos::View<T *,Layout,ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_);
    // Create a 1D host view of the external allocation
    this_array_host_ = TArray1DHost(inp_array, length_);
    // Assign temp point to inp_array pointer that is passed in
    temp_inp_array_ = inp_array;
    // Create a device copy of that host view
    this_array_ = create_mirror_view_and_copy(ExecSpace(), this_array_host_);
}

template <typename T>
CArrayKokkosPtr<T>::CArrayKokkosPtr(T * inp_array, size_t some_dim1, size_t some_dim2,
                              size_t some_dim3, size_t some_dim4,
                              size_t some_dim5, size_t some_dim6,
                              size_t some_dim7) {
    using TArray1D = Kokkos::View<T *,Layout,ExecSpace>;
    
    dim1_ = some_dim1;
    dim2_ = some_dim2;
    dim3_ = some_dim3;
    dim4_ = some_dim4;
    dim5_ = some_dim5;
    dim6_ = some_dim6;
    dim7_ = some_dim7;
    length_ = (dim1_ * dim2_ * dim3_ * dim4_ * dim5_ * dim6_ * dim7_);
    // Create a 1D host view of the external allocation
    this_array_host_ = TArray1DHost(inp_array, length_);
    // Assign temp point to inp_array pointer that is passed in
    temp_inp_array_ = inp_array;
    // Create a device copy of that host view
    this_array_ = create_mirror_view_and_copy(ExecSpace(), this_array_host_);
}

template <typename T>
KOKKOS_FUNCTION
T& CArrayKokkosPtr<T>::operator()(size_t i) const {
    assert(i < dim1_ && "i is out of bounds in CArrayKokkosPtr 1D!");
    return this_array_(i);
}

template <typename T>
KOKKOS_FUNCTION
T& CArrayKokkosPtr<T>::operator()(size_t i, size_t j) const {
    assert(i < dim1_ && "i is out of bounds in CArrayKokkosPtr 2D!");
    assert(j < dim2_ && "j is out of bounds in CArrayKokkosPtr 2D!");
    return this_array_(j + (i * dim2_));
}

template <typename T>
KOKKOS_FUNCTION
T& CArrayKokkosPtr<T>::operator()(size_t i, size_t j, size_t k) const {
    assert(i < dim1_ && "i is out of bounds in CArrayKokkosPtr 3D!");
    assert(j < dim2_ && "j is out of bounds in CArrayKokkosPtr 3D!");
    assert(k < dim3_ && "k is out of bounds in CArrayKokkosPtr 3D!");
    return this_array_(k + (j * dim3_) 
                         + (i * dim3_ * dim2_));
}

template <typename T>
KOKKOS_FUNCTION
T& CArrayKokkosPtr<T>::operator()(size_t i, size_t j, size_t k, size_t l) const {
    assert(i < dim1_ && "i is out of bounds in CArrayKokkosPtr 4D!");
    assert(j < dim2_ && "j is out of bounds in CArrayKokkosPtr 4D!");
    assert(k < dim3_ && "k is out of bounds in CArrayKokkosPtr 4D!");
    assert(l < dim4_ && "l is out of bounds in CArrayKokkosPtr 4D!");
    return this_array_(l + (k * dim4_) 
                         + (j * dim4_ * dim3_)  
                         + (i * dim4_ * dim3_ * dim2_));
}

template <typename T>
KOKKOS_FUNCTION
T& CArrayKokkosPtr<T>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m) const {
    assert(i < dim1_ && "i is out of bounds in CArrayKokkosPtr 5D!");
    assert(j < dim2_ && "j is out of bounds in CArrayKokkosPtr 5D!");
    assert(k < dim3_ && "k is out of bounds in CArrayKokkosPtr 5D!");
    assert(l < dim4_ && "l is out of bounds in CArrayKokkosPtr 5D!");
    assert(m < dim5_ && "m is out of bounds in CArrayKokkosPtr 5D!");
    return this_array_(m + (l * dim5_) 
                         + (k * dim5_ * dim4_) 
                         + (j * dim5_ * dim4_ * dim3_) 
                         + (i * dim5_ * dim4_ * dim3_ * dim2_));
}

template <typename T>
KOKKOS_FUNCTION
T& CArrayKokkosPtr<T>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n) const {
    assert(i < dim1_ && "i is out of bounds in CArrayKokkosPtr 6D!");
    assert(j < dim2_ && "j is out of bounds in CArrayKokkosPtr 6D!");
    assert(k < dim3_ && "k is out of bounds in CArrayKokkosPtr 6D!");
    assert(l < dim4_ && "l is out of bounds in CArrayKokkosPtr 6D!");
    assert(m < dim5_ && "m is out of bounds in CArrayKokkosPtr 6D!");
    assert(n < dim6_ && "n is out of bounds in CArrayKokkosPtr 6D!");
    return this_array_(n + (m * dim6_) 
                         + (l * dim6_ * dim5_)  
                         + (k * dim6_ * dim5_ * dim4_) 
                         + (j * dim6_ * dim5_ * dim4_ * dim3_)  
                         + (i * dim6_ * dim5_ * dim4_ * dim3_ * dim2_));
}

template <typename T>
KOKKOS_FUNCTION
T& CArrayKokkosPtr<T>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n, size_t o) const {
    assert(i < dim1_ && "i is out of bounds in CArrayKokkosPtr 7D!");
    assert(j < dim2_ && "j is out of bounds in CArrayKokkosPtr 7D!");
    assert(k < dim3_ && "k is out of bounds in CArrayKokkosPtr 7D!");
    assert(l < dim4_ && "l is out of bounds in CArrayKokkosPtr 7D!");
    assert(m < dim5_ && "m is out of bounds in CArrayKokkosPtr 7D!");
    assert(n < dim6_ && "n is out of bounds in CArrayKokkosPtr 7D!");
    assert(o < dim7_ && "o is out of bounds in CArrayKokkosPtr 7D!");
    return this_array_(o + (n * dim7_)
                         + (m * dim7_ * dim6_)
                         + (l * dim7_ * dim6_ * dim5_)
                         + (k * dim7_ * dim6_ * dim5_ * dim4_)
                         + (j * dim7_ * dim6_ * dim5_ * dim4_ * dim3_)
                         + (i * dim7_ * dim6_ * dim5_ * dim4_ * dim3_ * dim2_));
}

template <typename T>
CArrayKokkosPtr<T>& CArrayKokkosPtr<T>::operator= (const CArrayKokkosPtr& temp) {
    using TArray1D = Kokkos::View<T *,Layout,ExecSpace>;
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        dim1_ = temp.dim1_;
        dim2_ = temp.dim2_;
        dim3_ = temp.dim3_;
        dim4_ = temp.dim4_;
        dim5_ = temp.dim5_;
        dim6_ = temp.dim6_;
        dim7_ = temp.dim7_;
        length_ = temp.length_;
        temp_inp_array_ = temp.temp_inp_array_;
        this_array_host_ = TArray1DHost(temp_inp_array_, length_);
        this_array_ = create_mirror_view_and_copy(ExecSpace(), this_array_host_);
    }
    
    return *this;
}

// Return size
template <typename T>
KOKKOS_FUNCTION
size_t CArrayKokkosPtr<T>::size() {
    return length_;
}

template <typename T>
size_t CArrayKokkosPtr<T>::extent() {
    return length_;
}

template <typename T>
T* CArrayKokkosPtr<T>::pointer() {
    return this_array_.data();
}

template <typename T>
KOKKOS_FUNCTION
CArrayKokkosPtr<T>::~CArrayKokkosPtr() {}
// End CArrayKokkosPtr


//////////////////////////
// Inherited Class Array
//////////////////////////

/*
//template<class T, class Layout, class ExecSpace>
template<typename T>
class InheritedArray2L {

    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;

private:
    size_t dim1_, length_;

public:
    TArray1D this_array_;
    typename Kokkos::View<T*, Layout, ExecSpace>::HostMirror  h_this_array_;

    InheritedArray2L();
    
    InheritedArray2L(size_t some_dim1);

    KOKKOS_FUNCTION
    T& operator()(size_t i, size_t dest) const;

    template <typename U>
    void AllocateHost(size_t size, U *obj);

    void AllocateGPU();

    template <typename U, typename V>
    void InitModels(U *obj, V input);

    template <typename U>
    void ClearModels(U obj);

    InheritedArray2L& operator=(const InheritedArray2L& temp);

    // GPU Method
    // Method that returns size
    KOKKOS_FUNCTION
    size_t size();

    // Host Method
    // Method that returns size
    size_t extent();

    // Methods returns the raw pointer (most likely GPU) of the Kokkos View
    T* pointer();

    // Deconstructor
    KOKKOS_FUNCTION
    ~InheritedArray2L ();
}; // End of InheritedArray2L

// Default constructor
template <typename T>
InheritedArray2L<T>::InheritedArray2L() {}

// Overloaded 1D constructor
template <typename T>
InheritedArray2L<T>::InheritedArray2L(size_t some_dim1) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dim1_ = some_dim1;
    length_ = dim1_;
    this_array_ = TArray1D("this_array_", length_);
    h_this_array_ = Kokkos::create_mirror_view(this_array_);
}

template <typename T>
KOKKOS_FUNCTION
T& InheritedArray2L<T>::operator()(size_t i, size_t dest) const {
    assert(i < dim1_ && "i is out of bounds in InheritedArray2L 1D!");
    assert(dest < 2 && "dest is out of bounds in InheritedArray2L 1D!");
    if (dest == 0)
        return h_this_array_(i);
    else
        return this_array_(i);
}

template <typename T>
template <typename U>
void InheritedArray2L<T>::AllocateHost(size_t size, U *obj) {
    obj = (U *) kmalloc(size);
}

template <typename T>
void InheritedArray2L<T>::AllocateGPU() {
    Kokkos::deep_copy(this_array_, h_this_array_);
}

template <typename T>
template <typename U, typename V>
void InheritedArray2L<T>::InitModels(U *obj, V input) {
    Kokkos::parallel_for(
            "CreateObjects", 1, KOKKOS_CLASS_LAMBDA(const int&) {
                new ((V *)obj) V{input};
            });
}

template <typename T>
template <typename U>
void InheritedArray2L<T>::ClearModels(U obj) {
    Kokkos::parallel_for(
            "DestroyObjects", 1, KOKKOS_LAMBDA(const int&) {
              this_array_(0).obj->~U();
              this_array_(1).obj->~U();
            });
}

template <typename T>
InheritedArray2L<T>& InheritedArray2L<T>::operator= (const InheritedArray2L& temp) {
    using TArray1D = Kokkos::View<T *,Layout,ExecSpace>;
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        dim1_ = temp.dim1_;
        length_ = temp.length_;
        this_array_ = TArray1D("this_array_", length_);
    }
    
    return *this;
}

// Return size
template <typename T>
KOKKOS_FUNCTION
size_t InheritedArray2L<T>::size() {
    return length_;
}

template <typename T>
size_t InheritedArray2L<T>::extent() {
    return length_;
}

template <typename T>
T* InheritedArray2L<T>::pointer() {
    return this_array_.data();
}

template <typename T>
KOKKOS_FUNCTION
InheritedArray2L<T>::~InheritedArray2L() {}
*/

////////////////////////////////////////////////////////////////////////////////
// End of InheritedArray2L
////////////////////////////////////////////////////////////////////////////////


#endif







#endif // MATAR_H
