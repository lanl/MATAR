#ifndef MPI_TYPES_H
#define MPI_TYPES_H
/**********************************************************************************************
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

#include "host_types.h"
#include "kokkos_types.h"
#include <mpi.h>

#ifdef HAVE_MPI
namespace mtr
{

/*! \brief MPI version of the serial CArrayKokkos class.
 *
 */
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class MPICArrayKokkos {

    using TArray1D = Kokkos::View<T*, Layout, ExecSpace, MemoryTraits>;
    
private:
    size_t dims_[7];
    size_t order_;
    size_t length_;
    MPI_Status mpi_status_;
    MPI_Datatype mpi_datatype_;
    TArray1D this_array_;

public:
    MPICArrayKokkos();
    
    MPICArrayKokkos(MPI_Datatype mpi_type, size_t dim0, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPICArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPICArrayKokkos (MPI_Datatype mpi_type, size_t dim0, size_t dim1, size_t dim2, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPICArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPICArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, size_t dim4, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPICArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, size_t dim4, size_t dim5, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPICArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, size_t dim4, size_t dim5,
                 size_t dim6, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j, size_t k) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m,
                  size_t n) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m,
                  size_t n, size_t o) const;
    
    KOKKOS_INLINE_FUNCTION
    MPICArrayKokkos& operator=(const MPICArrayKokkos& temp);

    // GPU Method
    // Method that returns size
    KOKKOS_INLINE_FUNCTION
    size_t size() const;

    // Host Method
    // Method that returns size
    size_t extent();

    KOKKOS_INLINE_FUNCTION
    size_t dims(size_t i) const;

    KOKKOS_INLINE_FUNCTION
    size_t order() const;
 
    // Methods returns the raw pointer (most likely GPU) of the Kokkos View
    KOKKOS_INLINE_FUNCTION
    T* pointer() const;
    
    //return the view
    KOKKOS_INLINE_FUNCTION
    TArray1D get_kokkos_view() const;

    // MPI send wrapper
    void mtr_send(size_t count, int dest, int tag, MPI_Comm comm);

    // MPI recieve wrapper
    void mtr_recv(size_t count, int dest, int tag, MPI_Comm comm);

    // Deconstructor
    KOKKOS_INLINE_FUNCTION
    ~MPICArrayKokkos ();
}; // End of MPICArrayKokkos

// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos() {
    length_ = order_ = 0;
    for (int i = 0; i < 7; i++) {
        dims_[i] = 0;
    }
}

// Overloaded 1D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(MPI_Datatype mpi_type, size_t dim0, const std::string& tag_string) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dims_[0] = dim0;
    order_ = 1;
    length_ = dim0;
    this_array_ = TArray1D(tag_string, length_);
    mpi_datatype_ = mpi_type;
}

// Overloaded 2D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1, const std::string& tag_string) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    order_ = 2;
    length_ = (dim0 * dim1);
    this_array_ = TArray1D(tag_string, length_);
    mpi_datatype_ = mpi_type;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1,
                              size_t dim2, const std::string& tag_string) {
    using TArray1D = Kokkos::View<T*, Layout, ExecSpace>;
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    order_ = 3;
    length_ = (dim0 * dim1 * dim2);
    this_array_ = TArray1D(tag_string, length_);
    mpi_datatype_ = mpi_type;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1,
                              size_t dim2, size_t dim3, const std::string& tag_string) {
    using TArray1D = Kokkos::View<T *,Layout,ExecSpace>;
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    order_ = 4;
    length_ = (dim0 * dim1 * dim2 * dim3);
    this_array_ = TArray1D(tag_string, length_);
    mpi_datatype_ = mpi_type;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1,
                              size_t dim2, size_t dim3,
                              size_t dim4, const std::string& tag_string) {

    using TArray1D = Kokkos::View<T *,Layout,ExecSpace>;
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    order_ = 5;
    length_ = (dim0 * dim1 * dim2 * dim3 * dim4);
    this_array_ = TArray1D(tag_string, length_);
    mpi_datatype_ = mpi_type;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1,
                              size_t dim2, size_t dim3,
                              size_t dim4, size_t dim5, const std::string& tag_string) {
    using TArray1D = Kokkos::View<T *,Layout,ExecSpace>;
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    dims_[5] = dim5;
    order_ = 6;
    length_ = (dim0 * dim1 * dim2 * dim3 * dim4 * dim5);
    this_array_ = TArray1D(tag_string, length_);
    mpi_datatype_ = mpi_type;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1,
                              size_t dim2, size_t dim3,
                              size_t dim4, size_t dim5,
                              size_t dim6, const std::string& tag_string) {
    using TArray1D = Kokkos::View<T *,Layout,ExecSpace>;
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    dims_[5] = dim5;
    dims_[6] = dim6;
    order_ = 7;
    length_ = (dim0 * dim1 * dim2 * dim3 * dim4 * dim5 * dim6);
    this_array_ = TArray1D(tag_string, length_);
    mpi_datatype_ = mpi_type;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i) const {
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in MPICArrayKokkos 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPICArrayKokkos 1D!");
    return this_array_(i);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j) const {
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in MPICArrayKokkos 2D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPICArrayKokkos 2D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPICArrayKokkos 2D!");
    return this_array_(j + (i * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k) const {
    assert(order_ == 3 && "Tensor order (rank) does not match constructor in MPICArrayKokkos 3D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPICArrayKokkos 3D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPICArrayKokkos 3D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPICArrayKokkos 3D!");
    return this_array_(k + (j * dims_[2])
                         + (i * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l) const {
    assert(order_ == 4 && "Tensor order (rank) does not match constructor in MPICArrayKokkos 4D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPICArrayKokkos 4D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPICArrayKokkos 4D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPICArrayKokkos 4D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in MPICArrayKokkos 4D!");
    return this_array_(l + (k * dims_[3])
                         + (j * dims_[3] * dims_[2])
                         + (i * dims_[3] * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m) const {
    assert(order_ == 5 && "Tensor order (rank) does not match constructor in MPICArrayKokkos 5D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPICArrayKokkos 5D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPICArrayKokkos 5D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPICArrayKokkos 5D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in MPICArrayKokkos 5D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in MPICArrayKokkos 5D!");
    return this_array_(m + (l * dims_[4])
                         + (k * dims_[4] * dims_[3])
                         + (j * dims_[4] * dims_[3] * dims_[2])
                         + (i * dims_[4] * dims_[3] * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n) const {
    assert(order_ == 6 && "Tensor order (rank) does not match constructor in MPICArrayKokkos 6D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPICArrayKokkos 6D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPICArrayKokkos 6D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPICArrayKokkos 6D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in MPICArrayKokkos 6D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in MPICArrayKokkos 6D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in MPICArrayKokkos 6D!");
    return this_array_(n + (m * dims_[5])
                         + (l * dims_[5] * dims_[4])
                         + (k * dims_[5] * dims_[4] * dims_[3])
                         + (j * dims_[5] * dims_[4] * dims_[3] * dims_[2])
                         + (i * dims_[5] * dims_[4] * dims_[3] * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n, size_t o) const {
    assert(order_ == 7 && "Tensor order (rank) does not match constructor in MPICArrayKokkos 7D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPICArrayKokkos 7D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPICArrayKokkos 7D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPICArrayKokkos 7D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in MPICArrayKokkos 7D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in MPICArrayKokkos 7D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in MPICArrayKokkos 7D!");
    assert(o >= 0 && o < dims_[6] && "o is out of bounds in MPICArrayKokkos 7D!");
    return this_array_(o + (n * dims_[6])
                         + (m * dims_[6] * dims_[5])
                         + (l * dims_[6] * dims_[5] * dims_[4])
                         + (k * dims_[6] * dims_[5] * dims_[4] * dims_[3])
                         + (j * dims_[6] * dims_[5] * dims_[4] * dims_[3] * dims_[2])
                         + (i * dims_[6] * dims_[5] * dims_[4] * dims_[3] * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator= (const MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>& temp) {
    using TArray1D = Kokkos::View<T *,Layout,ExecSpace>;
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for

        order_ = temp.order_;
        length_ = temp.length_;
        this_array_ = temp.this_array_;
        mpi_datatype_ = temp.mpi_datatype_;
    }
    
    return *this;
}

// Return size
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::size() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
size_t MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::extent() {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::dims(size_t i) const {
    assert(i < order_ && "MPICArrayKokkos order (rank) does not match constructor, dim[i] does not exist!");
    assert(i >= 0 && dims_[i]>0 && "Access to MPICArrayKokkos dims is out of bounds!");
    return dims_[i];
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::order() const {
    return order_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::pointer() const {
    return this_array_.data();
}

//return the stored Kokkos view
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
Kokkos::View<T*, Layout, ExecSpace, MemoryTraits> MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_view() const {
    return this_array_;
}

//MPI_send wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::mtr_send(size_t count, int dest, int tag, MPI_Comm comm) {
    MPI_Send(this_array_.data(), count, mpi_datatype_, dest, tag, comm); 
}

//MPI_recv wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::mtr_recv(size_t count, int source, int tag, MPI_Comm comm) {
    MPI_Recv(this_array_.data(), count, mpi_datatype_, source, tag, comm, &mpi_status_); 
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::~MPICArrayKokkos() {}

////////////////////////////////////////////////////////////////////////////////
// End of MPICArrayKokkos
////////////////////////////////////////////////////////////////////////////////

/////////////////////////
// MPIDCArrayKokkos:  Dual type for managing data on both CPU and GPU.
/////////////////////////
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class MPIDCArrayKokkos {

    // this is manage
    using TArray1D = Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits>;
    
private:
    size_t dims_[7];
    size_t length_;
    size_t order_;  // tensor order (rank)
    MPI_Status mpi_status_;
    MPI_Datatype mpi_datatype_;
    TArray1D this_array_;

public:
    // Data member to access host view
    ViewCArray <T> host;

    MPIDCArrayKokkos();
    
    MPIDCArrayKokkos(MPI_Datatype mpi_type, size_t dim0, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIDCArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIDCArrayKokkos (MPI_Datatype mpi_type, size_t dim0, size_t dim1, size_t dim2, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIDCArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIDCArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, size_t dim4, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIDCArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, size_t dim4, size_t dim5, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIDCArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, size_t dim4, size_t dim5,
                 size_t dim6, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j, size_t k) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m,
                  size_t n) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j, size_t k, size_t l, size_t m,
                  size_t n, size_t o) const;
    
    KOKKOS_INLINE_FUNCTION
    MPIDCArrayKokkos& operator=(const MPIDCArrayKokkos& temp);

    // GPU Method
    // Method that returns size
    KOKKOS_INLINE_FUNCTION
    size_t size() const;

    // Host Method
    // Method that returns size
    KOKKOS_INLINE_FUNCTION
    size_t extent() const;

    KOKKOS_INLINE_FUNCTION
    size_t dims(size_t i) const;

    KOKKOS_INLINE_FUNCTION
    size_t order() const;
 
    // Method returns the raw device pointer of the Kokkos DualView
    KOKKOS_INLINE_FUNCTION
    T* device_pointer() const;

    // Method returns the raw host pointer of the Kokkos DualView
    KOKKOS_INLINE_FUNCTION
    T* host_pointer() const;

    // Method returns kokkos dual view
    KOKKOS_INLINE_FUNCTION
    TArray1D get_kokkos_dual_view() const;

    // Method that update host view
    void update_host();

    // Method that update device view
    void update_device();

    // MPI send wrapper
    void mtr_send(size_t count, int dest, int tag, MPI_Comm comm);

    // MPI recieve wrapper
    void mtr_recv(size_t count, int dest, int tag, MPI_Comm comm);

    // Deconstructor
    KOKKOS_INLINE_FUNCTION
    ~MPIDCArrayKokkos ();
}; // End of MPIDCArrayKokkos


// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIDCArrayKokkos() {
    length_ = order_ = 0;
    for (int i = 0; i < 7; i++) {
        dims_[i] = 0;
    }
}

// Overloaded 1D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIDCArrayKokkos(MPI_Datatype mpi_type, size_t dim0, const std::string& tag_string) {
    
    dims_[0] = dim0;
    order_ = 1;
    length_ = dim0;
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0);
    mpi_datatype_ = mpi_type;
}

// Overloaded 2D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIDCArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1, const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    order_ = 2;
    length_ = (dim0 * dim1);
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0, dim1);
    mpi_datatype_ = mpi_type;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIDCArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1,
                              size_t dim2, const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    order_ = 3;
    length_ = (dim0 * dim1 * dim2);
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0, dim1, dim2);
    mpi_datatype_ = mpi_type;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIDCArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1,
                              size_t dim2, size_t dim3, const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    order_ = 4;
    length_ = (dim0 * dim1 * dim2 * dim3);
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0, dim1, dim2, dim3);
    mpi_datatype_ = mpi_type;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIDCArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1,
                              size_t dim2, size_t dim3,
                              size_t dim4, const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    order_ = 5;
    length_ = (dim0 * dim1 * dim2 * dim3 * dim4);
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0, dim1, dim2, dim3, dim4);
    mpi_datatype_ = mpi_type;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIDCArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1,
                              size_t dim2, size_t dim3,
                              size_t dim4, size_t dim5, const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    dims_[5] = dim5;
    order_ = 6;
    length_ = (dim0 * dim1 * dim2 * dim3 * dim4 * dim5);
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0, dim1, dim2, dim3, dim4, dim5);
    mpi_datatype_ = mpi_type;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIDCArrayKokkos(MPI_Datatype mpi_type, size_t dim0, size_t dim1,
                              size_t dim2, size_t dim3,
                              size_t dim4, size_t dim5,
                              size_t dim6, const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    dims_[5] = dim5;
    dims_[6] = dim6;
    order_ = 7;
    length_ = (dim0 * dim1 * dim2 * dim3 * dim4 * dim5 * dim6);
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0, dim1, dim2, dim3, dim4, dim5, dim6);
    mpi_datatype_ = mpi_type;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i) const {
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in MPIDCArrayKokkos 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIDCArrayKokkos 1D!");
    return this_array_.d_view(i);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j) const {
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in MPIDCArrayKokkos 2D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIDCArrayKokkos 2D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIDCArrayKokkos 2D!");
    return this_array_.d_view(j + (i * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k) const {
    assert(order_ == 3 && "Tensor order (rank) does not match constructor in MPIDCArrayKokkos 3D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIDCArrayKokkos 3D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIDCArrayKokkos 3D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPIDCArrayKokkos 3D!");
    return this_array_.d_view(k + (j * dims_[2])
                                + (i * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l) const {
    assert(order_ == 4 && "Tensor order (rank) does not match constructor in MPIDCArrayKokkos 4D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIDCArrayKokkos 4D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIDCArrayKokkos 4D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPIDCArrayKokkos 4D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in MPIDCArrayKokkos 4D!");
    return this_array_.d_view(l + (k * dims_[3])
                                + (j * dims_[3] * dims_[2])
                                + (i * dims_[3] * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m) const {
    assert(order_ == 5 && "Tensor order (rank) does not match constructor in MPIDCArrayKokkos 5D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIDCArrayKokkos 5D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIDCArrayKokkos 5D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPIDCArrayKokkos 5D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in MPIDCArrayKokkos 5D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in MPIDCArrayKokkos 5D!");
    return this_array_.d_view(m + (l * dims_[4])
                                + (k * dims_[4] * dims_[3])
                                + (j * dims_[4] * dims_[3] * dims_[2])
                                + (i * dims_[4] * dims_[3] * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n) const {
    assert(order_ == 6 && "Tensor order (rank) does not match constructor in MPIDCArrayKokkos 6D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIDCArrayKokkos 6D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIDCArrayKokkos 6D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPIDCArrayKokkos 6D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in MPIDCArrayKokkos 6D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in MPIDCArrayKokkos 6D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in MPIDCArrayKokkos 6D!");
    return this_array_.d_view(n + (m * dims_[5])
                                + (l * dims_[5] * dims_[4])
                                + (k * dims_[5] * dims_[4] * dims_[3])
                                + (j * dims_[5] * dims_[4] * dims_[3] * dims_[2])
                                + (i * dims_[5] * dims_[4] * dims_[3] * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n, size_t o) const {
    assert(order_ == 7 && "Tensor order (rank) does not match constructor in MPIDCArrayKokkos 7D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIDCArrayKokkos 7D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIDCArrayKokkos 7D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPIDCArrayKokkos 7D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in MPIDCArrayKokkos 7D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in MPIDCArrayKokkos 7D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in MPIDCArrayKokkos 7D!");
    assert(o >= 0 && o < dims_[6] && "o is out of bounds in MPIDCArrayKokkos 7D!");
    return this_array_.d_view(o + (n * dims_[6])
                                + (m * dims_[6] * dims_[5])
                                + (l * dims_[6] * dims_[5] * dims_[4])
                                + (k * dims_[6] * dims_[5] * dims_[4] * dims_[3])
                                + (j * dims_[6] * dims_[5] * dims_[4] * dims_[3] * dims_[2])
                                + (i * dims_[6] * dims_[5] * dims_[4] * dims_[3] * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>& MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator= (const MPIDCArrayKokkos& temp) {
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for

        order_ = temp.order_;
        length_ = temp.length_;
        this_array_ = temp.this_array_;
        host = temp.host;
    }
    
    return *this;
}

// Return size
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::size() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::extent() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::dims(size_t i) const {
    assert(i < order_ && "MPIDCArrayKokkos order (rank) does not match constructor, dim[i] does not exist!");
    assert(i >= 0 && dims_[i]>0 && "Access to MPIDCArrayKokkos dims is out of bounds!");
    return dims_[i];
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::order() const {
    return order_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::device_pointer() const {
    return this_array_.d_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::host_pointer() const {
    return this_array_.h_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits> MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_dual_view() const {
  return this_array_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::update_host() {

    this_array_.template modify<typename TArray1D::execution_space>();
    this_array_.template sync<typename TArray1D::host_mirror_space>();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::update_device() {

    this_array_.template modify<typename TArray1D::host_mirror_space>();
    this_array_.template sync<typename TArray1D::execution_space>();
}

//MPI_send wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::mtr_send(size_t count, int dest, int tag, MPI_Comm comm) {
    update_host();
    MPI_Send(host_pointer(), count, mpi_datatype_, dest, tag, comm); 
}

//MPI_recv wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::mtr_recv(size_t count, int source, int tag, MPI_Comm comm) {
    MPI_Recv(host_pointer(), count, mpi_datatype_, source, tag, comm, &mpi_status_); 
    update_device();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
MPIDCArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::~MPIDCArrayKokkos() {}

} // end namespace

#endif // end if have MPI

#endif // MPI_TYPES_H

