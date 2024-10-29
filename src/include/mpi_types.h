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
#include <typeinfo>
#ifdef HAVE_MPI
#include <mpi.h>

namespace mtr
{

/////////////////////////
// Not a class, just helping functions for all MPI
/////////////////////////
template <typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
size_t simple_decomp_row_size(int world_size, int rank, int big_N) {
    int grid_size, rem, row, col, add; 
    size_t row_size;
    grid_size = sqrt(world_size); // assumes a square rootable world size
    rem = big_N % grid_size; // extra points
    row = rank / grid_size;
    col = rank % grid_size;
    add = (col < rem);
    //add = col / (grid_size - rem); // 1 if we have an extra piece, 0 otherwise
    row_size = big_N / grid_size + add;
    return row_size;
}

template <typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
size_t simple_decomp_row_start(int world_size, int rank, int big_N) {
    int grid_size, rem, row, col, add, sub; 
    size_t row_size, row_start;
    grid_size = sqrt(world_size); // assumes a square rootable world size
    rem = big_N % grid_size; // extra points
    row = rank / grid_size;
    col = rank % grid_size;
    add = (col < rem);
    //add = col / (grid_size - rem); // 1 if we have an extra piece, 0 otherwise
    sub = add * (grid_size - rem); // subtraction from start based on how many other ranks have an extra piece
    row_size = big_N / grid_size + add;
    row_start = row * row_size - sub;
    return row_start;
}

template <typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
size_t simple_decomp_col_size(int world_size, int rank, int big_N) {
    int grid_size, rem, row, col, add; 
    size_t col_size;
    grid_size = sqrt(world_size); // assumes a square rootable world size
    rem = big_N % grid_size; // extra points
    row = rank / grid_size;
    col = rank % grid_size;
    add = (row < rem);
    //add = row / (grid_size - rem); // 1 if we have an extra piece, 0 otherwise
    col_size = big_N / grid_size + add;
    return col_size;
}

template <typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
size_t simple_decomp_col_start(int world_size, int rank, int big_N) {
    int grid_size, rem, row, col, add, sub; 
    size_t col_size, col_start;
    grid_size = sqrt(world_size); // assumes a square rootable world size
    rem = big_N % grid_size; // extra points
    row = rank / grid_size;
    col = rank % grid_size;
    add = (row < rem);
    //add = row / (grid_size - rem); // 1 if we have an extra piece, 0 otherwise
    sub = add * (grid_size - rem); // subtraction from start based on how many other ranks have an extra piece
    col_size = big_N / grid_size + add;
    col_start = col * col_size - sub;
    return col_start;
}

template <typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
size_t PartitionSize(int world_size, int rank, int decomp_dim) {
    size_t neighbors = 0;
    if (decomp_dim == 1) {
        int i_w = rank - 1;
        int i_e = rank + 1;
        if (i_w >= 0) {
            neighbors++;
        }
        if (i_e < world_size) {
            neighbors++;
        }
        return neighbors;
    }
    else if (decomp_dim == 2) {
        int grid_size = sqrt(world_size);
        int grid_i = rank % grid_size;
        int grid_j = rank / grid_size;
        int j_n = grid_j - 1;
        int j_s = grid_j + 1;
        int i_w = grid_i - 1;
        int i_e = grid_i + 1;
        if (j_n >= 0) {
            neighbors++;
        }
        if (j_s < world_size) {
            neighbors++;
        }
        if (i_w >= 0) {
            neighbors++;
        }
        if (i_e < world_size) {
            neighbors++;
        }
        return neighbors;
    }
}

/////////////////////////
// MPIHaloKokkos:  Really only used for internal comms in the original MPIHaloKokkos class
/////////////////////////
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class MPIHaloKokkos {

    // this is manage
    using TArray1D = Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits>;
    
protected:
    size_t dims_[7];
    size_t length_;
    size_t order_;  // tensor order (rank)
    // mpi arrays below are all order north, south, west, east
    int mpi_neighbor_;
    int mpi_tag_;
    int mpi_halos_;
    MPI_Comm mpi_comm_;
    MPI_Status mpi_status_;
    MPI_Datatype mpi_datatype_;
    MPI_Request mpi_request_;
    TArray1D this_array_;
    
    void set_mpi_type();

public:
    // Data member to access host view
    ViewCArray <T> host;

    MPIHaloKokkos();
    
    MPIHaloKokkos(size_t dim0, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIHaloKokkos(size_t dim0, size_t dim1, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIHaloKokkos(size_t dim0, size_t dim1, size_t dim2, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIHaloKokkos(size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIHaloKokkos(size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, size_t dim4, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIHaloKokkos(size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, size_t dim4, size_t dim5, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPIHaloKokkos(size_t dim0, size_t dim1, size_t dim2,
                 size_t dim3, size_t dim4, size_t dim5,
                 size_t dim6, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    // These functions can setup the data needed for halo send/receives
    // Not necessary for standard MPI comms
    void mpi_setup(int neighbor, int tag, int halos, MPI_Comm comm);

    int get_neighbor();

    int get_tag();

    int get_halos();

    MPI_Comm get_comm();

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
    MPIHaloKokkos& operator=(const MPIHaloKokkos& temp);

    // GPU Method
    // Method that returns size
    KOKKOS_INLINE_FUNCTION
    size_t size() const;

    // Host Method
    // Method that returns size
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
    void halo_send();

    // MPI recieve wrapper
    void halo_recv();

    // MPI send wrapper
    void halo_isend();

    // MPI recieve wrapper
    void halo_irecv();

    // MPI wait wrapper for sender
    void wait_send();

    // MPI wait wrapper for receiver
    void wait_recv();

    // Deconstructor
    virtual KOKKOS_INLINE_FUNCTION
    ~MPIHaloKokkos ();
}; // End of MPIHaloKokkos


// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIHaloKokkos() {
    length_ = order_ = 0;
    for (int i = 0; i < 7; i++) {
        dims_[i] = 0;
    }
}

// Overloaded 1D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIHaloKokkos(size_t dim0, const std::string& tag_string) {
    
    dims_[0] = dim0;
    order_ = 1;
    length_ = dim0;
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0);
    set_mpi_type();
    mpi_comm_ = MPI_COMM_WORLD;
}

// Overloaded 2D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIHaloKokkos(size_t dim0, size_t dim1, const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    order_ = 2;
    length_ = (dim0 * dim1);
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0, dim1);
    set_mpi_type();
    mpi_comm_ = MPI_COMM_WORLD;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIHaloKokkos(size_t dim0, size_t dim1,
                              size_t dim2, const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    order_ = 3;
    length_ = (dim0 * dim1 * dim2);
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0, dim1, dim2);
    set_mpi_type();
    mpi_comm_ = MPI_COMM_WORLD;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIHaloKokkos(size_t dim0, size_t dim1,
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
    set_mpi_type();
    mpi_comm_ = MPI_COMM_WORLD;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIHaloKokkos(size_t dim0, size_t dim1,
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
    set_mpi_type();
    mpi_comm_ = MPI_COMM_WORLD;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIHaloKokkos(size_t dim0, size_t dim1,
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
    set_mpi_type();
    mpi_comm_ = MPI_COMM_WORLD;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIHaloKokkos(size_t dim0, size_t dim1,
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
    set_mpi_type();
    mpi_comm_ = MPI_COMM_WORLD;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::set_mpi_type() {
    if (typeid(T).name() == typeid(bool).name()) {
        mpi_datatype_ = MPI_C_BOOL;
    }
    else if (typeid(T).name() == typeid(int).name()) {
        mpi_datatype_ = MPI_INT;
    }
    else if (typeid(T).name() == typeid(long int).name()) {
        mpi_datatype_ = MPI_LONG;
    }
    else if (typeid(T).name() == typeid(long long int).name()) {
        mpi_datatype_ = MPI_LONG_LONG_INT;
    }
    else if (typeid(T).name() == typeid(float).name()) {
        mpi_datatype_ = MPI_FLOAT;
    }
    else if (typeid(T).name() == typeid(double).name()) {
        mpi_datatype_ = MPI_DOUBLE;
    }
    else {
        printf("Your entered MPIHaloKokkos type is not a supported type for MPI communications and is being set to int\n");
        mpi_datatype_ = MPI_INT;
    }
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i) const {
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in MPIHaloKokkos 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIHaloKokkos 1D!");
    return this_array_.d_view(i);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j) const {
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in MPIHaloKokkos 2D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIHaloKokkos 2D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIHaloKokkos 2D!");
    return this_array_.d_view(j + (i * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k) const {
    assert(order_ == 3 && "Tensor order (rank) does not match constructor in MPIHaloKokkos 3D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIHaloKokkos 3D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIHaloKokkos 3D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPIHaloKokkos 3D!");
    return this_array_.d_view(k + (j * dims_[2])
                                + (i * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l) const {
    assert(order_ == 4 && "Tensor order (rank) does not match constructor in MPIHaloKokkos 4D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIHaloKokkos 4D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIHaloKokkos 4D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPIHaloKokkos 4D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in MPIHaloKokkos 4D!");
    return this_array_.d_view(l + (k * dims_[3])
                                + (j * dims_[3] * dims_[2])
                                + (i * dims_[3] * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m) const {
    assert(order_ == 5 && "Tensor order (rank) does not match constructor in MPIHaloKokkos 5D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIHaloKokkos 5D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIHaloKokkos 5D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPIHaloKokkos 5D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in MPIHaloKokkos 5D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in MPIHaloKokkos 5D!");
    return this_array_.d_view(m + (l * dims_[4])
                                + (k * dims_[4] * dims_[3])
                                + (j * dims_[4] * dims_[3] * dims_[2])
                                + (i * dims_[4] * dims_[3] * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n) const {
    assert(order_ == 6 && "Tensor order (rank) does not match constructor in MPIHaloKokkos 6D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIHaloKokkos 6D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIHaloKokkos 6D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPIHaloKokkos 6D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in MPIHaloKokkos 6D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in MPIHaloKokkos 6D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in MPIHaloKokkos 6D!");
    return this_array_.d_view(n + (m * dims_[5])
                                + (l * dims_[5] * dims_[4])
                                + (k * dims_[5] * dims_[4] * dims_[3])
                                + (j * dims_[5] * dims_[4] * dims_[3] * dims_[2])
                                + (i * dims_[5] * dims_[4] * dims_[3] * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n, size_t o) const {
    assert(order_ == 7 && "Tensor order (rank) does not match constructor in MPIHaloKokkos 7D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIHaloKokkos 7D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPIHaloKokkos 7D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPIHaloKokkos 7D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in MPIHaloKokkos 7D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in MPIHaloKokkos 7D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in MPIHaloKokkos 7D!");
    assert(o >= 0 && o < dims_[6] && "o is out of bounds in MPIHaloKokkos 7D!");
    return this_array_.d_view(o + (n * dims_[6])
                                + (m * dims_[6] * dims_[5])
                                + (l * dims_[6] * dims_[5] * dims_[4])
                                + (k * dims_[6] * dims_[5] * dims_[4] * dims_[3])
                                + (j * dims_[6] * dims_[5] * dims_[4] * dims_[3] * dims_[2])
                                + (i * dims_[6] * dims_[5] * dims_[4] * dims_[3] * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>& MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::operator= (const MPIHaloKokkos& temp) {
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for

        order_ = temp.order_;
        length_ = temp.length_;
        this_array_ = temp.this_array_;
        host = temp.host;
        mpi_neighbor_ = temp.mpi_neighbor_;
        mpi_tag_ = temp.mpi_tag_;
        mpi_comm_ = temp.mpi_comm_;
        mpi_status_ = temp.mpi_status_;
        mpi_datatype_ = temp.mpi_datatype_;
        mpi_request_ = temp.mpi_request_;
    }
    
    return *this;
}

// Return size
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::size() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
size_t MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::extent() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::dims(size_t i) const {
    assert(i < order_ && "MPIHaloKokkos order (rank) does not match constructor, dim[i] does not exist!");
    assert(i >= 0 && dims_[i]>0 && "Access to MPIHaloKokkos dims is out of bounds!");
    return dims_[i];
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::order() const {
    return order_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::device_pointer() const {
    return this_array_.d_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::host_pointer() const {
    return this_array_.h_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits> MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_dual_view() const {
  return this_array_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::update_host() {

    this_array_.template modify<typename TArray1D::execution_space>();
    this_array_.template sync<typename TArray1D::host_mirror_space>();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::update_device() {

    this_array_.template modify<typename TArray1D::host_mirror_space>();
    this_array_.template sync<typename TArray1D::execution_space>();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::mpi_setup(int neighbor, int tag, int halos, MPI_Comm comm) {
    mpi_neighbor_ = neighbor;
    mpi_tag_ = tag;
    mpi_halos_ = halos;
    mpi_comm_ = comm;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
int MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::get_neighbor() {
    return mpi_neighbor_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
int MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::get_tag() {
    return mpi_tag_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
int MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::get_halos() {
    return mpi_halos_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPI_Comm MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::get_comm() {
    return mpi_comm_;
}

//MPI_iSend halo wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::halo_isend() {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Isend(device_pointer(), size(), mpi_datatype_, mpi_neighbor_, mpi_tag_, mpi_comm_, &mpi_request_); 
#else
    update_host();
    MPI_Isend(host_pointer(), size(), mpi_datatype_, mpi_neighbor_, mpi_tag_, mpi_comm_, &mpi_request_); 
#endif
}

//MPI_iRecv halo wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::halo_irecv() {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Irecv(device_pointer(), size(), mpi_datatype_, mpi_neighbor_, mpi_tag_, mpi_comm_, &mpi_request_); 
#else
    MPI_Irecv(host_pointer(), size(), mpi_datatype_, mpi_neighbor_, mpi_tag_, mpi_comm_, &mpi_request_); 
#endif
}

//MPI_Wait wrapper for the sender
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::wait_send() {
    MPI_Wait(&mpi_request_, &mpi_status_); 
}

//MPI_Wait wrapper for the receiver
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::wait_recv() {
    MPI_Wait(&mpi_request_, &mpi_status_); 
#ifndef HAVE_GPU_AWARE_MPI
    update_device();
#endif
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
MPIHaloKokkos<T,Layout,ExecSpace,MemoryTraits>::~MPIHaloKokkos() {}

////////////////////////////////////////////////////////////////////////////////
// End of MPIHaloKokkos
////////////////////////////////////////////////////////////////////////////////

/////////////////////////
// MPIPartitionKokkos:  Really only used for internal comms in the original MPIHaloKokkos class
/////////////////////////
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class MPIPartitionKokkos {

    // this is manage
    using TArray1D = Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits>;
    
protected:
    size_t dims_[7];
    size_t length_;
    size_t order_;  // tensor order (rank)
    
public:
    TArray1D this_array_;
    int mpi_world_size_;
    int mpi_rank_;
    int mpi_halos_;
    MPI_Comm mpi_comm_;
    MPIHaloKokkos <T> send_n_, recv_n_, send_s_, recv_s_, send_w_, recv_w_, send_e_, recv_e_;

    MPIPartitionKokkos();
    
    MPIPartitionKokkos(size_t dim0, int world_size, int rank, int halos, MPI_Comm comm, const std::string& tag_string = DEFAULTSTRINGARRAY);

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i) const;

    KOKKOS_INLINE_FUNCTION
    size_t size() const;

    size_t extent() const;

    KOKKOS_INLINE_FUNCTION
    MPIPartitionKokkos& operator=(const MPIPartitionKokkos& temp);

    // Deconstructor
    virtual KOKKOS_INLINE_FUNCTION
    ~MPIPartitionKokkos ();
}; // End of MPIPartitionKokkos


// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIPartitionKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIPartitionKokkos() {
    length_ = order_ = 0;
    for (int i = 0; i < 7; i++) {
        dims_[i] = 0;
    }
}

// Overloaded 1D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIPartitionKokkos<T,Layout,ExecSpace,MemoryTraits>::MPIPartitionKokkos(size_t dim0, int world_size, int rank, int halos, MPI_Comm comm, const std::string& tag_string) {
    
    dims_[0] = dim0;
    order_ = 1;
    length_ = dim0;
    this_array_ = TArray1D(tag_string, length_);
    mpi_world_size_ = world_size;
    mpi_rank_ = rank;
    mpi_halos_ = halos;
    mpi_comm_ = comm;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIPartitionKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i) const {
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in MPIPartitionKokkos 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIPartitionKokkos 1D!");
    return this_array_.d_view(i);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
MPIPartitionKokkos<T,Layout,ExecSpace,MemoryTraits>& MPIPartitionKokkos<T,Layout,ExecSpace,MemoryTraits>::operator= (const MPIPartitionKokkos& temp) {
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for

        order_ = temp.order_;
        length_ = temp.length_;
        mpi_world_size_ = temp.mpi_world_size_;
        mpi_rank_ = temp.mpi_rank_;
        mpi_halos_ = temp.mpi_halos_;
        mpi_comm_ = temp.mpi_comm_;
        this_array_ = temp.this_array_;
        send_n_ = temp.send_n_;
        recv_n_ = temp.recv_n_;
        send_s_ = temp.send_s_;
        recv_s_ = temp.recv_s_;
        send_w_ = temp.send_w_;
        recv_w_ = temp.recv_w_;
        send_e_ = temp.send_e_;
        recv_e_ = temp.recv_e_;
    }
    
    return *this;
}

// Return size
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPIPartitionKokkos<T,Layout,ExecSpace,MemoryTraits>::size() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
size_t MPIPartitionKokkos<T,Layout,ExecSpace,MemoryTraits>::extent() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
MPIPartitionKokkos<T,Layout,ExecSpace,MemoryTraits>::~MPIPartitionKokkos() {}

////////////////////////////////////////////////////////////////////////////////
// End of MPIPartitionKokkos
////////////////////////////////////////////////////////////////////////////////

/////////////////////////
// MPIPartition2Kokkos:  Really only used for internal comms in the original MPIHaloKokkos class
/////////////////////////
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class MPIPartition2Kokkos {

    // this is manage
    using TArray1D = Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits>;
    
protected:
    size_t dims_[7];
    size_t length_;
    size_t order_;  // tensor order (rank)
    
public:
    TArray1D this_array_;
    int mpi_world_size_;
    int mpi_rank_;
    int mpi_halos_;
    MPI_Comm mpi_comm_;
    MPIHaloKokkos <T> send_n_, send_s_, send_w_, send_e_, recv_n_, recv_s_, recv_w_, recv_e_;
    CArray <MPIHaloKokkos <T>> sends_;
    CArray <MPIHaloKokkos <T>> recvs_;

    MPIPartition2Kokkos();
    
    MPIPartition2Kokkos(size_t dim0, int world_size, int rank, int halos, MPI_Comm comm, const std::string& tag_string = DEFAULTSTRINGARRAY);

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i) const;

    KOKKOS_INLINE_FUNCTION
    size_t size() const;

    size_t extent() const;

    KOKKOS_INLINE_FUNCTION
    MPIPartition2Kokkos& operator=(const MPIPartition2Kokkos& temp);

    // Deconstructor
    virtual KOKKOS_INLINE_FUNCTION
    ~MPIPartition2Kokkos ();
}; // End of MPIPartition2Kokkos


// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIPartition2Kokkos<T,Layout,ExecSpace,MemoryTraits>::MPIPartition2Kokkos() {
    length_ = order_ = 0;
    for (int i = 0; i < 7; i++) {
        dims_[i] = 0;
    }
}

// Overloaded 1D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPIPartition2Kokkos<T,Layout,ExecSpace,MemoryTraits>::MPIPartition2Kokkos(size_t dim0, int world_size, int rank, int halos, MPI_Comm comm, const std::string& tag_string) {
    
    dims_[0] = dim0;
    order_ = 1;
    length_ = dim0;
    this_array_ = TArray1D(tag_string, 1);
    sends_ = CArray <MPIHaloKokkos <T>> (length_);
    recvs_ = CArray <MPIHaloKokkos <T>> (length_);
    mpi_world_size_ = world_size;
    mpi_rank_ = rank;
    mpi_halos_ = halos;
    mpi_comm_ = comm;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPIPartition2Kokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i) const {
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in MPIPartition2Kokkos 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPIPartition2Kokkos 1D!");
    return this_array_.d_view(i);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
MPIPartition2Kokkos<T,Layout,ExecSpace,MemoryTraits>& MPIPartition2Kokkos<T,Layout,ExecSpace,MemoryTraits>::operator= (const MPIPartition2Kokkos& temp) {
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for

        order_ = temp.order_;
        length_ = temp.length_;
        mpi_world_size_ = temp.mpi_world_size_;
        mpi_rank_ = temp.mpi_rank_;
        mpi_halos_ = temp.mpi_halos_;
        mpi_comm_ = temp.mpi_comm_;
        this_array_ = temp.this_array_;
        sends_ = temp.sends_;
        recvs_ = temp.recvs_;
    }
    
    return *this;
}

// Return size
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t MPIPartition2Kokkos<T,Layout,ExecSpace,MemoryTraits>::size() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
size_t MPIPartition2Kokkos<T,Layout,ExecSpace,MemoryTraits>::extent() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
MPIPartition2Kokkos<T,Layout,ExecSpace,MemoryTraits>::~MPIPartition2Kokkos() {}

////////////////////////////////////////////////////////////////////////////////
// End of MPIPartition2Kokkos
////////////////////////////////////////////////////////////////////////////////

/////////////////////////
// MPICArrayKokkos:  Dual type for managing distributed data on both CPU and GPU.
/////////////////////////
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class MPICArrayKokkos {

    // this is manage
    using TArray1D = Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits>;
    
protected:
    size_t dims_[3];
    size_t length_;
    size_t order_;  // tensor order (rank)
    int rank_;
    int world_size_;
    MPI_Status mpi_status_;
    MPI_Datatype mpi_datatype_;
    MPI_Request mpi_request_;
    TArray1D this_array_;
    //MPIPartitionKokkos <T> mpi_partition_;
    MPIPartition2Kokkos <T> mpi_partition_;
    
    void set_mpi_type();

public:
    // Data member to access host view
    ViewCArray <T> host;

    MPICArrayKokkos();
    
    MPICArrayKokkos(size_t dim0, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPICArrayKokkos(size_t dim0, size_t dim1, const std::string& tag_string = DEFAULTSTRINGARRAY);

    MPICArrayKokkos(size_t dim0, size_t dim1, size_t dim2, const std::string& tag_string = DEFAULTSTRINGARRAY);

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j, size_t k) const;

    KOKKOS_INLINE_FUNCTION
    MPICArrayKokkos& operator=(const MPICArrayKokkos& temp);

    // GPU Method
    // Method that returns size
    KOKKOS_INLINE_FUNCTION
    size_t size() const;

    // Host Method
    // Method that returns size
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
    void send(size_t count, int dest, int tag, MPI_Comm comm);

    // MPI recieve wrapper
    void recv(size_t count, int dest, int tag, MPI_Comm comm);

    // MPI broadcast wrapper
    void broadcast(size_t count, int root, MPI_Comm comm);

    // MPI scatter wrapper
    void scatter(size_t send_count, MPICArrayKokkos recv_buffer, size_t recv_count, int root, MPI_Comm comm);

    // MPI gather wrapper
    void gather(size_t send_count, MPICArrayKokkos recv_buffer, size_t recv_count, int root, MPI_Comm comm);

    // MPI allgather wrapper
    void allgather(size_t send_count, MPICArrayKokkos recv_buffer, size_t recv_count, MPI_Comm comm);

    // MPI send wrapper
    void isend(size_t count, int dest, int tag, MPI_Comm comm);

    // MPI recieve wrapper
    void irecv(size_t count, int dest, int tag, MPI_Comm comm);

    // MPI wait wrapper for sender
    void wait_send();

    // MPI wait wrapper for receiver
    void wait_recv();

    // set up the halo arrays
    //void mpi_decomp(int world_size, int rank, int halos, MPI_Comm comm);
    void mpi_decomp_unstructured(MPIPartition2Kokkos <T> partition);

    void mpi_decomp(MPIPartitionKokkos <T> partition);

    // update the halo arrays and transfer data to this_array_
    void mpi_halo_update();

    // Deconstructor
    virtual KOKKOS_INLINE_FUNCTION
    ~MPICArrayKokkos ();
}; // End of MPICArrayKokkos


// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos() {
    length_ = order_ = 0;
    for (int i = 0; i < 3; i++) {
        dims_[i] = 0;
    }
}

// Overloaded 1D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(size_t dim0, const std::string& tag_string) {
    
    dims_[0] = dim0;
    order_ = 1;
    length_ = dim0;
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0);
    set_mpi_type();
}

// Overloaded 2D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(size_t dim0, size_t dim1, const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    order_ = 2;
    length_ = (dim0 * dim1);
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0, dim1);
    set_mpi_type();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::MPICArrayKokkos(size_t dim0, size_t dim1,
                              size_t dim2, const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    dims_[2] = dim2;
    order_ = 3;
    length_ = (dim0 * dim1 * dim2);
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0, dim1, dim2);
    set_mpi_type();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::set_mpi_type() {
    if (typeid(T).name() == typeid(bool).name()) {
        mpi_datatype_ = MPI_C_BOOL;
    }
    else if (typeid(T).name() == typeid(int).name()) {
        mpi_datatype_ = MPI_INT;
    }
    else if (typeid(T).name() == typeid(long int).name()) {
        mpi_datatype_ = MPI_LONG;
    }
    else if (typeid(T).name() == typeid(long long int).name()) {
        mpi_datatype_ = MPI_LONG_LONG_INT;
    }
    else if (typeid(T).name() == typeid(float).name()) {
        mpi_datatype_ = MPI_FLOAT;
    }
    else if (typeid(T).name() == typeid(double).name()) {
        mpi_datatype_ = MPI_DOUBLE;
    }
    else {
        printf("Your entered MPICArrayKokkos type is not a supported type for MPI communications and is being set to int\n");
        mpi_datatype_ = MPI_INT;
    }
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i) const {
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in MPICArrayKokkos 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPICArrayKokkos 1D!");
    return this_array_.d_view(i);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j) const {
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in MPICArrayKokkos 2D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPICArrayKokkos 2D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPICArrayKokkos 2D!");
    return this_array_.d_view(j + (i * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k) const {
    assert(order_ == 3 && "Tensor order (rank) does not match constructor in MPICArrayKokkos 3D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in MPICArrayKokkos 3D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in MPICArrayKokkos 3D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in MPICArrayKokkos 3D!");
    return this_array_.d_view(k + (j * dims_[2])
                                + (i * dims_[2] * dims_[1]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>& MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::operator= (const MPICArrayKokkos& temp) {
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for

        order_ = temp.order_;
        length_ = temp.length_;
        this_array_ = temp.this_array_;
        host = temp.host;
        mpi_partition_ = temp.mpi_partition_;
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
size_t MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::extent() const {
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
T* MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::device_pointer() const {
    return this_array_.d_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::host_pointer() const {
    return this_array_.h_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits> MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_dual_view() const {
  return this_array_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::update_host() {

    this_array_.template modify<typename TArray1D::execution_space>();
    this_array_.template sync<typename TArray1D::host_mirror_space>();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::update_device() {

    this_array_.template modify<typename TArray1D::host_mirror_space>();
    this_array_.template sync<typename TArray1D::execution_space>();
}

//MPI_Send wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::send(size_t count, int dest, int tag, MPI_Comm comm) {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Send(device_pointer(), count, mpi_datatype_, dest, tag, comm); 
#else
    update_host();
    MPI_Send(host_pointer(), count, mpi_datatype_, dest, tag, comm); 
#endif
}

//MPI_Recv wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::recv(size_t count, int source, int tag, MPI_Comm comm) {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Recv(device_pointer(), count, mpi_datatype_, source, tag, comm, &mpi_status_); 
#else
    MPI_Recv(host_pointer(), count, mpi_datatype_, source, tag, comm, &mpi_status_); 
    update_device();
#endif
}

//MPI_Bcast wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::broadcast(size_t count, int root, MPI_Comm comm) {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Bcast(device_pointer(), count, mpi_datatype_, root, comm); 
#else
    update_host();
    MPI_Bcast(host_pointer(), count, mpi_datatype_, root, comm); 
    update_device();
#endif
}

//MPI_Scatter wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::scatter(size_t send_count, MPICArrayKokkos recv_buffer, size_t recv_count, int root, MPI_Comm comm) {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Scatter(device_pointer(), send_count, mpi_datatype_, recv_buffer.device_pointer(), recv_count, mpi_datatype_, root, comm); 
#else
    update_host();
    MPI_Scatter(host_pointer(), send_count, mpi_datatype_, recv_buffer.host_pointer(), recv_count, mpi_datatype_, root, comm); 
    recv_buffer.update_device();
#endif
}

//MPI_Gather wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::gather(size_t send_count, MPICArrayKokkos recv_buffer, size_t recv_count, int root, MPI_Comm comm) {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Gather(device_pointer(), send_count, mpi_datatype_, recv_buffer.device_pointer(), recv_count, mpi_datatype_, root, comm); 
#else
    update_host();
    MPI_Gather(host_pointer(), send_count, mpi_datatype_, recv_buffer.host_pointer(), recv_count, mpi_datatype_, root, comm); 
    recv_buffer.update_device();
#endif
}

//MPI_AllGather wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::allgather(size_t send_count, MPICArrayKokkos recv_buffer, size_t recv_count, MPI_Comm comm) {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Allgather(device_pointer(), send_count, mpi_datatype_, recv_buffer.device_pointer(), recv_count, mpi_datatype_, comm); 
#else
    update_host();
    MPI_Allgather(host_pointer(), send_count, mpi_datatype_, recv_buffer.host_pointer(), recv_count, mpi_datatype_, comm); 
    recv_buffer.update_device();
#endif
}

//MPI_Isend wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::isend(size_t count, int dest, int tag, MPI_Comm comm) {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Isend(device_pointer(), count, mpi_datatype_, dest, tag, comm, &mpi_request_); 
#else
    update_host();
    MPI_Isend(host_pointer(), count, mpi_datatype_, dest, tag, comm, &mpi_request_); 
#endif
}

//MPI_Irecv wrapper
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::irecv(size_t count, int source, int tag, MPI_Comm comm) {
#ifdef HAVE_GPU_AWARE_MPI
    MPI_Irecv(device_pointer(), count, mpi_datatype_, source, tag, comm, &mpi_request_); 
#else
    MPI_Irecv(host_pointer(), count, mpi_datatype_, source, tag, comm, &mpi_request_); 
#endif
}

//MPI_Wait wrapper for the sender
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::wait_send() {
    MPI_Wait(&mpi_request_, &mpi_status_); 
}

//MPI_Wait wrapper for the receiver
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::wait_recv() {
    MPI_Wait(&mpi_request_, &mpi_status_); 
#ifndef HAVE_GPU_AWARE_MPI
    update_device();
#endif
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::mpi_decomp_unstructured(MPIPartition2Kokkos <T> partition) {
    mpi_partition_ = partition;
    int neighbors = mpi_partition_.extent();
    int neighb, tag;
    int rank = mpi_partition_.mpi_rank_;
    int world_size = mpi_partition_.mpi_world_size_;
    int halos = mpi_partition_.mpi_halos_;
    MPI_Comm comm = mpi_partition_.mpi_comm_;
    if (order_ == 1) {
        int nidx = 0;
        int i_w = rank - 1;
        int i_e = rank + 1;
        if (i_w >= 0) {
            neighb = i_w;
            printf("%d) %d\n", rank, neighb);
            tag = rank * 10 + neighb;
            mpi_partition_.sends_(nidx) = MPIHaloKokkos <T> (1); 
            mpi_partition_.sends_(nidx).mpi_setup(neighb, tag, halos, comm); 
            tag = neighb * 10 + rank;
            mpi_partition_.recvs_(nidx) = MPIHaloKokkos <T> (1); 
            mpi_partition_.recvs_(nidx).mpi_setup(neighb, tag, halos, comm); 
            nidx++;
        }
        if (i_e < world_size) {
            neighb = i_e;
            printf("%d) %d\n", rank, neighb);
            tag = rank * 10 + neighb;
            mpi_partition_.sends_(nidx) = MPIHaloKokkos <T> (1); 
            mpi_partition_.sends_(nidx).mpi_setup(neighb, tag, halos, comm); 
            tag = neighb * 10 + rank;
            mpi_partition_.recvs_(nidx) = MPIHaloKokkos <T> (1); 
            mpi_partition_.recvs_(nidx).mpi_setup(neighb, tag, halos, comm); 
            nidx++;
        }
    }
    if (order_ == 2) {
        int nidx = 0;
        int mpi_dim_size = sqrt(world_size);
        int world_i = rank % mpi_dim_size;
        int world_j = rank / mpi_dim_size;
        int j_n = world_j - 1;
        int j_s = world_j + 1;
        int i_w = world_i - 1;
        int i_e = world_i + 1;
        if (j_n >= 0) {
            neighb = j_n * mpi_dim_size + world_i;
            printf("%d) %d\n", rank, neighb);
            tag = rank * 10 + neighb;
            //mpi_partition_.sends_(nidx) = MPIHaloKokkos <T> (dims_[0] - 2); 
            //mpi_partition_.sends_(nidx).mpi_setup(neighb, tag, halos, comm); 
            tag = neighb * 10 + rank;
            //mpi_partition_.recvs_(nidx) = MPIHaloKokkos <T> (dims_[0] - 2); 
            //mpi_partition_.recvs_(nidx).mpi_setup(neighb, tag, halos, comm); 
            nidx++;
        }
        if (j_s < mpi_dim_size) {
            neighb = j_s * mpi_dim_size + world_i;
            printf("%d) %d\n", rank, neighb);
            tag = rank * 10 + neighb;
            //mpi_partition_.sends_(nidx) = MPIHaloKokkos <T> (dims_[0] - 2); 
            //mpi_partition_.sends_(nidx).mpi_setup(neighb, tag, halos, comm); 
            tag = neighb * 10 + rank;
            //mpi_partition_.recvs_(nidx) = MPIHaloKokkos <T> (dims_[0] - 2); 
            //mpi_partition_.recvs_(nidx).mpi_setup(neighb, tag, halos, comm); 
            nidx++;
        }
        if (i_w >= 0) {
            neighb = world_j * mpi_dim_size + i_w;
            printf("%d) %d\n", rank, neighb);
            tag = rank * 10 + neighb;
            //mpi_partition_.sends_(nidx) = MPIHaloKokkos <T> (dims_[1] - 2); 
            //mpi_partition_.sends_(nidx).mpi_setup(neighb, tag, halos, comm); 
            tag = neighb * 10 + rank;
            //mpi_partition_.recvs_(nidx) = MPIHaloKokkos <T> (dims_[1] - 2); 
            //mpi_partition_.recvs_(nidx).mpi_setup(neighb, tag, halos, comm); 
            nidx++;
        }
        if (i_e < mpi_dim_size) {
            neighb = world_j * mpi_dim_size + i_e;
            printf("%d) %d\n", rank, neighb);
            tag = rank * 10 + neighb;
            //mpi_partition_.sends_(nidx) = MPIHaloKokkos <T> (dims_[1] - 2); 
            //mpi_partition_.sends_(nidx).mpi_setup(neighb, tag, halos, comm); 
            tag = neighb * 10 + rank;
            //mpi_partition_.recvs_(nidx) = MPIHaloKokkos <T> (dims_[1] - 2); 
            //mpi_partition_.recvs_(nidx).mpi_setup(neighb, tag, halos, comm); 
            nidx++;
        }
    }
    if (order_ == 3) {
        int nidx = 0;
        int mpi_dim_size = sqrt(world_size);
        int world_i = rank % mpi_dim_size;
        int world_j = rank / mpi_dim_size;
        int j_n = world_j - 1;
        int j_s = world_j + 1;
        int i_w = world_i - 1;
        int i_e = world_i + 1;
        if (j_n >= 0) {
            neighb = j_n * mpi_dim_size + world_i;
            tag = rank * 10 + neighb;
            mpi_partition_.sends_(nidx) = MPIHaloKokkos <T> (dims_[0] - 2, dims_[2]); 
            mpi_partition_.sends_(nidx).mpi_setup(neighb, tag, halos, comm); 
            tag = neighb * 10 + rank;
            mpi_partition_.recvs_(nidx) = MPIHaloKokkos <T> (dims_[0] - 2, dims_[2]); 
            mpi_partition_.recvs_(nidx).mpi_setup(neighb, tag, halos, comm); 
            nidx++;
        }
        if (j_s < mpi_dim_size) {
            neighb = j_s * mpi_dim_size + world_i;
            tag = rank * 10 + neighb;
            mpi_partition_.sends_(nidx) = MPIHaloKokkos <T> (dims_[0] - 2, dims_[2]); 
            mpi_partition_.sends_(nidx).mpi_setup(neighb, tag, halos, comm); 
            tag = neighb * 10 + rank;
            mpi_partition_.recvs_(nidx) = MPIHaloKokkos <T> (dims_[0] - 2, dims_[2]); 
            mpi_partition_.recvs_(nidx).mpi_setup(neighb, tag, halos, comm); 
            nidx++;
        }
        if (i_w >= 0) {
            neighb = world_j * mpi_dim_size + i_w;
            tag = rank * 10 + neighb;
            mpi_partition_.sends_(nidx) = MPIHaloKokkos <T> (dims_[0] - 2, dims_[2]); 
            mpi_partition_.sends_(nidx).mpi_setup(neighb, tag, halos, comm); 
            tag = neighb * 10 + rank;
            mpi_partition_.recvs_(nidx) = MPIHaloKokkos <T> (dims_[0] - 2, dims_[2]); 
            mpi_partition_.recvs_(nidx).mpi_setup(neighb, tag, halos, comm); 
            nidx++;
        }
        if (i_e < mpi_dim_size) {
            neighb = world_j * mpi_dim_size + i_e;
            tag = rank * 10 + neighb;
            mpi_partition_.sends_(nidx) = MPIHaloKokkos <T> (dims_[0] - 2, dims_[2]); 
            mpi_partition_.sends_(nidx).mpi_setup(neighb, tag, halos, comm); 
            tag = neighb * 10 + rank;
            mpi_partition_.recvs_(nidx) = MPIHaloKokkos <T> (dims_[0] - 2, dims_[2]); 
            mpi_partition_.recvs_(nidx).mpi_setup(neighb, tag, halos, comm); 
        }
    }
}

//void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::mpi_decomp(int world_size, int rank, int halos, MPI_Comm comm) {
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::mpi_decomp(MPIPartitionKokkos <T> partition) {
/*
    mpi_partition_ = partition;
    int neighbors = 0;
    int neighb, tag;
    int rank = mpi_partition_.mpi_rank_;
    int world_size = mpi_partition_.mpi_world_size_;
    int halos = mpi_partition_.mpi_halos_;
    MPI_Comm comm = mpi_partition_.mpi_comm_;
    if (order_ == 1) {
        int i_w = rank - 1;
        int i_e = rank + 1;
        mpi_partition_.send_w_ = MPIHaloKokkos <T> (1);
        if (i_w < 0) {
            neighb = -1;
        }
        else {
            neighb = i_w;
        }
        tag = rank * 10 + neighb;
        mpi_partition_.send_w_.mpi_setup(neighb, tag, halos, comm);
        tag = neighb * 10 + rank;
        mpi_partition_.recv_w_.mpi_setup(neighb, tag, halos, comm);
        mpi_partition_.send_e_ = MPIHaloKokkos <T> (1);
        if (i_e >= world_size) {
            neighb = -1;
        }
        else {
            neighb = i_e;
        }
        tag = rank * 10 + neighb;
        mpi_partition_.send_e_.mpi_setup(neighb, tag, halos, comm);
        tag = neighb * 10 + rank;
        mpi_partition_.recv_e_.mpi_setup(neighb, tag, halos, comm);

    }
    if (order_ == 2) {
        int mpi_dim_size = sqrt(world_size);
        int world_i = rank % mpi_dim_size;
        int world_j = rank / mpi_dim_size;
        int j_n = world_j - 1;
        int j_s = world_j + 1;
        int i_w = world_i - 1;
        int i_e = world_i + 1;

        // setup north
        mpi_partition_.send_n_ = MPIHaloKokkos <T> (dims_[0] - 2);
        mpi_partition_.recv_n_ = MPIHaloKokkos <T> (dims_[0] - 2);
        if (j_n < 0) {
            neighb = -1;
        }
        else {
            neighb = j_n * mpi_dim_size + world_i;
        }
        tag = rank * 10 + neighb;
        mpi_partition_.send_n_.mpi_setup(neighb, tag, halos, comm);
        tag = neighb * 10 + rank;
        mpi_partition_.recv_n_.mpi_setup(neighb, tag, halos, comm);
        // setup south
        mpi_partition_.send_s_ = MPIHaloKokkos <T> (dims_[0] - 2);
        mpi_partition_.recv_s_ = MPIHaloKokkos <T> (dims_[0] - 2);
        if (j_s >= mpi_dim_size) {
            neighb = -1;
        }
        else {
            neighb = j_s * mpi_dim_size + world_i;
        }
        tag = rank * 10 + neighb;
        mpi_partition_.send_s_.mpi_setup(neighb, tag, halos, comm);
        tag = neighb * 10 + rank;
        mpi_partition_.recv_s_.mpi_setup(neighb, tag, halos, comm);
        // setup west
        mpi_partition_.send_w_ = MPIHaloKokkos <T> (dims_[1] - 2);
        mpi_partition_.recv_w_ = MPIHaloKokkos <T> (dims_[1] - 2);
        if (i_w < 0) {
            neighb = -1;
        }
        else {
            neighb = world_j * mpi_dim_size + i_w;
        }
        tag = rank * 10 + neighb;
        mpi_partition_.send_w_.mpi_setup(neighb, tag, halos, comm);
        tag = neighb * 10 + rank;
        mpi_partition_.recv_w_.mpi_setup(neighb, tag, halos, comm);
        // setup east
        mpi_partition_.send_e_ = MPIHaloKokkos <T> (dims_[1] - 2);
        mpi_partition_.recv_e_ = MPIHaloKokkos <T> (dims_[1] - 2);
        if (i_e >= mpi_dim_size) {
            neighb = -1;
        }
        else {
            neighb = world_j * mpi_dim_size + i_e;
        }
        tag = rank * 10 + neighb;
        mpi_partition_.send_e_.mpi_setup(neighb, tag, halos, comm);
        tag = neighb * 10 + rank;
        mpi_partition_.recv_e_.mpi_setup(neighb, tag, halos, comm);
    }
    if (order_ == 3) {
        int mpi_dim_size = sqrt(world_size);
        int world_i = rank % mpi_dim_size;
        int world_j = rank / mpi_dim_size;
        int j_n = world_j - 1;
        int j_s = world_j + 1;
        int i_w = world_i - 1;
        int i_e = world_i + 1;

        // setup north
        mpi_partition_.send_n_ = MPIHaloKokkos <T> (dims_[0] - 2, dims_[2]);
        mpi_partition_.recv_n_ = MPIHaloKokkos <T> (dims_[0] - 2, dims_[2]);
        if (j_n < 0) {
            neighb = -1;
        }
        else {
            neighb = j_n * mpi_dim_size + world_i;
        }
        tag = rank * 10 + neighb;
        mpi_partition_.send_n_.mpi_setup(neighb, tag, halos, comm);
        tag = neighb * 10 + rank;
        mpi_partition_.recv_n_.mpi_setup(neighb, tag, halos, comm);
        // setup south
        mpi_partition_.send_s_ = MPIHaloKokkos <T> (dims_[0] - 2, dims_[2]);
        mpi_partition_.recv_s_ = MPIHaloKokkos <T> (dims_[0] - 2, dims_[2]);
        if (j_s >= mpi_dim_size) {
            neighb = -1;
        }
        else {
            neighb = j_s * mpi_dim_size + world_i;
        }
        tag = rank * 10 + neighb;
        mpi_partition_.send_s_.mpi_setup(neighb, tag, halos, comm);
        tag = neighb * 10 + rank;
        mpi_partition_.recv_s_.mpi_setup(neighb, tag, halos, comm);
        // setup west
        mpi_partition_.send_w_ = MPIHaloKokkos <T> (dims_[1] - 2, dims_[2]);
        mpi_partition_.recv_w_ = MPIHaloKokkos <T> (dims_[1] - 2, dims_[2]);
        if (i_w < 0) {
            neighb = -1;
        }
        else {
            neighb = world_j * mpi_dim_size + i_w;
        }
        tag = rank * 10 + neighb;
        mpi_partition_.send_w_.mpi_setup(neighb, tag, halos, comm);
        tag = neighb * 10 + rank;
        mpi_partition_.recv_w_.mpi_setup(neighb, tag, halos, comm);
        // setup east
        mpi_partition_.send_e_ = MPIHaloKokkos <T> (dims_[1] - 2, dims_[2]);
        mpi_partition_.recv_e_ = MPIHaloKokkos <T> (dims_[1] - 2, dims_[2]);
        if (i_e >= mpi_dim_size) {
            neighb = -1;
        }
        else {
            neighb = world_j * mpi_dim_size + i_e;
        }
        tag = rank * 10 + neighb;
        mpi_partition_.send_e_.mpi_setup(neighb, tag, halos, comm);
        tag = neighb * 10 + rank;
        mpi_partition_.recv_e_.mpi_setup(neighb, tag, halos, comm);
    }
*/
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::mpi_halo_update() {
    if (order_ == 1) {
        int halo = mpi_partition_.mpi_halos_; // just need one of them, all have same number of halos
        int halo_size_x = dims_[0] - 2; // remove outer halo ring
        if (mpi_partition_.send_w_.get_neighbor() >= 0) {
            // update halo with array values
            Kokkos::parallel_for("haloupdatewest", 1, KOKKOS_CLASS_LAMBDA(const int hh) {
                int halo_idx = halo + (halo - 1);
                mpi_partition_.send_w_(hh) = this_array_.d_view(halo_idx);
            }); 
            Kokkos::fence();
            mpi_partition_.send_w_.halo_isend();
            mpi_partition_.recv_w_.halo_irecv();
            // wait
            mpi_partition_.recv_w_.wait_recv();
            // update array with halo values
            Kokkos::parallel_for("haloupdatewest", 1, KOKKOS_CLASS_LAMBDA(const int hh) {
                int halo_idx = halo + (halo - 1);
                this_array_.d_view(halo_idx) = mpi_partition_.recv_w_(hh);
            }); 
            mpi_partition_.send_w_.wait_send();
        }
        if (mpi_partition_.send_e_.get_neighbor() >= 0) {
            // update halo with array values
            Kokkos::parallel_for("haloupdatewest", 1, KOKKOS_CLASS_LAMBDA(const int hh) {
                int halo_idx = halo_size_x;
                mpi_partition_.send_e_(hh) = this_array_.d_view(halo_idx);
            }); 
            Kokkos::fence();
            mpi_partition_.send_e_.halo_isend();
            mpi_partition_.recv_e_.halo_irecv();
            // wait
            mpi_partition_.recv_e_.wait_recv();
            // update array with halo values
            Kokkos::parallel_for("haloupdatewest", 1, KOKKOS_CLASS_LAMBDA(const int hh) {
                int halo_idx = halo_size_x;
                this_array_.d_view(halo_idx) = mpi_partition_.recv_e_(hh);
            }); 
            mpi_partition_.send_e_.wait_send();
        }
    }
    if (order_ == 2) {
        int halo = mpi_partition_.mpi_halos_; // just need one of them, all have same number of halos
        int halo_size_x = dims_[0] - 2; // remove outer halo ring
        int halo_size_y = dims_[1] - 2;
        if (mpi_partition_.send_n_.get_neighbor() >= 0) {
            // update halo with array values
            Kokkos::parallel_for("haloupdatenorth", halo_size_x, KOKKOS_CLASS_LAMBDA(const int hh) {
                int ii = halo + (halo - 1);
                int jj = halo + (halo - 1) + hh;
                mpi_partition_.send_n_(hh) = this_array_.d_view(ii*dims_[1] + jj); // row depending on # of halos
            }); 
            Kokkos::fence();
            mpi_partition_.send_n_.halo_isend();
            mpi_partition_.recv_n_.halo_irecv();
            // wait
            mpi_partition_.recv_n_.wait_recv();
            // update array with halo values
            Kokkos::parallel_for("haloupdatenorth2", halo_size_x, KOKKOS_CLASS_LAMBDA(const int hh) {
                int ii = halo + (halo - 1);
                int jj = halo + (halo - 1) + hh;
                this_array_.d_view(ii*dims_[1] + jj) = mpi_partition_.recv_n_(hh);
            }); 
            mpi_partition_.send_n_.wait_send();
        }
        if (mpi_partition_.send_s_.get_neighbor() >= 0) {
            // update halo with array values
            Kokkos::parallel_for("haloupdatesouth", halo_size_x, KOKKOS_CLASS_LAMBDA(const int hh) {
                int ii = halo_size_y;
                int jj = halo + (halo - 1) + hh;
                mpi_partition_.send_s_(hh) = this_array_.d_view(ii*dims_[1] + jj); // row depending on # of halos
            }); 
            Kokkos::fence();
            mpi_partition_.send_s_.halo_isend();
            mpi_partition_.recv_s_.halo_irecv();
            // wait
            mpi_partition_.recv_s_.wait_recv();
            // update array with halo values
            Kokkos::parallel_for("haloupdatesouth", halo_size_x, KOKKOS_CLASS_LAMBDA(const int hh) {
                int ii = halo_size_y;
                int jj = halo + (halo - 1) + hh;
                this_array_.d_view(ii*dims_[1] + jj) = mpi_partition_.recv_s_(hh);
            }); 
            mpi_partition_.send_s_.wait_send();
        }
        if (mpi_partition_.send_w_.get_neighbor() >= 0) {
            // update halo with array values
            Kokkos::parallel_for("haloupdatewest", halo_size_y, KOKKOS_CLASS_LAMBDA(const int hh) {
                int ii = halo + (halo - 1) + hh;
                int jj = halo + (halo - 1);
                mpi_partition_.send_w_(hh) = this_array_.d_view(ii*dims_[1] + jj); // column depending on # of halos
            }); 
            Kokkos::fence();
            mpi_partition_.send_w_.halo_isend();
            mpi_partition_.recv_w_.halo_irecv();
            // wait
            mpi_partition_.recv_w_.wait_recv();
            // update array with halo values
            Kokkos::parallel_for("haloupdatewest", halo_size_y, KOKKOS_CLASS_LAMBDA(const int hh) {
                int ii = halo + (halo - 1) + hh;
                int jj = halo + (halo - 1);
                this_array_.d_view(ii*dims_[1] + jj) = mpi_partition_.recv_w_(hh);
            }); 
            mpi_partition_.send_w_.wait_send();
        }
        if (mpi_partition_.send_e_.get_neighbor() >= 0) {
            // update halo with array values
            Kokkos::parallel_for("haloupdateeast", halo_size_y, KOKKOS_CLASS_LAMBDA(const int hh) {
                int ii = halo + (halo - 1) + hh;
                int jj = halo_size_x;
                mpi_partition_.send_e_(hh) = this_array_.d_view(ii*dims_[1] + jj); // column depending on # of halos
            }); 
            Kokkos::fence();
            mpi_partition_.send_e_.halo_isend();
            mpi_partition_.recv_e_.halo_irecv();
            // wait
            mpi_partition_.recv_e_.wait_recv();
            // update array with halo values
            Kokkos::parallel_for("haloupdateeast2", halo_size_y, KOKKOS_CLASS_LAMBDA(const int hh) {
                int ii = halo + (halo - 1) + hh;
                int jj = halo_size_x;
                this_array_.d_view(ii*dims_[1] + jj) = mpi_partition_.recv_e_(hh);
            }); 
            mpi_partition_.send_e_.wait_send();
        }
    }
    if (order_ == 3) {
        int halo = mpi_partition_.mpi_halos_; // just need one of them, all have same number of halos
        int halo_size_x = dims_[0] - 2; // remove outer halo ring
        int halo_size_y = dims_[1] - 2;
        int halo_size_z = dims_[2];
        if (mpi_partition_.send_n_.get_neighbor() >= 0) {
            // update halo with array values
            Kokkos::parallel_for("haloupdatenorth", policy2D({0,0},{halo_size_x, halo_size_z}), KOKKOS_CLASS_LAMBDA(const int hh, const int zz) {
                int ii = halo + (halo - 1);
                int jj = halo + (halo - 1) + hh;
                mpi_partition_.send_n_(hh, zz) = this_array_.d_view(ii*dims_[1]*dims_[2] + jj*dims_[2] + zz); // row depending on # of halos
            }); 
            Kokkos::fence();
            mpi_partition_.send_n_.halo_isend();
            mpi_partition_.recv_n_.halo_irecv();
            // wait
            mpi_partition_.recv_n_.wait_recv();
            // update array with halo values
            Kokkos::parallel_for("haloupdatenorth2", policy2D({0,0},{halo_size_x, halo_size_z}), KOKKOS_CLASS_LAMBDA(const int hh, const int zz) {
                int ii = halo + (halo - 1);
                int jj = halo + (halo - 1) + hh;
                this_array_.d_view(ii*dims_[1]*dims_[2] + jj*dims_[2] + zz) = mpi_partition_.recv_n_(hh, zz);
            }); 
            mpi_partition_.send_n_.wait_send();
        }
        if (mpi_partition_.send_s_.get_neighbor() >= 0) {
            // update halo with array values
            Kokkos::parallel_for("haloupdatesouth", policy2D({0,0},{halo_size_x, halo_size_z}), KOKKOS_CLASS_LAMBDA(const int hh, const int zz) {
                int ii = halo_size_y;
                int jj = halo + (halo - 1) + hh;
                mpi_partition_.send_s_(hh, zz) = this_array_.d_view(ii*dims_[1]*dims_[2] + jj*dims_[2] + zz); // row depending on # of halos
            }); 
            Kokkos::fence();
            mpi_partition_.send_s_.halo_isend();
            mpi_partition_.recv_s_.halo_irecv();
            // wait
            mpi_partition_.recv_s_.wait_recv();
            // update array with halo values
            Kokkos::parallel_for("haloupdatesouth2", policy2D({0,0},{halo_size_x, halo_size_z}), KOKKOS_CLASS_LAMBDA(const int hh, const int zz) {
                int ii = halo_size_y;
                int jj = halo + (halo - 1) + hh;
                this_array_.d_view(ii*dims_[1]*dims_[2] + jj*dims_[2] + zz) = mpi_partition_.recv_s_(hh, zz);
            }); 
            mpi_partition_.send_s_.wait_send();
        }
        if (mpi_partition_.send_w_.get_neighbor() >= 0) {
            // update halo with array values
            Kokkos::parallel_for("haloupdatewest", policy2D({0,0},{halo_size_y, halo_size_z}), KOKKOS_CLASS_LAMBDA(const int hh, const int zz) {
                int ii = halo + (halo - 1) + hh;
                int jj = halo + (halo - 1);
                mpi_partition_.send_w_(hh, zz) = this_array_.d_view(ii*dims_[1]*dims_[2] + jj*dims_[2] + zz); // column depending on # of halos
            }); 
            Kokkos::fence();
            mpi_partition_.send_w_.halo_isend();
            mpi_partition_.recv_w_.halo_irecv();
            // wait
            mpi_partition_.recv_w_.wait_recv();
            // update array with halo values
            Kokkos::parallel_for("haloupdatewest2", policy2D({0,0},{halo_size_y, halo_size_z}), KOKKOS_CLASS_LAMBDA(const int hh, const int zz) {
                int ii = halo + (halo - 1) + hh;
                int jj = halo + (halo - 1);
                this_array_.d_view(ii*dims_[1]*dims_[2] + jj*dims_[2] + zz) = mpi_partition_.recv_w_(hh, zz);
            }); 
            mpi_partition_.send_w_.wait_send();
        }
        if (mpi_partition_.send_e_.get_neighbor() >= 0) {
            // update halo with array values
            Kokkos::parallel_for("haloupdateeast", policy2D({0,0},{halo_size_y, halo_size_z}), KOKKOS_CLASS_LAMBDA(const int hh, const int zz) {
                int ii = halo + (halo - 1) + hh;
                int jj = halo_size_x;
                mpi_partition_.send_e_(hh, zz) = this_array_.d_view(ii*dims_[1]*dims_[2] + jj*dims_[2] + zz); // column depending on # of halos
            }); 
            Kokkos::fence();
            mpi_partition_.send_e_.halo_isend();
            mpi_partition_.recv_e_.halo_irecv();
            // wait
            mpi_partition_.recv_e_.wait_recv();
            // update array with halo values
            Kokkos::parallel_for("haloupdateeast2", policy2D({0,0},{halo_size_y, halo_size_z}), KOKKOS_CLASS_LAMBDA(const int hh, const int zz) {
                int ii = halo + (halo - 1) + hh;
                int jj = halo_size_x;
                this_array_.d_view(ii*dims_[1]*dims_[2] + jj*dims_[2] + zz) = mpi_partition_.recv_e_(hh, zz);
            }); 
            mpi_partition_.send_e_.wait_send();
        }
    }
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
MPICArrayKokkos<T,Layout,ExecSpace,MemoryTraits>::~MPICArrayKokkos() {}

////////////////////////////////////////////////////////////////////////////////
// End of MPICArrayKokkos
////////////////////////////////////////////////////////////////////////////////

} // end namespace

#endif // end if have MPI

#endif // MPI_TYPES_H

