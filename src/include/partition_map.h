#ifndef PARTITION_MAP_H
#define PARTITION_MAP_H
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
// PartitionMap:  Container storing local indices that belong on this process/rank as well as comms related data/functions.
/////////////////////////
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class PartitionMap {

    // this is manage
    using TArray1D = Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits>;
    
private:
    size_t length_;
    size_t order_;  // tensor order (rank)
    MPI_Datatype mpi_datatype_;
    TArray1D this_array_;
    
    void set_mpi_type();

public:
    // Data member to access host view
    ViewCArray <T> host;

    PartitionMap();
    
    PartitionMap(size_t length, const std::string& tag_string = DEFAULTSTRINGARRAY);

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i) const;

    KOKKOS_INLINE_FUNCTION
    PartitionMap& operator=(const PartitionMap& temp);

    // GPU Method
    // Method that returns size
    KOKKOS_INLINE_FUNCTION
    size_t size() const;

    // Host Method
    // Method that returns size
    KOKKOS_INLINE_FUNCTION
    size_t extent() const;

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

    // Deconstructor
    KOKKOS_INLINE_FUNCTION
    ~PartitionMap ();
}; // End of PartitionMap


// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
PartitionMap<T,Layout,ExecSpace,MemoryTraits>::PartitionMap() {
    length_ = 0;
}

// Overloaded 1D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
PartitionMap<T,Layout,ExecSpace,MemoryTraits>::PartitionMap(size_t dim0, const std::string& tag_string) {
    
    length_ = dim0;
    this_array_ = TArray1D(tag_string, length_);
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), dim0);
    set_mpi_type();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void PartitionMap<T,Layout,ExecSpace,MemoryTraits>::set_mpi_type() {
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
        printf("Your entered PartitionMap type is not a supported type for MPI communications and is being set to int\n");
        mpi_datatype_ = MPI_INT;
    }
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& PartitionMap<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i) const {
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in PartitionMap 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in PartitionMap 1D!");
    return this_array_.d_view(i);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
PartitionMap<T,Layout,ExecSpace,MemoryTraits>& PartitionMap<T,Layout,ExecSpace,MemoryTraits>::operator= (const PartitionMap& temp) {
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        length_ = temp.length_;
        this_array_ = temp.this_array_;
        host = temp.host;
        mpi_datatype_ = temp.mpi_datatype_;
    }
    
    return *this;
}

// Return size
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t PartitionMap<T,Layout,ExecSpace,MemoryTraits>::size() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t PartitionMap<T,Layout,ExecSpace,MemoryTraits>::extent() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* PartitionMap<T,Layout,ExecSpace,MemoryTraits>::device_pointer() const {
    return this_array_.d_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* PartitionMap<T,Layout,ExecSpace,MemoryTraits>::host_pointer() const {
    return this_array_.h_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits> PartitionMap<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_dual_view() const {
  return this_array_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void PartitionMap<T,Layout,ExecSpace,MemoryTraits>::update_host() {

    this_array_.template modify<typename TArray1D::execution_space>();
    this_array_.template sync<typename TArray1D::host_mirror_space>();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void PartitionMap<T,Layout,ExecSpace,MemoryTraits>::update_device() {

    this_array_.template modify<typename TArray1D::host_mirror_space>();
    this_array_.template sync<typename TArray1D::execution_space>();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
PartitionMap<T,Layout,ExecSpace,MemoryTraits>::~PartitionMap() {}

////////////////////////////////////////////////////////////////////////////////
// End of PartitionMap
////////////////////////////////////////////////////////////////////////////////

} // end namespace

#endif // end if have MPI

#endif // PARTITION_MAP_H

