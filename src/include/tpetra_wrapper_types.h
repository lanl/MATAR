#ifndef TPETRA_WRAPPER_TYPES_H
#define TPETRA_WRAPPER_TYPES_H
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
#ifdef TRILINOS_INTERFACE
#include "host_types.h"
#include "kokkos_types.h"
#include <typeinfo>
#include <memory> // for shared_ptr
#ifdef HAVE_MPI
#include <mpi.h>
#include "partition_map.h"
#include "communication_plan.h"
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_Tuple.hpp>
#include <Teuchos_VerboseObject.hpp>

#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Kokkos_Core.hpp>
#include "Tpetra_Details_DefaultTypes.hpp"
#include "Tpetra_Import.hpp"

namespace mtr
{

/////////////////////////
// TpetraPartitionMap:  Container storing global indices corresponding to local indices that belong on this process/rank as well as comms related data/functions.
/////////////////////////
template <typename T = long long int, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class TpetraPartitionMap {

    // this is manage
    using TArray1D = Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits>;
    using TArray1D_host = Kokkos::View <T*, Layout, HostSpace, MemoryTraits>;
    using TArray1D_dev = Kokkos::View <T*, Layout, ExecSpace, MemoryTraits>;

    // Trilinos type definitions
    typedef Tpetra::Map<>::local_ordinal_type LO;
    typedef Tpetra::Map<>::global_ordinal_type GO;

    typedef Tpetra::MultiVector<real_t, LO, GO> MV;

    typedef Kokkos::ViewTraits<LO*, Kokkos::LayoutLeft, void, void>::size_type SizeType;
    typedef Tpetra::Details::DefaultTypes::node_type node_type;
    using traits = Kokkos::ViewTraits<LO*, Kokkos::LayoutLeft, void, void>;

    using array_layout    = typename traits::array_layout;
    using execution_space = typename traits::execution_space;
    using device_type     = typename traits::device_type;
    using memory_traits   = typename traits::memory_traits;
    using global_size_t   = Tpetra::global_size_t;
    
protected:
    size_t length_;
    size_t order_;  // tensor order (rank)
    MPI_Datatype mpi_datatype_;
    TArray1D this_array_;
    
    void set_mpi_type();

public:

    //pointer to wrapped Tpetra map
    Teuchos::RCP<Tpetra::Map<LO, GO, node_type>> tpetra_map; // map of node indices

    int num_global_;

    // Data member to access host view
    ViewCArray <T> host;

    //MPI communicator
    MPI_Comm mpi_comm_;

    TpetraPartitionMap();

    //Copy Constructor
    TpetraPartitionMap(const TpetraPartitionMap<T, Layout, ExecSpace,MemoryTraits> &temp){
        *this = temp;
    }
     
    //TpetraPartitionMap(size_t global_length, MPI_Comm mpi_comm_ = MPI_COMM_WORLD, const std::string& tag_string = DEFAULTSTRINGARRAY);

    TpetraPartitionMap(DCArrayKokkos<T,Layout,ExecSpace,MemoryTraits> &indices, MPI_Comm mpi_comm_ = MPI_COMM_WORLD);

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i) const;

    KOKKOS_INLINE_FUNCTION
    TpetraPartitionMap& operator=(const TpetraPartitionMap& temp);

    // GPU Method
    // Method that returns size
    KOKKOS_INLINE_FUNCTION
    size_t size() const;

    // Host Method
    // Method that returns size
    KOKKOS_INLINE_FUNCTION
    size_t extent() const;
    
    int getLocalIndex(int global_index) const;

    int getGlobalIndex(int local_index) const;

    bool isProcessGlobalIndex(int global_index) const;

    bool isProcessLocalIndex(int local_index) const;

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
    virtual KOKKOS_INLINE_FUNCTION
    ~TpetraPartitionMap ();
}; // End of TpetraPartitionMap


// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::TpetraPartitionMap() {
    length_ = 0;
}

// Constructor for contiguous index decomposition
// template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
// TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::TpetraPartitionMap(size_t global_length, MPI_Comm mpi_comm_, const std::string& tag_string) {
    
//     Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<GO>(mpi_comm_));
//     tpetra_map = Teuchos::rcp(new Tpetra::Map<LO, GO, node_type>((int) global_length, 0, teuchos_comm));
//     num_global_ = global_length;
//     TArray1D_host indices_host = tpetra_map->getMyGlobalIndices();
//     length_ = indices_host.size();
//     this_array_ = TArray1D(tag_string, length_);
//     // Create host ViewCArray
//     //host = ViewCArray <T> (this_array_.h_view.data(), dim0);
//     set_mpi_type();
// }

// Constructor to pass matar dual view of indices
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::TpetraPartitionMap(DCArrayKokkos<T,Layout,ExecSpace,MemoryTraits> &indices, MPI_Comm mpi_comm_) {
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
    this_array_ = static_cast<DCArrayKokkos<GO,Layout,ExecSpace,MemoryTraits>>(indices).get_kokkos_dual_view();
    // Create host ViewCArray
    host = ViewCArray <T> (this_array_.h_view.data(), indices.size());
    set_mpi_type();
    tpetra_map = Teuchos::rcp(new Tpetra::Map<LO, GO, node_type>(Teuchos::OrdinalTraits<GO>::invalid(), this_array_.d_view, 0, teuchos_comm));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::set_mpi_type() {
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
        printf("Your entered TpetraPartitionMap type is not a supported type for MPI communications and is being set to int\n");
        mpi_datatype_ = MPI_INT;
    }
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i) const {
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in TpetraPartitionMap 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraPartitionMap 1D!");
    return this_array_.d_view(i);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>& TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::operator= (const TpetraPartitionMap& temp) {
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        length_ = temp.length_;
        this_array_ = temp.this_array_;
        host = temp.host;
        mpi_datatype_ = temp.mpi_datatype_;
        tpetra_map = temp.tpetra_map;
        mpi_comm_ = temp.mpi_comm_;
        num_global_= temp.num_global_;
    }
    
    return *this;
}

// Return size
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::size() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::extent() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::device_pointer() const {
    return this_array_.d_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::host_pointer() const {
    return this_array_.h_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits> TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_dual_view() const {
  return this_array_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::update_host() {

    this_array_.template modify<typename TArray1D::execution_space>();
    this_array_.template sync<typename TArray1D::host_mirror_space>();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::update_device() {

    this_array_.template modify<typename TArray1D::host_mirror_space>();
    this_array_.template sync<typename TArray1D::execution_space>();
}

// Return local index (on this process/rank) corresponding to the input global index
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
int TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::getLocalIndex(int global_index) const {
    int local_index = tpetra_map->getLocalElement(global_index);
    return local_index;
}

// Return global index corresponding to the input local (on this process/rank) index
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
int TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::getGlobalIndex(int local_index) const {
    int global_index = tpetra_map->getGlobalElement(local_index);
    return global_index;
}

// Return global index corresponding to the input local (on this process/rank) index
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
bool TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::isProcessGlobalIndex(int global_index) const {
    bool belongs = tpetra_map->isNodeGlobalElement(global_index);
    return belongs;
}

// Return global index corresponding to the input local (on this process/rank) index
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
bool TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::isProcessLocalIndex(int local_index) const {
    bool belongs = tpetra_map->isNodeGlobalElement(local_index);
    return belongs;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::~TpetraPartitionMap() {}

////////////////////////////////////////////////////////////////////////////////
// End of TpetraPartitionMap
////////////////////////////////////////////////////////////////////////////////

/////////////////////////
// TpetraMVArray:  Dual type for managing distributed data on both CPU and GPU with a partition map.
/////////////////////////
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class TpetraMVArray {

    // this is manage
    using  TArray1D = Kokkos::DualView <T**, Layout, ExecSpace, MemoryTraits>;

    size_t dims_[7];
    size_t length_;
    size_t order_;  // tensor order (rank)
    int mpi_recv_rank_;
    int mpi_tag_;
    MPI_Comm mpi_comm_;
    MPI_Status mpi_status_;
    MPI_Datatype mpi_datatype_;
    MPI_Request mpi_request_;
    TArray1D this_array_;
    
    // Trilinos type definitions
    typedef Tpetra::Map<>::local_ordinal_type LO;
    typedef Tpetra::Map<>::global_ordinal_type GO;

    typedef Tpetra::MultiVector<real_t, LO, GO> MV;

    typedef Kokkos::ViewTraits<LO*, Kokkos::LayoutLeft, void, void>::size_type SizeType;
    typedef Tpetra::Details::DefaultTypes::node_type node_type;
    using traits = Kokkos::ViewTraits<LO*, Kokkos::LayoutLeft, void, void>;

    using array_layout    = typename traits::array_layout;
    using execution_space = typename traits::execution_space;
    using device_type     = typename traits::device_type;
    using memory_traits   = typename traits::memory_traits;
    using global_size_t   = Tpetra::global_size_t;

    typedef Kokkos::View<real_t*, Kokkos::LayoutRight, device_type, memory_traits> values_array;
    typedef Kokkos::View<GO*, array_layout, device_type, memory_traits> global_indices_array;
    typedef Kokkos::View<LO*, array_layout, device_type, memory_traits> indices_array;
    typedef MV::dual_view_type::t_dev vec_array;
    typedef MV::dual_view_type::t_host host_vec_array;
    typedef Kokkos::View<const real_t**, array_layout, HostSpace, memory_traits> const_host_vec_array;
    typedef Kokkos::View<const real_t**, array_layout, device_type, memory_traits> const_vec_array;
    typedef Kokkos::View<const int**, array_layout, HostSpace, memory_traits> const_host_ivec_array;
    typedef Kokkos::View<int**, array_layout, HostSpace, memory_traits> host_ivec_array;
    typedef MV::dual_view_type dual_vec_array;
    

public:
    
    Teuchos::RCP<Tpetra::Map<LO, GO, node_type>> pmap;
    Teuchos::RCP<Tpetra::Map<LO, GO, node_type>> comm_pmap;
    Teuchos::RCP<MV>       tpetra_vector;
    Teuchos::RCP<MV>       tpetra_sub_vector; //for owned comms situations
    Teuchos::RCP<Tpetra::Import<LO, GO>> importer; // tpetra comms object

    //data for arrays that own both shared and local data and aren't intended to communicate with another MATAR type
    //This is simplifying for cases such as a local + ghost storage vector where you need to update the ghost entries
    bool own_comms; //This Mapped MPI Array contains its own communication plan; just call array_comms()
    
    void set_mpi_type();

    TpetraMVArray();
    
    //Copy Constructor
    TpetraMVArray(const TpetraMVArray<T, Layout, ExecSpace,MemoryTraits> &temp){
        *this = temp;
    }

    //Tpetra type for 1D case(still allocates dim0 by 1 using **T)
    TpetraMVArray(size_t dim0, TpetraPartitionMap<long long int,Layout,ExecSpace,MemoryTraits> input_pmap, const std::string& tag_string = DEFAULTSTRINGARRAY);

    //Tpetra type only goes up to 2D access
    TpetraMVArray(size_t dim0, size_t dim1, TpetraPartitionMap<long long int,Layout,ExecSpace,MemoryTraits> input_pmap, const std::string& tag_string = DEFAULTSTRINGARRAY);

    //Tpetra type for 1D case(still allocates dim0 by 1 using **T); this constructor takes an RCP pointer to a Tpetra Map directly
    TpetraMVArray(size_t dim0, Teuchos::RCP<Tpetra::Map<LO, GO, node_type>> input_pmap, const std::string& tag_string = DEFAULTSTRINGARRAY);

    //Tpetra type only goes up to 2D access; this constructor takes an RCP pointer to a Tpetra Map directly
    TpetraMVArray(size_t dim0, size_t dim1, Teuchos::RCP<Tpetra::Map<LO, GO, node_type>> input_pmap, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    TpetraMVArray(Teuchos::RCP<MV> input_tpetra_vector, const std::string& tag_string = DEFAULTSTRINGARRAY);

    void own_comm_setup(Teuchos::RCP<Tpetra::Map<LO, GO, node_type>> other_pmap); //only call if the map in the arg is a uniquely owned submap of the arrays map
    
    void own_comm_setup(TpetraPartitionMap<long long int,Layout,ExecSpace,MemoryTraits> other_pmap); //only call if the map in the arg is a uniquely owned submap of the arrays map

    void perform_comms();

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j) const;
    
    KOKKOS_INLINE_FUNCTION
    TpetraMVArray& operator=(const TpetraMVArray& temp);

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

    // Deconstructor
    virtual KOKKOS_INLINE_FUNCTION
    ~TpetraMVArray ();
}; // End of TpetraMVArray


// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::TpetraMVArray(): pmap(NULL){
    length_ = order_ = 0;
    for (int i = 0; i < 7; i++) {
        dims_[i] = 0;
    }
}

// Overloaded 1D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::TpetraMVArray(size_t dim0,
                                                             TpetraPartitionMap<long long int,Layout,ExecSpace,MemoryTraits> input_pmap,
                                                             const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = 1;
    order_ = 1;
    length_ = dim0;
    pmap = input_pmap.tpetra_map;
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dim0, 1);
    tpetra_vector   = Teuchos::rcp(new MV(pmap, this_array_));
}

// Overloaded 2D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::TpetraMVArray(size_t dim0, size_t dim1,
                                                                            TpetraPartitionMap<long long int,Layout,ExecSpace,MemoryTraits> input_pmap,
                                                                            const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    pmap = input_pmap.tpetra_map;
    order_ = 2;
    length_ = (dim0 * dim1);
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dim0, dim1);
    tpetra_vector   = Teuchos::rcp(new MV(pmap, this_array_));
}

// Overloaded 1D constructor taking an RPC pointer to a Tpetra Map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::TpetraMVArray(size_t dim0,
                                                             Teuchos::RCP<Tpetra::Map<LO, GO, node_type>> input_pmap,
                                                             const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = 1;
    order_ = 1;
    length_ = dim0;
    pmap = input_pmap;
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dim0, 1);
    tpetra_vector   = Teuchos::rcp(new MV(pmap, this_array_));
}

// Overloaded 2D constructor taking an RPC pointer to a Tpetra Map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::TpetraMVArray(size_t dim0, size_t dim1,
                                                                            Teuchos::RCP<Tpetra::Map<LO, GO, node_type>> input_pmap,
                                                                            const std::string& tag_string) {
    
    dims_[0] = dim0;
    dims_[1] = dim1;
    pmap = input_pmap;
    order_ = 2;
    length_ = (dim0 * dim1);
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dim0, dim1);
    tpetra_vector   = Teuchos::rcp(new MV(pmap, this_array_));
}

// Tpetra vector argument constructor: CURRENTLY DOESN'T WORK SINCE WE CANT GET DUAL VIEW FROM THE MULTIVECTOR
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::TpetraMVArray(Teuchos::RCP<MV> input_tpetra_vector, const std::string& tag_string){
    
    tpetra_vector   = input_tpetra_vector;
    pmap = input_tpetra_vector->getMap(); 
    //this_array_ = tpetra_vector->getWrappedDualView();
    dims_[0] = tpetra_vector->getMap()->getLocalNumElements();
    dims_[1] = tpetra_vector->getNumVectors();

    if(dims_[1]==1){
        order_ = 1;
    }
    else{
        order_ = 2;
    }
    length_ = (dims_[0] * dims_[1]);
    set_mpi_type();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::set_mpi_type() {
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
        printf("Your entered TpetraMVArray type is not a supported type for MPI communications and is being set to int\n");
        mpi_datatype_ = MPI_INT;
    }
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i) const {
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in TpetraMVArray 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraMVArray 1D!");
    return this_array_.d_view(i,0);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j) const {
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in TpetraMVArray 2D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraMVArray 2D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraMVArray 2D!");
    return this_array_.d_view(i,j);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>& TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::operator= (const TpetraMVArray& temp) {
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for

        order_ = temp.order_;
        length_ = temp.length_;
        this_array_ = temp.this_array_;
        mpi_recv_rank_ = temp.mpi_recv_rank_;
        mpi_tag_ = temp.mpi_tag_;
        mpi_comm_ = temp.mpi_comm_;
        mpi_status_ = temp.mpi_status_;
        mpi_datatype_ = temp.mpi_datatype_;
        mpi_request_ = temp.mpi_request_;
        tpetra_vector = temp.tpetra_vector;
        tpetra_sub_vector = temp.tpetra_sub_vector;
        pmap = temp.pmap;
        comm_pmap = temp.comm_pmap;
        importer = temp.importer;
        own_comms = temp.own_comms;
    }
    
    return *this;
}

// Return size
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::size() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::extent() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::dims(size_t i) const {
    assert(i < order_ && "TpetraMVArray order (rank) does not match constructor, dim[i] does not exist!");
    assert(i >= 0 && dims_[i]>0 && "Access to TpetraMVArray dims is out of bounds!");
    return dims_[i];
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::order() const {
    return order_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::device_pointer() const {
    return this_array_.d_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::host_pointer() const {
    return this_array_.h_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
Kokkos::DualView <T**, Layout, ExecSpace, MemoryTraits> TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_dual_view() const {
  return this_array_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::update_host() {

    this_array_.template modify<typename TArray1D::execution_space>();
    this_array_.template sync<typename TArray1D::host_mirror_space>();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::update_device() {

    this_array_.template modify<typename TArray1D::host_mirror_space>();
    this_array_.template sync<typename TArray1D::execution_space>();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::own_comm_setup(TpetraPartitionMap<long long int,Layout,ExecSpace,MemoryTraits> other_pmap) {
    own_comms = true;
    comm_pmap = other_pmap.tpetra_map;
    tpetra_sub_vector = Teuchos::rcp(new MV(*tpetra_vector, comm_pmap));
    importer = Teuchos::rcp(new Tpetra::Import<LO, GO>(comm_pmap, pmap));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::own_comm_setup(Teuchos::RCP<Tpetra::Map<LO, GO, node_type>> other_pmap) {
    own_comms = true;
    comm_pmap = other_pmap;
    tpetra_sub_vector = Teuchos::rcp(new MV(*tpetra_vector, comm_pmap));
    importer = Teuchos::rcp(new Tpetra::Import<LO, GO>(comm_pmap, pmap));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::perform_comms() {
    tpetra_vector->doImport(*tpetra_sub_vector, *importer, Tpetra::INSERT);
}

//MPI_Barrier wrapper
//template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
//void TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::barrier(MPI_Comm comm) {
//    MPI_Barrier(comm); 
//}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::~TpetraMVArray() {}

////////////////////////////////////////////////////////////////////////////////
// End of TpetraMVArray
////////////////////////////////////////////////////////////////////////////////

} // end namespace

#endif // end if have MPI
#endif // end if TRILINOS_INTERFACE

#endif // TPETRA_WRAPPER_TYPES_H

