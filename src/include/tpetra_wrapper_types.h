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

// Repartition Package
#include <Zoltan2_XpetraMultiVectorAdapter.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_PartitioningSolution.hpp>
#include <Zoltan2_InputTraits.hpp>

// Trilinos type definitions
typedef Tpetra::Map<>::local_ordinal_type tpetra_LO;
typedef Tpetra::Map<>::global_ordinal_type tpetra_GO;

typedef Kokkos::ViewTraits<tpetra_LO*, Kokkos::LayoutLeft, void, void>::size_type tpetra_SizeType;
typedef Tpetra::Details::DefaultTypes::node_type tpetra_node_type;
using tpetra_traits = Kokkos::ViewTraits<tpetra_LO*, Kokkos::LayoutLeft, void, void>;

using tpetra_array_layout    = typename tpetra_traits::array_layout;
using tpetra_execution_space = typename tpetra_traits::execution_space;
using tpetra_device_type     = typename tpetra_traits::device_type;
using tpetra_memory_traits   = typename tpetra_traits::memory_traits;
using tpetra_global_size_t   = Tpetra::global_size_t;

namespace mtr
{

/////////////////////////
// TpetraPartitionMap:  Container storing global indices corresponding to local indices that belong on this process/rank as well as comms related data/functions.
/////////////////////////
template <typename T = long long int, typename Layout = tpetra_array_layout, typename ExecSpace = tpetra_execution_space, typename MemoryTraits = tpetra_memory_traits>
class TpetraPartitionMap {

    // this is manage
    using TArray1D = Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits>;
    using TArray1D_host = Kokkos::View <const T*, Layout, HostSpace, MemoryTraits>;
    using TArray1D_dev = Kokkos::View <const T*, Layout, ExecSpace, MemoryTraits>;
    
    
protected:
    size_t length_;
    MPI_Datatype mpi_datatype_;
    TArray1D_host host;
    TArray1D_dev device;
    
    void set_mpi_type();

public:

    //pointer to wrapped Tpetra map
    Teuchos::RCP<Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> tpetra_map; // map of node indices

    int num_global_;

    //MPI communicator
    MPI_Comm mpi_comm_;

    TpetraPartitionMap();

    //Copy Constructor
    TpetraPartitionMap(const TpetraPartitionMap<T, Layout, ExecSpace,MemoryTraits> &temp){
        *this = temp;
    }
     
    TpetraPartitionMap(size_t global_length, MPI_Comm mpi_comm = MPI_COMM_WORLD, const std::string& tag_string = DEFAULTSTRINGARRAY);

    TpetraPartitionMap(DCArrayKokkos<T> &indices, MPI_Comm mpi_comm = MPI_COMM_WORLD, const std::string& tag_string = DEFAULTSTRINGARRAY);

    TpetraPartitionMap(Teuchos::RCP<Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_tpetra_map, const std::string& tag_string = DEFAULTSTRINGARRAY);

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
    
    KOKKOS_INLINE_FUNCTION
    int getLocalIndex(int global_index) const;
    
    KOKKOS_INLINE_FUNCTION
    long long int getGlobalIndex(int local_index) const;
    
    KOKKOS_INLINE_FUNCTION
    bool isProcessGlobalIndex(int global_index) const;
    
    KOKKOS_INLINE_FUNCTION
    bool isProcessLocalIndex(int local_index) const;

    // Method returns the raw device pointer of the Kokkos DualView
    KOKKOS_INLINE_FUNCTION
    T* device_pointer() const;

    // Method returns the raw host pointer of the Kokkos DualView
    T* host_pointer() const;

    void print() const;

    // // Method returns kokkos dual view
    // KOKKOS_INLINE_FUNCTION
    // TArray1D get_kokkos_dual_view() const;

    // // Method that update host view
    // void update_host();

    // // Method that update device view
    // void update_device();

    // Deconstructor
    virtual KOKKOS_INLINE_FUNCTION
    ~TpetraPartitionMap ();
}; // End of TpetraPartitionMap


// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::TpetraPartitionMap() {
    length_ = 0;
}

//Constructor for contiguous index decomposition
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::TpetraPartitionMap(size_t global_length, MPI_Comm mpi_comm, const std::string& tag_string) {
    mpi_comm_ = mpi_comm;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm));
    tpetra_map = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) global_length, 0, teuchos_comm));
    num_global_ = global_length;
    TArray1D_host host = tpetra_map->getMyGlobalIndices();
    TArray1D_dev device = tpetra_map->getMyGlobalIndicesDevice();
    length_ = host.size();
    set_mpi_type();
}

// Constructor to pass matar dual view of indices
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::TpetraPartitionMap(DCArrayKokkos<T> &indices, MPI_Comm mpi_comm, const std::string& tag_string) {
    mpi_comm_ = mpi_comm;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm));
    tpetra_map = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>(Teuchos::OrdinalTraits<tpetra_GO>::invalid(), indices.get_kokkos_dual_view().d_view, 0, teuchos_comm));
    TArray1D_host host = tpetra_map->getMyGlobalIndices();
    TArray1D_dev device = tpetra_map->getMyGlobalIndicesDevice();
    length_ = host.size();
    num_global_ = tpetra_map->getGlobalNumElements();
    set_mpi_type();
}

// Constructor to pass an existing Tpetra map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::TpetraPartitionMap(Teuchos::RCP<Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_tpetra_map, const std::string& tag_string) {
    tpetra_map = input_tpetra_map;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = tpetra_map->getComm();
    mpi_comm_ = getRawMpiComm(*teuchos_comm);
    TArray1D_host host = input_tpetra_map->getMyGlobalIndices();
    TArray1D_dev device = input_tpetra_map->getMyGlobalIndicesDevice();
    length_ = host.size();
    num_global_ = tpetra_map->getGlobalNumElements();
    set_mpi_type();
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
    return device(i);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>& TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::operator= (const TpetraPartitionMap& temp) {
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        length_ = temp.length_;
        host = temp.host;
        device = temp.device;
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
    return device.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T* TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::host_pointer() const {
    return host.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::print() const {
        std::ostream &out = std::cout;
        Teuchos::RCP<Teuchos::FancyOStream> fos;
        fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(out));
        tpetra_map->describe(*fos,Teuchos::VERB_EXTREME);
}

// Return local index (on this process/rank) corresponding to the input global index
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
int TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::getLocalIndex(int global_index) const {
    int local_index = tpetra_map->getLocalElement(global_index);
    return local_index;
}

// Return global index corresponding to the input local (on this process/rank) index
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
long long int TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::getGlobalIndex(int local_index) const {
    int global_index = tpetra_map->getGlobalElement(local_index);
    return global_index;
}

// Return global index corresponding to the input local (on this process/rank) index
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
bool TpetraPartitionMap<T,Layout,ExecSpace,MemoryTraits>::isProcessGlobalIndex(int global_index) const {
    bool belongs = tpetra_map->isNodeGlobalElement(global_index);
    return belongs;
}

// Return global index corresponding to the input local (on this process/rank) index
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
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
template <typename T, typename Layout = tpetra_array_layout, typename ExecSpace = tpetra_execution_space, typename MemoryTraits = tpetra_memory_traits>
class TpetraMVArray {

    // this is manage
    using  TArray1D = Kokkos::DualView <T**, Layout, ExecSpace, MemoryTraits>;
    using  TArray1D_host = Kokkos::View <T**, Layout, ExecSpace, MemoryTraits>;

    size_t dims_[2];
    size_t global_dim1_;
    size_t submap_size_;
    size_t length_;
    size_t order_;  // tensor order (rank)
    MPI_Comm mpi_comm_;
    MPI_Datatype mpi_datatype_;
    TArray1D this_array_;
    
    // Trilinos type definitions
    typedef Tpetra::MultiVector<real_t, tpetra_LO, tpetra_GO> MV;

    typedef Kokkos::View<real_t*, Kokkos::LayoutRight, tpetra_device_type, tpetra_memory_traits> values_array;
    typedef Kokkos::View<tpetra_GO*, tpetra_array_layout, tpetra_device_type, tpetra_memory_traits> global_indices_array;
    typedef Kokkos::View<tpetra_LO*, tpetra_array_layout, tpetra_device_type, tpetra_memory_traits> indices_array;
    typedef MV::dual_view_type::t_dev vec_array;
    typedef MV::dual_view_type::t_host host_vec_array;
    typedef Kokkos::View<const real_t**, tpetra_array_layout, HostSpace, tpetra_memory_traits> const_host_vec_array;
    typedef Kokkos::View<const real_t**, tpetra_array_layout, tpetra_device_type, tpetra_memory_traits> const_vec_array;
    typedef Kokkos::View<const int**, tpetra_array_layout, HostSpace, tpetra_memory_traits> const_host_ivec_array;
    typedef Kokkos::View<int**, tpetra_array_layout, HostSpace, tpetra_memory_traits> host_ivec_array;
    typedef MV::dual_view_type dual_vec_array;
    Teuchos::RCP<Tpetra::Import<tpetra_LO, tpetra_GO>> importer; // tpetra comms object
    

public:
    
    //data for arrays that own both shared and local data and aren't intended to communicate with another MATAR type
    //This is simplifying for cases such as a local + ghost storage vector where you need to update the ghost entries
    bool own_comms; //This Mapped MPI Array contains its own communication plan; just call array_comms()
    
    void set_mpi_type();
    TpetraPartitionMap<tpetra_GO, Layout, ExecSpace, MemoryTraits> pmap;
    TpetraPartitionMap<tpetra_GO, Layout, ExecSpace, MemoryTraits> comm_pmap;
    Teuchos::RCP<Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> tpetra_pmap;
    Teuchos::RCP<Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> tpetra_comm_pmap;
    Teuchos::RCP<MV>       tpetra_vector;
    Teuchos::RCP<MV>       tpetra_sub_vector; //for owned comms situations

    TpetraMVArray();
    
    //Copy Constructor
    TpetraMVArray(const TpetraMVArray<T, Layout, ExecSpace,MemoryTraits> &temp){
        *this = temp;
    }

    //Tpetra type for 1D case(still allocates dim0 by 1 using **T); partitions with unique indices per process
    TpetraMVArray(size_t dim0, const std::string& tag_string = DEFAULTSTRINGARRAY, MPI_Comm mpi_comm = MPI_COMM_WORLD);

    //Tpetra type only goes up to 2D access; partitions along rows with unique indices per process
    TpetraMVArray(size_t dim0, size_t dim1, const std::string& tag_string = DEFAULTSTRINGARRAY, MPI_Comm mpi_comm = MPI_COMM_WORLD);

    //Tpetra type for 1D case with a partition map passed in
    TpetraMVArray(TpetraPartitionMap<long long int,Layout,ExecSpace,MemoryTraits> input_pmap, const std::string& tag_string = DEFAULTSTRINGARRAY);

    //2D Tpetra type with a partition map passed in
    TpetraMVArray(TpetraPartitionMap<long long int,Layout,ExecSpace,MemoryTraits> input_pmap, size_t dim1, const std::string& tag_string = DEFAULTSTRINGARRAY);

    //Tpetra type for 1D case(still allocates dim0 by 1 using **T); this constructor takes an RCP pointer to a Tpetra Map directly
    TpetraMVArray(Teuchos::RCP<Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap, const std::string& tag_string = DEFAULTSTRINGARRAY);

    //Tpetra type only goes up to 2D access; this constructor takes an RCP pointer to a Tpetra Map directly
    TpetraMVArray(Teuchos::RCP<Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap, size_t dim1, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    TpetraMVArray(Teuchos::RCP<MV> input_tpetra_vector, const std::string& tag_string = DEFAULTSTRINGARRAY);

    void own_comm_setup(Teuchos::RCP<Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> other_pmap); //only call if the map in the arg is a uniquely owned submap of the arrays map
    
    void own_comm_setup(TpetraPartitionMap<long long int,Layout,ExecSpace,MemoryTraits> other_pmap); //only call if the map in the arg is a uniquely owned submap of the arrays map

    void perform_comms();

    void repartition_vector(); //repartitions this vector using the zoltan2 multijagged algorithm on its data

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j) const;

    T& host(size_t i) const;

    T& host(size_t i, size_t j) const;
    
    KOKKOS_INLINE_FUNCTION
    TpetraMVArray& operator=(const TpetraMVArray& temp);

    // GPU Method
    // Method that returns size
    KOKKOS_INLINE_FUNCTION
    size_t size() const;

    size_t submap_size() const;

    KOKKOS_INLINE_FUNCTION
    long long int getSubMapGlobalIndex(int local_index) const;

    KOKKOS_INLINE_FUNCTION
    long long int getMapGlobalIndex(int local_index) const;

    KOKKOS_INLINE_FUNCTION
    int getSubMapLocalIndex(long long int local_index) const;

    KOKKOS_INLINE_FUNCTION
    int getMapLocalIndex(long long int local_index) const;

    // Host Method
    // Method that returns size
    KOKKOS_INLINE_FUNCTION
    size_t extent() const;

    KOKKOS_INLINE_FUNCTION
    size_t dims(size_t i) const;

    size_t global_dim() const;

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
TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::TpetraMVArray(): tpetra_pmap(NULL){
    length_ = order_ = 0;
    for (int i = 0; i < 7; i++) {
        dims_[i] = 0;
    }
}

// Overloaded 1D constructor where you provide dimensions, partitioning is done along first dimension
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::TpetraMVArray(size_t dim0, const std::string& tag_string, MPI_Comm mpi_comm) {
    mpi_comm_ = mpi_comm;
    global_dim1_ = dim0;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) dim0, 0, teuchos_comm));
    pmap = TpetraPartitionMap<tpetra_GO,Layout,ExecSpace,MemoryTraits>(tpetra_pmap);
    dims_[0] = tpetra_pmap->getLocalNumElements();
    dims_[1] = 1;
    order_ = 1;
    length_ = dims_[0];
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], 1);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

// Overloaded 2D constructor where you provide dimensions, partitioning is done along first dimension
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::TpetraMVArray(size_t dim0, size_t dim1, const std::string& tag_string, MPI_Comm mpi_comm) {
    mpi_comm_ = mpi_comm;
    global_dim1_ = dim0;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) dim0, 0, teuchos_comm));
    pmap = TpetraPartitionMap<tpetra_GO,Layout,ExecSpace,MemoryTraits>(tpetra_pmap);
    dims_[0] = tpetra_pmap->getLocalNumElements();
    dims_[1] = dim1;
    order_ = 2;
    length_ = (dims_[0] * dims_[1]);
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], dim1);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

// Overloaded 1D constructor where you provide a partition map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::TpetraMVArray(TpetraPartitionMap<long long int,Layout,ExecSpace,MemoryTraits> input_pmap,
                                                             const std::string& tag_string) {
    mpi_comm_ = input_pmap.mpi_comm_;                                                            
    global_dim1_ = input_pmap.num_global_;
    tpetra_pmap = input_pmap.tpetra_map;
    pmap = input_pmap;
    dims_[0] = tpetra_pmap->getLocalNumElements();
    dims_[1] = 1;
    order_ = 1;
    length_ = dims_[0];
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], 1);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

// Overloaded 2D constructor where you provide a partition map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::TpetraMVArray(TpetraPartitionMap<long long int,Layout,ExecSpace,MemoryTraits> input_pmap,
                                                              size_t dim1, const std::string& tag_string) {
    mpi_comm_ = input_pmap.mpi_comm_;
    global_dim1_ = input_pmap.num_global_;
    tpetra_pmap = input_pmap.tpetra_map;
    pmap = input_pmap;
    dims_[0] = tpetra_pmap->getLocalNumElements();
    dims_[1] = dim1;
    order_ = 2;
    length_ = (dims_[0] * dims_[1]);
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], dim1);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

// Overloaded 1D constructor taking an RPC pointer to a Tpetra Map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::TpetraMVArray(Teuchos::RCP<Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap,
                                                              const std::string& tag_string) {
    
    global_dim1_ = input_pmap->getGlobalNumElements();
    dims_[0] = input_pmap->getLocalNumElements();
    dims_[1] = 1;
    order_ = 1;
    length_ = dims_[0];
    tpetra_pmap = input_pmap;
    pmap = TpetraPartitionMap<tpetra_GO,Layout,ExecSpace,MemoryTraits>(tpetra_pmap);
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], 1);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

// Overloaded 2D constructor taking an RPC pointer to a Tpetra Map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::TpetraMVArray(Teuchos::RCP<Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap,
                                                              size_t dim1, const std::string& tag_string) {
    
    global_dim1_ = input_pmap->getGlobalNumElements();
    dims_[0] = input_pmap->getLocalNumElements();
    dims_[1] = dim1;
    tpetra_pmap = input_pmap;
    pmap = TpetraPartitionMap<tpetra_GO,Layout,ExecSpace,MemoryTraits>(tpetra_pmap);
    order_ = 2;
    length_ = (dims_[0] * dims_[1]);
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], dim1);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

// Tpetra vector argument constructor: CURRENTLY DOESN'T WORK SINCE WE CANT GET DUAL VIEW FROM THE MULTIVECTOR
// template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
// TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::TpetraMVArray(Teuchos::RCP<MV> input_tpetra_vector, const std::string& tag_string){
    
//     tpetra_vector   = input_tpetra_vector;
//     tpetra_pmap = input_tpetra_vector->getMap(); 
//     //this_array_ = tpetra_vector->getWrappedDualView();
//     dims_[0] = tpetra_vector->getMap()->getLocalNumElements()();
//     dims_[1] = tpetra_vector->getNumVectors();

//     if(dims_[1]==1){
//         order_ = 1;
//     }
//     else{
//         order_ = 2;
//     }
//     length_ = (dims_[0] * dims_[1]);
//     set_mpi_type();
// }

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

// Return global index corresponding to the input local (on this process/rank) index for the sub map this vector comms from
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
long long int TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::getSubMapGlobalIndex(int local_index) const {
    long long int global_index = tpetra_comm_pmap->getGlobalElement(local_index);
    return global_index;
}

// Return global index corresponding to the input local (on this process/rank) index
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
long long int TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::getMapGlobalIndex(int local_index) const {
    long long int global_index = tpetra_pmap->getGlobalElement(local_index);
    return global_index;
}

// Return global index corresponding to the input local (on this process/rank) index for the sub map this vector comms from
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
int TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::getSubMapLocalIndex(long long int global_index) const {
    int local_index = tpetra_comm_pmap->getLocalElement(global_index);
    return local_index;
}

// Return global index corresponding to the input local (on this process/rank) index
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
int TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::getMapLocalIndex(long long int global_index) const {
    int local_index = tpetra_pmap->getLocalElement(global_index);
    return local_index;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T& TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::host(size_t i) const {
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in TpetraMVArray 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraMVArray 1D!");
    return this_array_.h_view(i,0);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T& TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::host(size_t i, size_t j) const {
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in TpetraMVArray 2D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraMVArray 2D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraMVArray 2D!");
    return this_array_.h_view(i,j);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>& TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::operator= (const TpetraMVArray& temp) {
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for
        global_dim1_ = temp.global_dim1_;
        order_ = temp.order_;
        length_ = temp.length_;
        this_array_ = temp.this_array_;
        mpi_comm_ = temp.mpi_comm_;
        mpi_datatype_ = temp.mpi_datatype_;
        tpetra_vector = temp.tpetra_vector;
        tpetra_sub_vector = temp.tpetra_sub_vector;
        pmap = temp.pmap;
        comm_pmap = temp.comm_pmap;
        tpetra_pmap = temp.tpetra_pmap;
        tpetra_comm_pmap = temp.tpetra_comm_pmap;
        importer = temp.importer;
        own_comms = temp.own_comms;
        submap_size_ = temp.submap_size_;
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
size_t TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::global_dim() const {
    return global_dim1_;
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
    tpetra_comm_pmap = other_pmap.tpetra_map;
    comm_pmap = TpetraPartitionMap<tpetra_GO,Layout,ExecSpace,MemoryTraits>(tpetra_comm_pmap);
    int local_offset = tpetra_pmap->getLocalElement((tpetra_comm_pmap->getMinGlobalIndex()));
    tpetra_sub_vector = Teuchos::rcp(new MV(*tpetra_vector, tpetra_comm_pmap, local_offset));
    submap_size_ = tpetra_comm_pmap->getLocalNumElements();
    importer = Teuchos::rcp(new Tpetra::Import<tpetra_LO, tpetra_GO>(tpetra_comm_pmap, tpetra_pmap));
}

//requires both tpetra_pmap and other_pmap to be contiguous and for other_pmap to be a subset of tpetra_pmap on every process
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::own_comm_setup(Teuchos::RCP<Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> other_pmap) {
    own_comms = true;
    tpetra_comm_pmap = other_pmap;
    comm_pmap = TpetraPartitionMap<tpetra_GO,Layout,ExecSpace,MemoryTraits>(tpetra_comm_pmap);
    int local_offset = tpetra_pmap->getLocalElement((tpetra_comm_pmap->getMinGlobalIndex()));
    tpetra_sub_vector = Teuchos::rcp(new MV(*tpetra_vector, tpetra_comm_pmap, local_offset));
    submap_size_ = tpetra_comm_pmap->getLocalNumElements();
    importer = Teuchos::rcp(new Tpetra::Import<tpetra_LO, tpetra_GO>(tpetra_comm_pmap, tpetra_pmap));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::perform_comms() {
    if(own_comms){
        tpetra_vector->doImport(*tpetra_sub_vector, *importer, Tpetra::INSERT);
    }
    else{}

}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::repartition_vector() {

    int num_dim = dims_[1];
    int nranks, process_rank;
    MPI_Comm_rank(mpi_comm_, &process_rank);
    MPI_Comm_size(mpi_comm_, &nranks);
    // construct input adapted needed by Zoltan2 problem
    //typedef Xpetra::MultiVector<real_t, tpetra_LO, tpetra_GO, tpetra_node_type> xvector_t;
    typedef Zoltan2::XpetraMultiVectorAdapter<MV> inputAdapter_t;
    typedef Zoltan2::EvaluatePartition<inputAdapter_t> quality_t;

    // Teuchos::RCP<xvector_t> xpetra_vector = 
    //     Teuchos::rcp(new Xpetra::TpetraMultiVector<real_t, tpetra_LO, tpetra_GO, tpetra_node_type>(tpetra_vector));

    //Teuchos::RCP<inputAdapter_t> problem_adapter =  Teuchos::rcp(new inputAdapter_t(xpetra_vector));
    Teuchos::RCP<inputAdapter_t> problem_adapter =  Teuchos::rcp(new inputAdapter_t(tpetra_vector));

    // Create parameters for an RCB problem

    double tolerance = 1.05;

    Teuchos::ParameterList params("Node Partition Params");
    params.set("debug_level", "basic_status");
    params.set("debug_procs", "0");
    params.set("error_check_level", "debug_mode_assertions");

    // params.set("algorithm", "rcb");
    params.set("algorithm", "multijagged");
    params.set("imbalance_tolerance", tolerance);
    params.set("num_global_parts", nranks);
    params.set("partitioning_objective", "minimize_cut_edge_count");

    Teuchos::RCP<Zoltan2::PartitioningProblem<inputAdapter_t>> problem =
        Teuchos::rcp(new Zoltan2::PartitioningProblem<inputAdapter_t>(&(*problem_adapter), &params));

    // Solve the problem

    problem->solve();

    // create metric object where communicator is Teuchos default

    quality_t* metricObject1 = new quality_t(&(*problem_adapter), &params, // problem1->getComm(),
                                             &problem->getSolution());
    // // Check the solution.

    if (process_rank == 0)
    {
        metricObject1->printMetrics(std::cout);
    }

    if (process_rank == 0)
    {
        real_t imb = metricObject1->getObjectCountImbalance();
        if (imb <= tolerance)
        {
            std::cout << "pass: " << imb << std::endl;
        }
        else
        {
            std::cout << "fail: " << imb << std::endl;
        }
        std::cout << std::endl;
    }
    delete metricObject1;

    // // migrate rows of the vector so they correspond to the partition recommended by Zoltan2

    // Teuchos::RCP<MV> partitioned_node_coords_distributed = Teuchos::rcp(new MV(map, num_dim));

    // Teuchos::RCP<xvector_t> xpartitioned_node_coords_distributed =
    //     Teuchos::rcp(new Xpetra::TpetraMultiVector<real_t, LO, GO, node_type>(partitioned_node_coords_distributed));

    problem_adapter->applyPartitioningSolution(*tpetra_vector, tpetra_vector, problem->getSolution());
    
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>(*(tpetra_vector->getMap())));
    // *partitioned_node_coords_distributed = Xpetra::toTpetra<real_t, LO, GO, node_type>(*xpartitioned_node_coords_distributed);

    // Teuchos::RCP<Tpetra::Map<LO, GO, node_type>> partitioned_map = Teuchos::rcp(new Tpetra::Map<LO, GO, node_type>(*(partitioned_node_coords_distributed->getMap())));

    Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> partitioned_map_one_to_one;
    partitioned_map_one_to_one = Tpetra::createOneToOne<tpetra_LO, tpetra_GO, tpetra_node_type>(tpetra_pmap);
    //Teuchos::RCP<MV> partitioned_node_coords_one_to_one_distributed = Teuchos::rcp(new MV(partitioned_map_one_to_one, num_dim));

    // Tpetra::Import<LO, GO> importer_one_to_one(partitioned_map, partitioned_map_one_to_one);
    // partitioned_node_coords_one_to_one_distributed->doImport(*partitioned_node_coords_distributed, importer_one_to_one, Tpetra::INSERT);
    // node_coords_distributed = partitioned_node_coords_one_to_one_distributed;
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>(*partitioned_map_one_to_one));
    pmap = TpetraPartitionMap<tpetra_GO,Layout,ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false; //reset submap setup now that full map is different
    dims_[0] = tpetra_pmap->getLocalNumElements();
    length_ = (dims_[0] * dims_[1]);
    // // migrate density vector if this is a restart file read
    // if (simparam.restart_file&&repartition_node_densities)
    // {
    //     Teuchos::RCP<MV> partitioned_node_densities_distributed = Teuchos::rcp(new MV(partitioned_map, 1));

    //     // create import object using local node indices map and all indices map
    //     Tpetra::Import<LO, GO> importer(map, partitioned_map);

    //     // comms to get ghosts
    //     partitioned_node_densities_distributed->doImport(*design_node_densities_distributed, importer, Tpetra::INSERT);
    //     design_node_densities_distributed = partitioned_node_densities_distributed;
    // }

    // // update nlocal_nodes and node map
    // map = partitioned_map;
    // nlocal_nodes = map->getLocalNumElements();
}

// Return size of the submap
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
size_t TpetraMVArray<T,Layout,ExecSpace,MemoryTraits>::submap_size() const {
    return submap_size_;
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

