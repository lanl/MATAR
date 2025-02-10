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
#include <Tpetra_LRMultiVector_decl.hpp>
#include <Tpetra_LRMultiVector_def.hpp>
#include <Kokkos_Core.hpp>
#include "Tpetra_Details_DefaultTypes.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_Details_makeColMap.hpp"
#include "Tpetra_Import_Util2.hpp"

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

//forward declarations for friendship
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
class TpetraCommunicationPlan;
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
class TpetraLRCommunicationPlan;

/////////////////////////
// TpetraPartitionMap:  Container storing global indices corresponding to local indices that belong on this process/rank as well as comms related data/functions.
/////////////////////////
template <typename ExecSpace = tpetra_execution_space, typename MemoryTraits = tpetra_memory_traits>
class TpetraPartitionMap {

    // these are unmanaged
    using TArray1D_host = Kokkos::View <const long long int*, tpetra_array_layout, HostSpace, MemoryTraits>;
    using TArray1D_dev = Kokkos::View <const long long int*, tpetra_array_layout, ExecSpace, MemoryTraits>;
    
    
protected:
    size_t length_;
    MPI_Datatype mpi_datatype_;
    TArray1D_host host;
    

public:
    
    TArray1D_dev device;
    //pointer to wrapped Tpetra map
    Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> tpetra_map; // map of node indices

    int num_global_;

    //MPI communicator
    MPI_Comm mpi_comm_;

    TpetraPartitionMap();

    //Copy Constructor
    KOKKOS_INLINE_FUNCTION
    TpetraPartitionMap(const TpetraPartitionMap<ExecSpace,MemoryTraits> &temp){
        *this = temp;
    }
     
    TpetraPartitionMap(size_t global_length, MPI_Comm mpi_comm = MPI_COMM_WORLD, const std::string& tag_string = DEFAULTSTRINGARRAY);

    TpetraPartitionMap(DCArrayKokkos<long long int> &indices, MPI_Comm mpi_comm = MPI_COMM_WORLD, const std::string& tag_string = DEFAULTSTRINGARRAY);

    TpetraPartitionMap(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_tpetra_map, const std::string& tag_string = DEFAULTSTRINGARRAY);

    KOKKOS_INLINE_FUNCTION
    const long long int& operator()(size_t i) const;

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
    
    long long int getGlobalIndex(int local_index) const;

    long long int getMinGlobalIndex() const;

    long long int getMaxGlobalIndex() const;
    
    bool isProcessGlobalIndex(int global_index) const;
    
    bool isProcessLocalIndex(int local_index) const;

    // Method returns the raw device pointer of the Kokkos DualView
    KOKKOS_INLINE_FUNCTION
    long long int* device_pointer() const;

    // Method returns the raw host pointer of the Kokkos DualView
    long long int* host_pointer() const;

    void print() const;

    // Deconstructor
    virtual KOKKOS_INLINE_FUNCTION
    ~TpetraPartitionMap ();
}; // End of TpetraPartitionMap


// Default constructor
template <typename ExecSpace, typename MemoryTraits>
TpetraPartitionMap<ExecSpace,MemoryTraits>::TpetraPartitionMap() {
    length_ = 0;
}

//Constructor for contiguous index decomposition
template <typename ExecSpace, typename MemoryTraits>
TpetraPartitionMap<ExecSpace,MemoryTraits>::TpetraPartitionMap(size_t global_length, MPI_Comm mpi_comm, const std::string& tag_string) {
    mpi_comm_ = mpi_comm;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm));
    tpetra_map = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) global_length, 0, teuchos_comm));
    num_global_ = global_length;
    TArray1D_host host = tpetra_map->getMyGlobalIndices();
    TArray1D_dev device = tpetra_map->getMyGlobalIndicesDevice();
    length_ = host.size();
    mpi_datatype_ = MPI_LONG_LONG_INT;
}

// Constructor to pass matar dual view of indices
template <typename ExecSpace, typename MemoryTraits>
TpetraPartitionMap<ExecSpace,MemoryTraits>::TpetraPartitionMap(DCArrayKokkos<long long int> &indices, MPI_Comm mpi_comm, const std::string& tag_string) {
    mpi_comm_ = mpi_comm;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm));
    tpetra_map = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>(Teuchos::OrdinalTraits<tpetra_GO>::invalid(), indices.get_kokkos_dual_view().d_view, 0, teuchos_comm));
    TArray1D_host host = tpetra_map->getMyGlobalIndices();
    TArray1D_dev device = tpetra_map->getMyGlobalIndicesDevice();
    length_ = host.size();
    num_global_ = tpetra_map->getGlobalNumElements();
    mpi_datatype_ = MPI_LONG_LONG_INT;
}

// Constructor to pass an existing Tpetra map
template <typename ExecSpace, typename MemoryTraits>
TpetraPartitionMap<ExecSpace,MemoryTraits>::TpetraPartitionMap(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_tpetra_map, const std::string& tag_string) {
    tpetra_map = input_tpetra_map;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = tpetra_map->getComm();
    mpi_comm_ = getRawMpiComm(*teuchos_comm);
    TArray1D_host host = input_tpetra_map->getMyGlobalIndices();
    TArray1D_dev device = input_tpetra_map->getMyGlobalIndicesDevice();
    length_ = host.size();
    num_global_ = tpetra_map->getGlobalNumElements();
    mpi_datatype_ = MPI_LONG_LONG_INT;
}

template <typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
const long long int& TpetraPartitionMap<ExecSpace,MemoryTraits>::operator()(size_t i) const {
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in TpetraPartitionMap 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraPartitionMap 1D!");
    return device(i);
}

template <typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
TpetraPartitionMap<ExecSpace,MemoryTraits>& TpetraPartitionMap<ExecSpace,MemoryTraits>::operator= (const TpetraPartitionMap& temp) {
    
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
template <typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t TpetraPartitionMap<ExecSpace,MemoryTraits>::size() const {
    return length_;
}

template <typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t TpetraPartitionMap<ExecSpace,MemoryTraits>::extent() const {
    return length_;
}

template <typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
long long int* TpetraPartitionMap<ExecSpace,MemoryTraits>::device_pointer() const {
    return device.data();
}

template <typename ExecSpace, typename MemoryTraits>
long long int* TpetraPartitionMap<ExecSpace,MemoryTraits>::host_pointer() const {
    return host.data();
}

template <typename ExecSpace, typename MemoryTraits>
void TpetraPartitionMap<ExecSpace,MemoryTraits>::print() const {
        std::ostream &out = std::cout;
        Teuchos::RCP<Teuchos::FancyOStream> fos;
        fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(out));
        tpetra_map->describe(*fos,Teuchos::VERB_EXTREME);
}

// Return local index (on this process/rank) corresponding to the input global index
template <typename ExecSpace, typename MemoryTraits>
int TpetraPartitionMap<ExecSpace,MemoryTraits>::getLocalIndex(int global_index) const {
    int local_index = tpetra_map->getLocalElement(global_index);
    return local_index;
}

// Return global index corresponding to the input local (on this process/rank) index
template <typename ExecSpace, typename MemoryTraits>
long long int TpetraPartitionMap<ExecSpace,MemoryTraits>::getGlobalIndex(int local_index) const {
    int global_index = tpetra_map->getGlobalElement(local_index);
    return global_index;
}

// Return smallest global index (on this process/rank)
template <typename ExecSpace, typename MemoryTraits>
long long int TpetraPartitionMap<ExecSpace,MemoryTraits>::getMinGlobalIndex() const {
    int global_index = tpetra_map->getMinGlobalIndex();
    return global_index;
}

// Return largest global index (on this process/rank)
template <typename ExecSpace, typename MemoryTraits>
long long int TpetraPartitionMap<ExecSpace,MemoryTraits>::getMaxGlobalIndex() const {
    int global_index = tpetra_map->getMaxGlobalIndex();
    return global_index;
}

// Return global index corresponding to the input local (on this process/rank) index
template <typename ExecSpace, typename MemoryTraits>
bool TpetraPartitionMap<ExecSpace,MemoryTraits>::isProcessGlobalIndex(int global_index) const {
    bool belongs = tpetra_map->isNodeGlobalElement(global_index);
    return belongs;
}

// Return global index corresponding to the input local (on this process/rank) index
template <typename ExecSpace, typename MemoryTraits>
bool TpetraPartitionMap<ExecSpace,MemoryTraits>::isProcessLocalIndex(int local_index) const {
    bool belongs = tpetra_map->isNodeGlobalElement(local_index);
    return belongs;
}

template <typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
TpetraPartitionMap<ExecSpace,MemoryTraits>::~TpetraPartitionMap() {}

////////////////////////////////////////////////////////////////////////////////
// End of TpetraPartitionMap
////////////////////////////////////////////////////////////////////////////////

/////////////////////////
// TpetraDCArray:  Tpetra wrapper for a distributed multivector (several components per vector element).
/////////////////////////

template <typename T, typename Layout = Kokkos::LayoutRight, typename ExecSpace = tpetra_execution_space, typename MemoryTraits = tpetra_memory_traits>
class TpetraDCArray {

    // this is manage
    using  TArray1D = Kokkos::DualView <T**, Layout, ExecSpace, MemoryTraits>;

    size_t dims_[7];
    size_t global_dim1_;
    size_t submap_size_;
    size_t length_, component_length_;
    size_t order_;  // tensor order (rank)
    MPI_Comm mpi_comm_;
    MPI_Datatype mpi_datatype_;
    TArray1D this_array_;
    
    // Trilinos type definitions
    typedef typename Tpetra::LRMultiVector<T, tpetra_LO, tpetra_GO> MV; //stands for MultiVector
    typedef typename Kokkos::View<T*, Kokkos::LayoutRight, tpetra_device_type, tpetra_memory_traits> values_array;
    typedef Kokkos::View<tpetra_GO*, Layout, tpetra_device_type, tpetra_memory_traits> global_indices_array;
    typedef Kokkos::View<tpetra_LO*, Layout, tpetra_device_type, tpetra_memory_traits> indices_array;
    typedef typename MV::dual_view_type::t_dev vec_array;
    typedef typename MV::dual_view_type::t_host host_vec_array;
    typedef typename Kokkos::View<const T**, Layout, HostSpace, tpetra_memory_traits> const_host_vec_array;
    typedef typename Kokkos::View<const T**, Layout, tpetra_device_type, tpetra_memory_traits> const_vec_array;
    typedef Kokkos::View<const int**, Layout, HostSpace, tpetra_memory_traits> const_host_ivec_array;
    typedef Kokkos::View<int**, Layout, HostSpace, tpetra_memory_traits> host_ivec_array;
    typedef typename MV::dual_view_type dual_vec_array;

    Teuchos::RCP<Tpetra::Import<tpetra_LO, tpetra_GO>> importer; // tpetra comms object
    

public:
    
    friend class TpetraLRCommunicationPlan<T,Layout,ExecSpace,MemoryTraits>;
    //data for arrays that own both shared and local data and aren't intended to communicate with another MATAR type
    //This is simplifying for cases such as a local + ghost storage vector where you need to update the ghost entries
    bool own_comms; //This Mapped MPI Array contains its own communication plan; just call array_comms()
    
    void set_mpi_type();
    TpetraPartitionMap<ExecSpace, MemoryTraits> pmap;
    TpetraPartitionMap<ExecSpace, MemoryTraits> comm_pmap;
    Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> tpetra_pmap;
    Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> tpetra_comm_pmap;
    Teuchos::RCP<MV>       tpetra_vector;
    Teuchos::RCP<MV>       tpetra_sub_vector; //for owned comms situations

    TpetraDCArray();
    
    //Copy Constructor
    KOKKOS_INLINE_FUNCTION
    TpetraDCArray(const TpetraDCArray<T, Layout, ExecSpace,MemoryTraits> &temp){
        *this = temp;
    }

    /* Default Contigous Map Constructors*/
    //Tpetra type for 1D case(still allocates dim0 by 1 using **T); partitions with unique indices per process
    TpetraDCArray(size_t dim0, const std::string& tag_string = DEFAULTSTRINGARRAY, MPI_Comm mpi_comm = MPI_COMM_WORLD);

    //Tpetra type for 2D case; partitions along rows with unique indices per process
    TpetraDCArray(size_t dim0, size_t dim1, const std::string& tag_string = DEFAULTSTRINGARRAY, MPI_Comm mpi_comm = MPI_COMM_WORLD);
    
    //Tpetra type for 3D case; partitions along rows with unique indices per process
    TpetraDCArray(size_t dim0, size_t dim1, size_t dim2, const std::string& tag_string = DEFAULTSTRINGARRAY, MPI_Comm mpi_comm = MPI_COMM_WORLD);
    
    //Tpetra type for 4D case; partitions along rows with unique indices per process
    TpetraDCArray(size_t dim0, size_t dim1, size_t dim2, size_t dim3,
                  const std::string& tag_string = DEFAULTSTRINGARRAY, MPI_Comm mpi_comm = MPI_COMM_WORLD);
    
    //Tpetra type for 5D case; partitions along rows with unique indices per process
    TpetraDCArray(size_t dim0, size_t dim1, size_t dim2, size_t dim3,
                  size_t dim4, const std::string& tag_string = DEFAULTSTRINGARRAY, MPI_Comm mpi_comm = MPI_COMM_WORLD);
    
    //Tpetra type for 6D case; partitions along rows with unique indices per process
    TpetraDCArray(size_t dim0, size_t dim1, size_t dim2, size_t dim3,
                  size_t dim4, size_t dim5, const std::string& tag_string = DEFAULTSTRINGARRAY, MPI_Comm mpi_comm = MPI_COMM_WORLD);
    
    //Tpetra type for 7D case; partitions along rows with unique indices per process
    TpetraDCArray(size_t dim0, size_t dim1, size_t dim2, size_t dim3,
                  size_t dim4, size_t dim5, size_t dim6, const std::string& tag_string = DEFAULTSTRINGARRAY, MPI_Comm mpi_comm = MPI_COMM_WORLD);

    /* Specified Map Constructors*/
    //Tpetra type for 1D case with a partition map passed in
    TpetraDCArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap,
                  const std::string& tag_string = DEFAULTSTRINGARRAY);

    //2D Tpetra type with a partition map passed in
    TpetraDCArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap, size_t dim1,
                  const std::string& tag_string = DEFAULTSTRINGARRAY);

    //3D Tpetra type with a partition map passed in
    TpetraDCArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap, size_t dim1,
                  size_t dim2, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    //4D Tpetra type with a partition map passed in
    TpetraDCArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap, size_t dim1, size_t dim2, size_t dim3,
                  const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    //5D Tpetra type with a partition map passed in
    TpetraDCArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap, size_t dim1, size_t dim2, size_t dim3,
                  size_t dim4, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    //6D Tpetra type with a partition map passed in
    TpetraDCArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap, size_t dim1, size_t dim2, size_t dim3,
                  size_t dim4, size_t dim5, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    //7D Tpetra type with a partition map passed in
    TpetraDCArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap, size_t dim1, size_t dim2, size_t dim3,
                  size_t dim4, size_t dim5, size_t dim6, const std::string& tag_string = DEFAULTSTRINGARRAY);

    //Tpetra type for 1D case(still allocates dim0 by 1 using **T); this constructor takes an RCP pointer to a Tpetra Map directly
    TpetraDCArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap,
                  const std::string& tag_string = DEFAULTSTRINGARRAY);

    //Tpetra type only goes up to 2D access; this constructor takes an RCP pointer to a Tpetra Map directly
    TpetraDCArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap, size_t dim1,
                  const std::string& tag_string = DEFAULTSTRINGARRAY);

    //Tpetra type only goes up to 3D access; this constructor takes an RCP pointer to a Tpetra Map directly
    TpetraDCArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap, size_t dim1,
                  size_t dim2, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    //Tpetra type only goes up to 4D access; this constructor takes an RCP pointer to a Tpetra Map directly
    TpetraDCArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap, size_t dim1, size_t dim2, size_t dim3,
                  const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    //Tpetra type only goes up to 5D access; this constructor takes an RCP pointer to a Tpetra Map directly
    TpetraDCArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap, size_t dim1, size_t dim2, size_t dim3,
                  size_t dim4, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    //Tpetra type only goes up to 6D access; this constructor takes an RCP pointer to a Tpetra Map directly
    TpetraDCArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap, size_t dim1, size_t dim2, size_t dim3,
                  size_t dim4, size_t dim5, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    //Tpetra type only goes up to 7D access; this constructor takes an RCP pointer to a Tpetra Map directly
    TpetraDCArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap, size_t dim1, size_t dim2, size_t dim3,
                  size_t dim4, size_t dim5, size_t dim6, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    //construct an array that views a contiguous subset of another array; start index denotes the local index in super vector to start the sub view
    TpetraDCArray(const TpetraDCArray<T, Layout, ExecSpace,MemoryTraits> &super_vector,
                  const TpetraPartitionMap<ExecSpace,MemoryTraits> &sub_pmap, size_t start_index);

    // 1D array setup
    void data_setup(const std::string& tag_string);

    // 2D array setup
    void data_setup( size_t dim1,
                const std::string& tag_string);

    // 3D array setup
    void data_setup( size_t dim1, size_t dim2,
                const std::string& tag_string);

    // 4D array setup
    void data_setup( size_t dim1, size_t dim2, size_t dim3,
                const std::string& tag_string);

    // 5D array setup
    void data_setup( size_t dim1, size_t dim2, size_t dim3,
                size_t dim4, const std::string& tag_string);

    // 6D array setup
    void data_setup( size_t dim1, size_t dim2, size_t dim3,
                size_t dim4, size_t dim5, const std::string& tag_string);

    // 7D array setup
    void data_setup(size_t dim1, size_t dim2, size_t dim3,
                size_t dim4, size_t dim5, size_t dim6, const std::string& tag_string);

    void own_comm_setup(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> other_pmap); //only call if the map in the arg is a uniquely owned submap of the arrays map
    
    void own_comm_setup(TpetraPartitionMap<ExecSpace,MemoryTraits> &other_pmap); //only call if the map in the arg is a uniquely owned submap of the arrays map

    void perform_comms();

    void repartition_vector(); //repartitions this vector using the zoltan2 multijagged algorithm on its data

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j, size_t k) const;

    KOKKOS_INLINE_FUNCTION
    T& operator() (size_t i, size_t j, size_t k,
                   size_t l) const;

    KOKKOS_INLINE_FUNCTION
    T& operator() (size_t i, size_t j, size_t k,
                   size_t l, size_t m) const;

    KOKKOS_INLINE_FUNCTION
    T& operator() (size_t i, size_t j, size_t k,
                   size_t l, size_t m, size_t n) const;

    KOKKOS_INLINE_FUNCTION
    T& operator() (size_t i, size_t j, size_t k,
                   size_t l, size_t m, size_t n, size_t o) const;

    T& host(size_t i) const;

    T& host(size_t i, size_t j) const;

    T& host(size_t i, size_t j, size_t k) const;

    T& host(size_t i, size_t j, size_t k,
            size_t l) const;
    
    T& host(size_t i, size_t j, size_t k,
            size_t l, size_t m) const;
    
    T& host(size_t i, size_t j, size_t k,
            size_t l, size_t m, size_t n) const;

    T& host(size_t i, size_t j, size_t k,
            size_t l, size_t m, size_t n, size_t o) const;
    
    KOKKOS_INLINE_FUNCTION
    TpetraDCArray& operator=(const TpetraDCArray& temp);

    // GPU Method
    // Method that returns size
    KOKKOS_INLINE_FUNCTION
    size_t size() const;

    size_t submap_size() const;

    long long int getSubMapGlobalIndex(int local_index) const;

    long long int getMapGlobalIndex(int local_index) const;

    int getSubMapLocalIndex(long long int local_index) const;

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

    //print vector data
    void print() const;

    // Deconstructor
    virtual KOKKOS_INLINE_FUNCTION
    ~TpetraDCArray ();
}; // End of TpetraDCArray


// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(): tpetra_pmap(NULL){
    length_ = order_ = component_length_ = 0;
    own_comms = false;
    for (int i = 0; i < 7; i++) {
        dims_[i] = 0;
    }
}

// Overloaded 1D constructor where you provide dimensions, partitioning is done along first dimension
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(size_t dim0, const std::string& tag_string, MPI_Comm mpi_comm) {
    mpi_comm_ = mpi_comm;
    global_dim1_ = dim0;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) dim0, 0, teuchos_comm));
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(tag_string);
}

// Overloaded 2D constructor where you provide dimensions, partitioning is done along first dimension
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(size_t dim0, size_t dim1, const std::string& tag_string, MPI_Comm mpi_comm) {
    mpi_comm_ = mpi_comm;
    global_dim1_ = dim0;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) dim0, 0, teuchos_comm));
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, tag_string);
}

// Overloaded 3D constructor where you provide dimensions, partitioning is done along first dimension
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(size_t dim0, size_t dim1, size_t dim2, const std::string& tag_string, MPI_Comm mpi_comm) {
    mpi_comm_ = mpi_comm;
    global_dim1_ = dim0;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) dim0, 0, teuchos_comm));
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, tag_string);
}

// Overloaded 4D constructor where you provide dimensions, partitioning is done along first dimension
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(size_t dim0, size_t dim1, size_t dim2, size_t dim3, const std::string& tag_string, MPI_Comm mpi_comm) {
    mpi_comm_ = mpi_comm;
    global_dim1_ = dim0;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) dim0, 0, teuchos_comm));
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, dim3, tag_string);
}

// Overloaded 5D constructor where you provide dimensions, partitioning is done along first dimension
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(size_t dim0, size_t dim1, size_t dim2, size_t dim3,
                                                              size_t dim4, const std::string& tag_string, MPI_Comm mpi_comm) {
    mpi_comm_ = mpi_comm;
    global_dim1_ = dim0;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) dim0, 0, teuchos_comm));
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, dim3, dim4, tag_string);
}

// Overloaded 6D constructor where you provide dimensions, partitioning is done along first dimension
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(size_t dim0, size_t dim1, size_t dim2, size_t dim3,
                                                              size_t dim4, size_t dim5, const std::string& tag_string, MPI_Comm mpi_comm) {
    mpi_comm_ = mpi_comm;
    global_dim1_ = dim0;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) dim0, 0, teuchos_comm));
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, dim3, dim4, dim5, tag_string);
}

// Overloaded 7D constructor where you provide dimensions, partitioning is done along first dimension
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(size_t dim0, size_t dim1, size_t dim2, size_t dim3,
                                                              size_t dim4, size_t dim5, size_t dim6, const std::string& tag_string, MPI_Comm mpi_comm) {
    mpi_comm_ = mpi_comm;
    global_dim1_ = dim0;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) dim0, 0, teuchos_comm));
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, dim3, dim4, dim5, dim6, tag_string);
}

// Overloaded 1D constructor where you provide a partition map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap,
                                                             const std::string& tag_string) {
    mpi_comm_ = input_pmap.mpi_comm_;                                                            
    global_dim1_ = input_pmap.num_global_;
    tpetra_pmap = input_pmap.tpetra_map;
    pmap = input_pmap;
    own_comms = false;
    data_setup(tag_string);
}

// Overloaded 2D constructor where you provide a partition map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap,
                                                              size_t dim1, const std::string& tag_string) {
    mpi_comm_ = input_pmap.mpi_comm_;
    global_dim1_ = input_pmap.num_global_;
    tpetra_pmap = input_pmap.tpetra_map;
    pmap = input_pmap;
    own_comms = false;
    data_setup(dim1, tag_string);
}

// Overloaded 3D constructor where you provide a partition map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap,
                                                              size_t dim1, size_t dim2, const std::string& tag_string) {
    mpi_comm_ = input_pmap.mpi_comm_;
    global_dim1_ = input_pmap.num_global_;
    tpetra_pmap = input_pmap.tpetra_map;
    pmap = input_pmap;
    own_comms = false;
    data_setup(dim1, dim2, tag_string);
}

// Overloaded 4D constructor where you provide a partition map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap,
                                                              size_t dim1, size_t dim2, size_t dim3, const std::string& tag_string) {
    mpi_comm_ = input_pmap.mpi_comm_;
    global_dim1_ = input_pmap.num_global_;
    tpetra_pmap = input_pmap.tpetra_map;
    pmap = input_pmap;
    own_comms = false;
    data_setup(dim1, dim2, dim3, tag_string);
}

// Overloaded 5D constructor where you provide a partition map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap,
                                                              size_t dim1, size_t dim2, size_t dim3, size_t dim4, const std::string& tag_string) {
    mpi_comm_ = input_pmap.mpi_comm_;
    global_dim1_ = input_pmap.num_global_;
    tpetra_pmap = input_pmap.tpetra_map;
    pmap = input_pmap;
    own_comms = false;
    data_setup(dim1, dim2, dim3, dim4, tag_string);
}

// Overloaded 6D constructor where you provide a partition map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap,
                                                              size_t dim1, size_t dim2, size_t dim3, size_t dim4,
                                                              size_t dim5, const std::string& tag_string) {
    mpi_comm_ = input_pmap.mpi_comm_;
    global_dim1_ = input_pmap.num_global_;
    tpetra_pmap = input_pmap.tpetra_map;
    pmap = input_pmap;
    own_comms = false;
    data_setup(dim1, dim2, dim3, dim4, dim5, tag_string);
}

// Overloaded 7D constructor where you provide a partition map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap,
                                                              size_t dim1, size_t dim2, size_t dim3, size_t dim4,
                                                              size_t dim5, size_t dim6, const std::string& tag_string) {
    mpi_comm_ = input_pmap.mpi_comm_;
    global_dim1_ = input_pmap.num_global_;
    tpetra_pmap = input_pmap.tpetra_map;
    pmap = input_pmap;
    own_comms = false;
    data_setup(dim1, dim2, dim3, dim4, dim5, dim6, tag_string);
}

// Overloaded 1D constructor taking an RPC pointer to a Tpetra Map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap,
                                                              const std::string& tag_string) {
    
    global_dim1_ = input_pmap->getGlobalNumElements();
    tpetra_pmap = input_pmap;
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(tag_string);
}

// Overloaded 2D constructor taking an RPC pointer to a Tpetra Map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap,
                                                              size_t dim1, const std::string& tag_string) {
    
    global_dim1_ = input_pmap->getGlobalNumElements();
    tpetra_pmap = input_pmap;
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, tag_string);
}

// Overloaded 3D constructor taking an RPC pointer to a Tpetra Map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap,
                                                              size_t dim1, size_t dim2, const std::string& tag_string) {
    
    global_dim1_ = input_pmap->getGlobalNumElements();
    tpetra_pmap = input_pmap;
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, tag_string);
}

// Overloaded 4D constructor taking an RPC pointer to a Tpetra Map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap,
                                                              size_t dim1, size_t dim2, size_t dim3, const std::string& tag_string) {
    
    global_dim1_ = input_pmap->getGlobalNumElements();
    tpetra_pmap = input_pmap;
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, dim3, tag_string);
}

// Overloaded 5D constructor taking an RPC pointer to a Tpetra Map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap,
                                                              size_t dim1, size_t dim2, size_t dim3, size_t dim4, const std::string& tag_string) {
    
    global_dim1_ = input_pmap->getGlobalNumElements();
    tpetra_pmap = input_pmap;
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, dim3, dim4, tag_string);
}

// Overloaded 6D constructor taking an RPC pointer to a Tpetra Map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap,
                                                              size_t dim1, size_t dim2, size_t dim3, size_t dim4,
                                                              size_t dim5, const std::string& tag_string) {
    
    global_dim1_ = input_pmap->getGlobalNumElements();
    tpetra_pmap = input_pmap;
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, dim3, dim4, dim5, tag_string);
}

// Overloaded 7D constructor taking an RPC pointer to a Tpetra Map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap,
                                                              size_t dim1, size_t dim2, size_t dim3, size_t dim4,
                                                              size_t dim5, size_t dim6, const std::string& tag_string) {
    
    global_dim1_ = input_pmap->getGlobalNumElements();
    tpetra_pmap = input_pmap;
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, dim3, dim4, dim5, dim6, tag_string);
}

//construct an array that views a contiguous subset of another array; start index denotes the local index in super vector to start the sub view
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(const TpetraDCArray<T, Layout, ExecSpace,MemoryTraits> &super_vector,
                const TpetraPartitionMap<ExecSpace,MemoryTraits> &sub_pmap, size_t start_index){
    mpi_comm_ = sub_pmap.mpi_comm_;                                                            
    global_dim1_ = sub_pmap.num_global_;
    tpetra_pmap = sub_pmap.tpetra_map;
    pmap = sub_pmap;
    own_comms = false;
    dims_[0] = tpetra_pmap->getLocalNumElements();
    for (int iter = 1; iter < super_vector.order_; iter++){
            dims_[iter] = super_vector.dims_[iter];
        } // end for

    if(super_vector.order_==1){
        dims_[1] = 1;
    }
    order_ = super_vector.order_;
    component_length_ = super_vector.component_length_;
    length_ = dims_[0]*component_length_;
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(super_vector.this_array_, std::pair<size_t,size_t>(start_index, super_vector.this_array_.extent(0)), Kokkos::ALL());
    tpetra_vector = Teuchos::rcp(new MV(*(super_vector.tpetra_vector), tpetra_pmap, start_index));
}

// Tpetra vector argument constructor: CURRENTLY DOESN'T WORK SINCE WE CANT GET DUAL VIEW FROM THE MULTIVECTOR
// template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
// TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDCArray(Teuchos::RCP<MV> input_tpetra_vector, const std::string& tag_string){
    
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

// Overloaded 1D array setup
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::data_setup(const std::string& tag_string) {
    dims_[0] = tpetra_pmap->getLocalNumElements();
    dims_[1] = 1;
    order_ = 1;
    length_ = dims_[0];
    component_length_ = 1;
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], component_length_);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

// Overloaded 2D array setup
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::data_setup(size_t dim1, const std::string& tag_string) {

    dims_[0] = tpetra_pmap->getLocalNumElements();
    dims_[1] = dim1;
    order_ = 2;
    length_ = (dims_[0] * dims_[1]);
    component_length_ = dims_[1];
    // Create host ViewCArray
    //set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], component_length_);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

// Overloaded 3D array setup
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::data_setup(size_t dim1, size_t dim2, const std::string& tag_string) {

    dims_[0] = tpetra_pmap->getLocalNumElements();
    dims_[1] = dim1;
    dims_[2] = dim2;
    order_ = 3;
    length_ = (dims_[0] * dims_[1] * dims_[2]);
    component_length_ = dims_[1] * dims_[2];
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], component_length_);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

// Overloaded 4D array setup
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::data_setup(size_t dim1, size_t dim2, size_t dim3, const std::string& tag_string) {

    dims_[0] = tpetra_pmap->getLocalNumElements();
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    order_ = 4;
    length_ = (dims_[0] * dims_[1] * dims_[2]* dims_[3]);
    component_length_ = dims_[1] * dims_[2] * dims_[3];
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], component_length_);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

// Overloaded 5D array setup
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::data_setup(size_t dim1, size_t dim2, size_t dim3,
                                                           size_t dim4, const std::string& tag_string) {

    dims_[0] = tpetra_pmap->getLocalNumElements();
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    order_ = 5;
    length_ = (dims_[0] * dims_[1] * dims_[2] * dims_[3]* dims_[4]);
    component_length_ = dims_[1] * dims_[2] * dims_[3] * dims_[4];
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], component_length_);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

// Overloaded 6D array setup
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::data_setup(size_t dim1, size_t dim2, size_t dim3,
                                                           size_t dim4, size_t dim5, const std::string& tag_string) {

    dims_[0] = tpetra_pmap->getLocalNumElements();
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    dims_[5] = dim5;
    order_ = 6;
    length_ = (dims_[0] * dims_[1] * dims_[2] * dims_[3] * dims_[4] * dims_[5]);
    component_length_ = dims_[1] * dims_[2] * dims_[3] * dims_[4] * dims_[5];
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], component_length_);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

// Overloaded 7D array setup
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::data_setup(size_t dim1, size_t dim2, size_t dim3,size_t dim4, size_t dim5, size_t dim6,
                                                           const std::string& tag_string) {
    
    dims_[0] = tpetra_pmap->getLocalNumElements();
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    dims_[5] = dim5;
    dims_[6] = dim6;
    order_ = 7;
    length_ = (dims_[0] * dims_[1] * dims_[2] * dims_[3] * dims_[4] * dims_[5] * dims_[6]);
    component_length_ = dims_[1] * dims_[2] * dims_[3] * dims_[4] * dims_[5] * dims_[6];
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], component_length_);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::set_mpi_type() {
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
        printf("Your entered TpetraDCArray type is not a supported type for MPI communications and is being set to int\n");
        mpi_datatype_ = MPI_INT;
    }
}

//1D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i) const {
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in TpetraDCArray 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDCArray 1D!");
    return this_array_.d_view(i,0);
}

//2D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j) const {
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in TpetraDCArray 2D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDCArray 2D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDCArray 2D!");
    return this_array_.d_view(i,j);
}

// 3D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k) const {
    assert(order_ == 3 && "Tensor order (rank) does not match constructor in TpetraDCArray 3D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDCArray 3D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDCArray 3D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDCArray 3D!");
    return this_array_.d_view(i, j * dims_[2] + k);
}

// 4D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l) const {
    assert(order_ == 4 && "Tensor order (rank) does not match constructor in TpetraDCArray 4D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDCArray 4D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDCArray 4D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDCArray 4D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in TpetraDCArray 4D!");
    return this_array_.d_view(i, j * dims_[2] * dims_[3] + k * dims_[3] + l);
}

// 5D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m) const {
    assert(order_ == 5 && "Tensor order (rank) does not match constructor in TpetraDCArray 5D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDCArray 5D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDCArray 5D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDCArray 5D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in TpetraDCArray 5D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in TpetraDCArray 5D!");
    return this_array_.d_view(i, j * dims_[2] * dims_[3] * dims_[4] +
                              k * dims_[3] * dims_[4] + l * dims_[4] + m);
}

// 6D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n) const {
    assert(order_ == 6 && "Tensor order (rank) does not match constructor in TpetraDCArray 6D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDCArray 6D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDCArray 6D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDCArray 6D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in TpetraDCArray 6D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in TpetraDCArray 6D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in TpetraDCArray 6D!");
    return this_array_.d_view(i, j * dims_[2] * dims_[3] * dims_[4]* dims_[5] +
                              k * dims_[3] * dims_[4]* dims_[5] + l * dims_[4]* dims_[5] +
                              m* dims_[5] + n);
}

// 7D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n, size_t o) const {
    assert(order_ == 7 && "Tensor order (rank) does not match constructor in TpetraDCArray 7D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDCArray 7D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDCArray 7D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDCArray 7D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in TpetraDCArray 7D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in TpetraDCArray 7D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in TpetraDCArray 7D!");
    assert(o >= 0 && o < dims_[6] && "o is out of bounds in TpetraDCArray 7D!");
    return this_array_.d_view(i,  j * dims_[2] * dims_[3] * dims_[4] * dims_[5] * dims_[6] +
                              k * dims_[3] * dims_[4]* dims_[5] * dims_[6] + l * dims_[4]* dims_[5] * dims_[6] +
                              m* dims_[5] * dims_[6] + n * dims_[6] + o);
}

// Return global index corresponding to the input local (on this process/rank) index for the sub map this vector comms from
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
long long int TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::getSubMapGlobalIndex(int local_index) const {
    long long int global_index = tpetra_comm_pmap->getGlobalElement(local_index);
    return global_index;
}

// Return global index corresponding to the input local (on this process/rank) index
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
long long int TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::getMapGlobalIndex(int local_index) const {
    long long int global_index = tpetra_pmap->getGlobalElement(local_index);
    return global_index;
}

// Return global index corresponding to the input local (on this process/rank) index for the sub map this vector comms from
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
int TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::getSubMapLocalIndex(long long int global_index) const {
    int local_index = tpetra_comm_pmap->getLocalElement(global_index);
    return local_index;
}

// Return global index corresponding to the input local (on this process/rank) index
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
int TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::getMapLocalIndex(long long int global_index) const {
    int local_index = tpetra_pmap->getLocalElement(global_index);
    return local_index;
}

//1D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T& TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::host(size_t i) const {
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in TpetraDCArray 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDCArray 1D!");
    return this_array_.h_view(i,0);
}

//2D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T& TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::host(size_t i, size_t j) const {
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in TpetraDCArray 2D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDCArray 2D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDCArray 2D!");
    return this_array_.h_view(i,j);
}

// 3D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T& TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::host(size_t i, size_t j, size_t k) const {
    assert(order_ == 3 && "Tensor order (rank) does not match constructor in TpetraDCArray 3D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDCArray 3D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDCArray 3D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDCArray 3D!");
    return this_array_.h_view(i, j * dims_[2] + k);
}

// 4D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T& TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::host(size_t i, size_t j, size_t k, size_t l) const {
    assert(order_ == 4 && "Tensor order (rank) does not match constructor in TpetraDCArray 4D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDCArray 4D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDCArray 4D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDCArray 4D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in TpetraDCArray 4D!");
    return this_array_.h_view(i, j * dims_[2] * dims_[3] + k * dims_[3] + l);
}

// 5D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T& TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::host(size_t i, size_t j, size_t k, size_t l,
                               size_t m) const {
    assert(order_ == 5 && "Tensor order (rank) does not match constructor in TpetraDCArray 5D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDCArray 5D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDCArray 5D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDCArray 5D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in TpetraDCArray 5D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in TpetraDCArray 5D!");
    return this_array_.h_view(i, j * dims_[2] * dims_[3] * dims_[4] +
                              k * dims_[3] * dims_[4] + l * dims_[4] + m);
}

// 6D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T& TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::host(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n) const {
    assert(order_ == 6 && "Tensor order (rank) does not match constructor in TpetraDCArray 6D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDCArray 6D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDCArray 6D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDCArray 6D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in TpetraDCArray 6D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in TpetraDCArray 6D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in TpetraDCArray 6D!");
    return this_array_.h_view(i, j * dims_[2] * dims_[3] * dims_[4]* dims_[5] +
                              k * dims_[3] * dims_[4]* dims_[5] + l * dims_[4]* dims_[5] +
                              m* dims_[5] + n);
}

// 7D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T& TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::host(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n, size_t o) const {
    assert(order_ == 7 && "Tensor order (rank) does not match constructor in TpetraDCArray 7D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDCArray 7D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDCArray 7D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDCArray 7D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in TpetraDCArray 7D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in TpetraDCArray 7D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in TpetraDCArray 7D!");
    assert(o >= 0 && o < dims_[6] && "o is out of bounds in TpetraDCArray 7D!");
    return this_array_.h_view(i,  j * dims_[2] * dims_[3] * dims_[4] * dims_[5] * dims_[6] +
                              k * dims_[3] * dims_[4]* dims_[5] * dims_[6] + l * dims_[4]* dims_[5] * dims_[6] +
                              m* dims_[5] * dims_[6] + n * dims_[6] + o);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>& TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::operator= (const TpetraDCArray& temp) {
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for

        if(temp.order_==1){
            dims_[1] = 1;
        }

        global_dim1_ = temp.global_dim1_;
        order_ = temp.order_;
        length_ = temp.length_;
        component_length_ = temp.component_length_;
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
size_t TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::size() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::extent() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::dims(size_t i) const {
    assert(i < order_ && "TpetraDCArray order (rank) does not match constructor, dim[i] does not exist!");
    assert(i >= 0 && dims_[i]>0 && "Access to TpetraDCArray dims is out of bounds!");
    return dims_[i];
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
size_t TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::global_dim() const {
    return global_dim1_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::order() const {
    return order_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::device_pointer() const {
    return this_array_.d_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::host_pointer() const {
    return this_array_.h_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
Kokkos::DualView <T**, Layout, ExecSpace, MemoryTraits> TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_dual_view() const {
  return this_array_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::update_host() {

    this_array_.template modify<typename TArray1D::execution_space>();
    this_array_.template sync<typename TArray1D::host_mirror_space>();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::update_device() {

    this_array_.template modify<typename TArray1D::host_mirror_space>();
    this_array_.template sync<typename TArray1D::execution_space>();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::own_comm_setup(TpetraPartitionMap<ExecSpace,MemoryTraits> &other_pmap) {
    own_comms = true;
    tpetra_comm_pmap = other_pmap.tpetra_map;
    comm_pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_comm_pmap);
    int local_offset = tpetra_pmap->getLocalElement((tpetra_comm_pmap->getMinGlobalIndex()));
    tpetra_sub_vector = Teuchos::rcp(new MV(*tpetra_vector, tpetra_comm_pmap, local_offset));
    submap_size_ = tpetra_comm_pmap->getLocalNumElements();
    importer = Teuchos::rcp(new Tpetra::Import<tpetra_LO, tpetra_GO>(tpetra_comm_pmap, tpetra_pmap));
}

//requires both tpetra_pmap and other_pmap to be contiguous and for other_pmap to be a subset of tpetra_pmap on every process
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::own_comm_setup(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> other_pmap) {
    own_comms = true;
    tpetra_comm_pmap = other_pmap;
    comm_pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_comm_pmap);
    int local_offset = tpetra_pmap->getLocalElement((tpetra_comm_pmap->getMinGlobalIndex()));
    tpetra_sub_vector = Teuchos::rcp(new MV(*tpetra_vector, tpetra_comm_pmap, local_offset));
    submap_size_ = tpetra_comm_pmap->getLocalNumElements();
    importer = Teuchos::rcp(new Tpetra::Import<tpetra_LO, tpetra_GO>(tpetra_comm_pmap, tpetra_pmap));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::perform_comms() {
    if(own_comms){
        tpetra_vector->doImport(*tpetra_sub_vector, *importer, Tpetra::INSERT);
    }
    else{}

}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::print() const {
        std::ostream &out = std::cout;
        Teuchos::RCP<Teuchos::FancyOStream> fos;
        fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(out));
        tpetra_vector->describe(*fos,Teuchos::VERB_EXTREME);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::repartition_vector() {

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

    TArray1D this_array_temp = TArray1D(this_array_.d_view.label(), dims_[0], component_length_);
    Teuchos::RCP<MV> temp_tpetra_vector = Teuchos::rcp(new MV(tpetra_pmap, this_array_temp));
    problem_adapter->applyPartitioningSolution(*tpetra_vector, temp_tpetra_vector, problem->getSolution());
    
    // std::ostream &out = std::cout;
    // Teuchos::RCP<Teuchos::FancyOStream> fos;
    // fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(out));
    // temp_tpetra_vector->describe(*fos,Teuchos::VERB_EXTREME);
    // temp_tpetra_vector->getMap()->describe(*fos,Teuchos::VERB_EXTREME);
    
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>(*(temp_tpetra_vector->getMap())));
    tpetra_vector = temp_tpetra_vector;
    // *partitioned_node_coords_distributed = Xpetra::toTpetra<real_t, LO, GO, node_type>(*xpartitioned_node_coords_distributed);

    // Teuchos::RCP<Tpetra::Map<LO, GO, node_type>> partitioned_map = Teuchos::rcp(new Tpetra::Map<LO, GO, node_type>(*(partitioned_node_coords_distributed->getMap())));

    Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> partitioned_map_one_to_one;
    partitioned_map_one_to_one = Tpetra::createOneToOne<tpetra_LO, tpetra_GO, tpetra_node_type>(tpetra_pmap);
    temp_tpetra_vector = Teuchos::rcp(new MV(partitioned_map_one_to_one, num_dim));

    Tpetra::Import<tpetra_LO, tpetra_GO> importer_one_to_one(tpetra_vector->getMap(), partitioned_map_one_to_one);
    temp_tpetra_vector->doImport(*tpetra_vector, importer_one_to_one, Tpetra::INSERT);
    // node_coords_distributed = partitioned_node_coords_one_to_one_distributed;
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>(*partitioned_map_one_to_one));
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false; //reset submap setup now that full map is different
    dims_[0] = tpetra_pmap->getLocalNumElements();
    length_ = (dims_[0] * component_length_);

    //copy new partitioned vector into another one constructed with our managed dual view
    this_array_temp = TArray1D(this_array_.d_view.label(), dims_[0], component_length_);
    tpetra_vector = Teuchos::rcp(new MV(tpetra_pmap, this_array_temp));
    tpetra_vector->assign(*temp_tpetra_vector);
    this_array_ = this_array_temp;
    
    //for whatever reason, when using one process the device contains the updated data, when using several the host does
    //so we need this if block
    if(this_array_.template need_sync<typename TArray1D::execution_space>()){
        this_array_.template sync<typename TArray1D::execution_space>();
    }
    else{
        this_array_.template sync<typename TArray1D::host_mirror_space>();
    }
}

// Return size of the submap
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
size_t TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::submap_size() const {
    return submap_size_;
}

//MPI_Barrier wrapper
//template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
//void TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::barrier(MPI_Comm comm) {
//    MPI_Barrier(comm); 
//}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
TpetraDCArray<T,Layout,ExecSpace,MemoryTraits>::~TpetraDCArray() {}

////////////////////////////////////////////////////////////////////////////////
// End of TpetraDCArray
////////////////////////////////////////////////////////////////////////////////

/////////////////////////
// TpetraDFArray:  Tpetra wrapper for a distributed multivector (several components per vector element).
/////////////////////////

template <typename T, typename Layout = tpetra_array_layout, typename ExecSpace = tpetra_execution_space, typename MemoryTraits = tpetra_memory_traits>
class TpetraDFArray {

    // this is manage
    using  TArray1D = Kokkos::DualView <T**, Layout, ExecSpace, MemoryTraits>;

    size_t dims_[7];
    size_t global_dim1_;
    size_t submap_size_;
    size_t length_, component_length_;
    size_t order_;  // tensor order (rank)
    MPI_Comm mpi_comm_;
    MPI_Datatype mpi_datatype_;
    TArray1D this_array_;
    
    // Trilinos type definitions
    typedef typename Tpetra::MultiVector<T, tpetra_LO, tpetra_GO> MV; //stands for MultiVector
    typedef typename  Kokkos::View<T*, Kokkos::LayoutRight, tpetra_device_type, tpetra_memory_traits> values_array;
    typedef Kokkos::View<tpetra_GO*, tpetra_array_layout, tpetra_device_type, tpetra_memory_traits> global_indices_array;
    typedef Kokkos::View<tpetra_LO*, tpetra_array_layout, tpetra_device_type, tpetra_memory_traits> indices_array;
    typedef typename MV::dual_view_type::t_dev vec_array;
    typedef typename MV::dual_view_type::t_host host_vec_array;
    typedef typename Kokkos::View<const T**, tpetra_array_layout, HostSpace, tpetra_memory_traits> const_host_vec_array;
    typedef typename Kokkos::View<const T**, tpetra_array_layout, tpetra_device_type, tpetra_memory_traits> const_vec_array;
    typedef typename MV::dual_view_type dual_vec_array;

    Teuchos::RCP<Tpetra::Import<tpetra_LO, tpetra_GO>> importer; // tpetra comms object
    

public:
    friend class TpetraCommunicationPlan<T,Layout,ExecSpace,MemoryTraits>;
    //data for arrays that own both shared and local data and aren't intended to communicate with another MATAR type
    //This is simplifying for cases such as a local + ghost storage vector where you need to update the ghost entries
    bool own_comms; //This Mapped MPI Array contains its own communication plan; just call array_comms()
    
    void set_mpi_type();
    TpetraPartitionMap<ExecSpace, MemoryTraits> pmap;
    TpetraPartitionMap<ExecSpace, MemoryTraits> comm_pmap;
    Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> tpetra_pmap;
    Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> tpetra_comm_pmap;
    Teuchos::RCP<MV>       tpetra_vector;
    Teuchos::RCP<MV>       tpetra_sub_vector; //for owned comms situations

    TpetraDFArray();
    
    //Copy Constructor
    KOKKOS_INLINE_FUNCTION
    TpetraDFArray(const TpetraDFArray<T, Layout, ExecSpace,MemoryTraits> &temp){
        *this = temp;
    }

    /* Default Contigous Map Constructors*/
    //Tpetra type for 1D case(still allocates dim0 by 1 using **T); partitions with unique indices per process
    TpetraDFArray(size_t dim0, const std::string& tag_string = DEFAULTSTRINGARRAY, MPI_Comm mpi_comm = MPI_COMM_WORLD);

    //Tpetra type for 2D case; partitions along rows with unique indices per process
    TpetraDFArray(size_t dim0, size_t dim1, const std::string& tag_string = DEFAULTSTRINGARRAY, MPI_Comm mpi_comm = MPI_COMM_WORLD);
    
    //Tpetra type for 3D case; partitions along rows with unique indices per process
    TpetraDFArray(size_t dim0, size_t dim1, size_t dim2, const std::string& tag_string = DEFAULTSTRINGARRAY, MPI_Comm mpi_comm = MPI_COMM_WORLD);
    
    //Tpetra type for 4D case; partitions along rows with unique indices per process
    TpetraDFArray(size_t dim0, size_t dim1, size_t dim2, size_t dim3,
                  const std::string& tag_string = DEFAULTSTRINGARRAY, MPI_Comm mpi_comm = MPI_COMM_WORLD);
    
    //Tpetra type for 5D case; partitions along rows with unique indices per process
    TpetraDFArray(size_t dim0, size_t dim1, size_t dim2, size_t dim3,
                  size_t dim4, const std::string& tag_string = DEFAULTSTRINGARRAY, MPI_Comm mpi_comm = MPI_COMM_WORLD);
    
    //Tpetra type for 6D case; partitions along rows with unique indices per process
    TpetraDFArray(size_t dim0, size_t dim1, size_t dim2, size_t dim3,
                  size_t dim4, size_t dim5, const std::string& tag_string = DEFAULTSTRINGARRAY, MPI_Comm mpi_comm = MPI_COMM_WORLD);
    
    //Tpetra type for 7D case; partitions along rows with unique indices per process
    TpetraDFArray(size_t dim0, size_t dim1, size_t dim2, size_t dim3,
                  size_t dim4, size_t dim5, size_t dim6, const std::string& tag_string = DEFAULTSTRINGARRAY, MPI_Comm mpi_comm = MPI_COMM_WORLD);

    /* Specified Map Constructors*/
    //Tpetra type for 1D case with a partition map passed in
    TpetraDFArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap,
                  const std::string& tag_string = DEFAULTSTRINGARRAY);

    //2D Tpetra type with a partition map passed in
    TpetraDFArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap, size_t dim1,
                  const std::string& tag_string = DEFAULTSTRINGARRAY);

    //3D Tpetra type with a partition map passed in
    TpetraDFArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap, size_t dim1,
                  size_t dim2, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    //4D Tpetra type with a partition map passed in
    TpetraDFArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap, size_t dim1, size_t dim2, size_t dim3,
                  const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    //5D Tpetra type with a partition map passed in
    TpetraDFArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap, size_t dim1, size_t dim2, size_t dim3,
                  size_t dim4, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    //6D Tpetra type with a partition map passed in
    TpetraDFArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap, size_t dim1, size_t dim2, size_t dim3,
                  size_t dim4, size_t dim5, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    //7D Tpetra type with a partition map passed in
    TpetraDFArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap, size_t dim1, size_t dim2, size_t dim3,
                  size_t dim4, size_t dim5, size_t dim6, const std::string& tag_string = DEFAULTSTRINGARRAY);

    //Tpetra type for 1D case(still allocates dim0 by 1 using **T); this constructor takes an RCP pointer to a Tpetra Map directly
    TpetraDFArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap,
                  const std::string& tag_string = DEFAULTSTRINGARRAY);

    //Tpetra type only goes up to 2D access; this constructor takes an RCP pointer to a Tpetra Map directly
    TpetraDFArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap, size_t dim1,
                  const std::string& tag_string = DEFAULTSTRINGARRAY);

    //Tpetra type only goes up to 3D access; this constructor takes an RCP pointer to a Tpetra Map directly
    TpetraDFArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap, size_t dim1,
                  size_t dim2, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    //Tpetra type only goes up to 4D access; this constructor takes an RCP pointer to a Tpetra Map directly
    TpetraDFArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap, size_t dim1, size_t dim2, size_t dim3,
                  const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    //Tpetra type only goes up to 5D access; this constructor takes an RCP pointer to a Tpetra Map directly
    TpetraDFArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap, size_t dim1, size_t dim2, size_t dim3,
                  size_t dim4, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    //Tpetra type only goes up to 6D access; this constructor takes an RCP pointer to a Tpetra Map directly
    TpetraDFArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap, size_t dim1, size_t dim2, size_t dim3,
                  size_t dim4, size_t dim5, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    //Tpetra type only goes up to 7D access; this constructor takes an RCP pointer to a Tpetra Map directly
    TpetraDFArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap, size_t dim1, size_t dim2, size_t dim3,
                  size_t dim4, size_t dim5, size_t dim6, const std::string& tag_string = DEFAULTSTRINGARRAY);
    
    //construct an array that views a contiguous subset of another array; start index denotes the local index in super vector to start the sub view
    TpetraDFArray(const TpetraDFArray<T, Layout, ExecSpace,MemoryTraits> &super_vector,
                  const TpetraPartitionMap<ExecSpace,MemoryTraits> &sub_pmap, size_t start_index);

    // 1D array setup
    void data_setup(const std::string& tag_string);

    // 2D array setup
    void data_setup( size_t dim1,
                const std::string& tag_string);

    // 3D array setup
    void data_setup( size_t dim1, size_t dim2,
                const std::string& tag_string);

    // 4D array setup
    void data_setup( size_t dim1, size_t dim2, size_t dim3,
                const std::string& tag_string);

    // 5D array setup
    void data_setup( size_t dim1, size_t dim2, size_t dim3,
                size_t dim4, const std::string& tag_string);

    // 6D array setup
    void data_setup( size_t dim1, size_t dim2, size_t dim3,
                size_t dim4, size_t dim5, const std::string& tag_string);

    // 7D array setup
    void data_setup(size_t dim1, size_t dim2, size_t dim3,
                size_t dim4, size_t dim5, size_t dim6, const std::string& tag_string);

    void own_comm_setup(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> other_pmap); //only call if the map in the arg is a uniquely owned submap of the arrays map
    
    void own_comm_setup(TpetraPartitionMap<ExecSpace,MemoryTraits> &other_pmap); //only call if the map in the arg is a uniquely owned submap of the arrays map

    void perform_comms();

    void repartition_vector(); //repartitions this vector using the zoltan2 multijagged algorithm on its data

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j) const;

    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j, size_t k) const;

    KOKKOS_INLINE_FUNCTION
    T& operator() (size_t i, size_t j, size_t k,
                   size_t l) const;

    KOKKOS_INLINE_FUNCTION
    T& operator() (size_t i, size_t j, size_t k,
                   size_t l, size_t m) const;

    KOKKOS_INLINE_FUNCTION
    T& operator() (size_t i, size_t j, size_t k,
                   size_t l, size_t m, size_t n) const;

    KOKKOS_INLINE_FUNCTION
    T& operator() (size_t i, size_t j, size_t k,
                   size_t l, size_t m, size_t n, size_t o) const;

    T& host(size_t i) const;

    T& host(size_t i, size_t j) const;

    T& host(size_t i, size_t j, size_t k) const;

    T& host(size_t i, size_t j, size_t k,
            size_t l) const;
    
    T& host(size_t i, size_t j, size_t k,
            size_t l, size_t m) const;
    
    T& host(size_t i, size_t j, size_t k,
            size_t l, size_t m, size_t n) const;

    T& host(size_t i, size_t j, size_t k,
            size_t l, size_t m, size_t n, size_t o) const;
    
    KOKKOS_INLINE_FUNCTION
    TpetraDFArray& operator=(const TpetraDFArray& temp);

    // GPU Method
    // Method that returns size
    KOKKOS_INLINE_FUNCTION
    size_t size() const;

    size_t submap_size() const;

    long long int getSubMapGlobalIndex(int local_index) const;

    long long int getMapGlobalIndex(int local_index) const;

    int getSubMapLocalIndex(long long int local_index) const;

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

    //print vector data
    void print() const;

    // Deconstructor
    virtual KOKKOS_INLINE_FUNCTION
    ~TpetraDFArray ();
}; // End of TpetraDFArray


// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(): tpetra_pmap(NULL){
    length_ = order_ = component_length_ = 0;
    own_comms = false;
    for (int i = 0; i < 7; i++) {
        dims_[i] = 0;
    }
}

// Overloaded 1D constructor where you provide dimensions, partitioning is done along first dimension
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(size_t dim0, const std::string& tag_string, MPI_Comm mpi_comm) {
    mpi_comm_ = mpi_comm;
    global_dim1_ = dim0;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) dim0, 0, teuchos_comm));
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(tag_string);
}

// Overloaded 2D constructor where you provide dimensions, partitioning is done along first dimension
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(size_t dim0, size_t dim1, const std::string& tag_string, MPI_Comm mpi_comm) {
    mpi_comm_ = mpi_comm;
    global_dim1_ = dim0;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) dim0, 0, teuchos_comm));
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, tag_string);
}

// Overloaded 3D constructor where you provide dimensions, partitioning is done along first dimension
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(size_t dim0, size_t dim1, size_t dim2, const std::string& tag_string, MPI_Comm mpi_comm) {
    mpi_comm_ = mpi_comm;
    global_dim1_ = dim0;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) dim0, 0, teuchos_comm));
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, tag_string);
}

// Overloaded 4D constructor where you provide dimensions, partitioning is done along first dimension
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(size_t dim0, size_t dim1, size_t dim2, size_t dim3, const std::string& tag_string, MPI_Comm mpi_comm) {
    mpi_comm_ = mpi_comm;
    global_dim1_ = dim0;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) dim0, 0, teuchos_comm));
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, dim3, tag_string);
}

// Overloaded 5D constructor where you provide dimensions, partitioning is done along first dimension
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(size_t dim0, size_t dim1, size_t dim2, size_t dim3,
                                                              size_t dim4, const std::string& tag_string, MPI_Comm mpi_comm) {
    mpi_comm_ = mpi_comm;
    global_dim1_ = dim0;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) dim0, 0, teuchos_comm));
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, dim3, dim4, tag_string);
}

// Overloaded 6D constructor where you provide dimensions, partitioning is done along first dimension
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(size_t dim0, size_t dim1, size_t dim2, size_t dim3,
                                                              size_t dim4, size_t dim5, const std::string& tag_string, MPI_Comm mpi_comm) {
    mpi_comm_ = mpi_comm;
    global_dim1_ = dim0;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) dim0, 0, teuchos_comm));
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, dim3, dim4, dim5, tag_string);
}

// Overloaded 7D constructor where you provide dimensions, partitioning is done along first dimension
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(size_t dim0, size_t dim1, size_t dim2, size_t dim3,
                                                              size_t dim4, size_t dim5, size_t dim6, const std::string& tag_string, MPI_Comm mpi_comm) {
    mpi_comm_ = mpi_comm;
    global_dim1_ = dim0;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) dim0, 0, teuchos_comm));
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, dim3, dim4, dim5, dim6, tag_string);
}

// Overloaded 1D constructor where you provide a partition map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap,
                                                             const std::string& tag_string) {
    mpi_comm_ = input_pmap.mpi_comm_;                                                            
    global_dim1_ = input_pmap.num_global_;
    tpetra_pmap = input_pmap.tpetra_map;
    pmap = input_pmap;
    own_comms = false;
    data_setup(tag_string);
}

// Overloaded 2D constructor where you provide a partition map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap,
                                                              size_t dim1, const std::string& tag_string) {
    mpi_comm_ = input_pmap.mpi_comm_;
    global_dim1_ = input_pmap.num_global_;
    tpetra_pmap = input_pmap.tpetra_map;
    pmap = input_pmap;
    own_comms = false;
    data_setup(dim1, tag_string);
}

// Overloaded 3D constructor where you provide a partition map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap,
                                                              size_t dim1, size_t dim2, const std::string& tag_string) {
    mpi_comm_ = input_pmap.mpi_comm_;
    global_dim1_ = input_pmap.num_global_;
    tpetra_pmap = input_pmap.tpetra_map;
    pmap = input_pmap;
    own_comms = false;
    data_setup(dim1, dim2, tag_string);
}

// Overloaded 4D constructor where you provide a partition map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap,
                                                              size_t dim1, size_t dim2, size_t dim3, const std::string& tag_string) {
    mpi_comm_ = input_pmap.mpi_comm_;
    global_dim1_ = input_pmap.num_global_;
    tpetra_pmap = input_pmap.tpetra_map;
    pmap = input_pmap;
    own_comms = false;
    data_setup(dim1, dim2, dim3, tag_string);
}

// Overloaded 5D constructor where you provide a partition map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap,
                                                              size_t dim1, size_t dim2, size_t dim3, size_t dim4, const std::string& tag_string) {
    mpi_comm_ = input_pmap.mpi_comm_;
    global_dim1_ = input_pmap.num_global_;
    tpetra_pmap = input_pmap.tpetra_map;
    pmap = input_pmap;
    own_comms = false;
    data_setup(dim1, dim2, dim3, dim4, tag_string);
}

// Overloaded 6D constructor where you provide a partition map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap,
                                                              size_t dim1, size_t dim2, size_t dim3, size_t dim4,
                                                              size_t dim5, const std::string& tag_string) {
    mpi_comm_ = input_pmap.mpi_comm_;
    global_dim1_ = input_pmap.num_global_;
    tpetra_pmap = input_pmap.tpetra_map;
    pmap = input_pmap;
    own_comms = false;
    data_setup(dim1, dim2, dim3, dim4, dim5, tag_string);
}

// Overloaded 7D constructor where you provide a partition map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap,
                                                              size_t dim1, size_t dim2, size_t dim3, size_t dim4,
                                                              size_t dim5, size_t dim6, const std::string& tag_string) {
    mpi_comm_ = input_pmap.mpi_comm_;
    global_dim1_ = input_pmap.num_global_;
    tpetra_pmap = input_pmap.tpetra_map;
    pmap = input_pmap;
    own_comms = false;
    data_setup(dim1, dim2, dim3, dim4, dim5, dim6, tag_string);
}

// Overloaded 1D constructor taking an RPC pointer to a Tpetra Map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap,
                                                              const std::string& tag_string) {
    
    global_dim1_ = input_pmap->getGlobalNumElements();
    tpetra_pmap = input_pmap;
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(tag_string);
}

// Overloaded 2D constructor taking an RPC pointer to a Tpetra Map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap,
                                                              size_t dim1, const std::string& tag_string) {
    
    global_dim1_ = input_pmap->getGlobalNumElements();
    tpetra_pmap = input_pmap;
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, tag_string);
}

// Overloaded 3D constructor taking an RPC pointer to a Tpetra Map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap,
                                                              size_t dim1, size_t dim2, const std::string& tag_string) {
    
    global_dim1_ = input_pmap->getGlobalNumElements();
    tpetra_pmap = input_pmap;
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, tag_string);
}

// Overloaded 4D constructor taking an RPC pointer to a Tpetra Map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap,
                                                              size_t dim1, size_t dim2, size_t dim3, const std::string& tag_string) {
    
    global_dim1_ = input_pmap->getGlobalNumElements();
    tpetra_pmap = input_pmap;
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, dim3, tag_string);
}

// Overloaded 5D constructor taking an RPC pointer to a Tpetra Map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap,
                                                              size_t dim1, size_t dim2, size_t dim3, size_t dim4, const std::string& tag_string) {
    
    global_dim1_ = input_pmap->getGlobalNumElements();
    tpetra_pmap = input_pmap;
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, dim3, dim4, tag_string);
}

// Overloaded 6D constructor taking an RPC pointer to a Tpetra Map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap,
                                                              size_t dim1, size_t dim2, size_t dim3, size_t dim4,
                                                              size_t dim5, const std::string& tag_string) {
    
    global_dim1_ = input_pmap->getGlobalNumElements();
    tpetra_pmap = input_pmap;
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, dim3, dim4, dim5, tag_string);
}

// Overloaded 7D constructor taking an RPC pointer to a Tpetra Map
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> input_pmap,
                                                              size_t dim1, size_t dim2, size_t dim3, size_t dim4,
                                                              size_t dim5, size_t dim6, const std::string& tag_string) {
    
    global_dim1_ = input_pmap->getGlobalNumElements();
    tpetra_pmap = input_pmap;
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false;
    data_setup(dim1, dim2, dim3, dim4, dim5, dim6, tag_string);
}

//construct an array that views a contiguous subset of another array; start index denotes the local index in super vector to start the sub view
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(const TpetraDFArray<T, Layout, ExecSpace,MemoryTraits> &super_vector,
                const TpetraPartitionMap<ExecSpace,MemoryTraits> &sub_pmap, size_t start_index){
    mpi_comm_ = sub_pmap.mpi_comm_;                                                            
    global_dim1_ = sub_pmap.num_global_;
    tpetra_pmap = sub_pmap.tpetra_map;
    pmap = sub_pmap;
    own_comms = false;
    dims_[0] = tpetra_pmap->getLocalNumElements();
    for (int iter = 1; iter < super_vector.order_; iter++){
            dims_[iter] = super_vector.dims_[iter];
        } // end for

    if(super_vector.order_==1){
        dims_[1] = 1;
    }
    order_ = super_vector.order_;
    component_length_ = super_vector.component_length_;
    length_ = dims_[0]*component_length_;
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(super_vector.this_array_, std::pair<size_t,size_t>(start_index, super_vector.this_array_.extent(0)), Kokkos::ALL());
    tpetra_vector = Teuchos::rcp(new MV(*(super_vector.tpetra_vector), tpetra_pmap, start_index));
}

// Tpetra vector argument constructor: CURRENTLY DOESN'T WORK SINCE WE CANT GET DUAL VIEW FROM THE MULTIVECTOR
// template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
// TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::TpetraDFArray(Teuchos::RCP<MV> input_tpetra_vector, const std::string& tag_string){
    
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

// Overloaded 1D array setup
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::data_setup(const std::string& tag_string) {
    dims_[0] = tpetra_pmap->getLocalNumElements();
    dims_[1] = 1;
    order_ = 1;
    length_ = dims_[0];
    component_length_ = 1;
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], component_length_);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

// Overloaded 2D array setup
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::data_setup(size_t dim1, const std::string& tag_string) {

    dims_[0] = tpetra_pmap->getLocalNumElements();
    dims_[1] = dim1;
    order_ = 2;
    length_ = (dims_[0] * dims_[1]);
    component_length_ = dims_[1];
    // Create host ViewCArray
    //set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], component_length_);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

// Overloaded 3D array setup
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::data_setup(size_t dim1, size_t dim2, const std::string& tag_string) {

    dims_[0] = tpetra_pmap->getLocalNumElements();
    dims_[1] = dim1;
    dims_[2] = dim2;
    order_ = 3;
    length_ = (dims_[0] * dims_[1] * dims_[2]);
    component_length_ = dims_[1] * dims_[2];
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], component_length_);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

// Overloaded 4D array setup
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::data_setup(size_t dim1, size_t dim2, size_t dim3, const std::string& tag_string) {

    dims_[0] = tpetra_pmap->getLocalNumElements();
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    order_ = 4;
    length_ = (dims_[0] * dims_[1] * dims_[2]* dims_[3]);
    component_length_ = dims_[1] * dims_[2] * dims_[3];
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], component_length_);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

// Overloaded 5D array setup
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::data_setup(size_t dim1, size_t dim2, size_t dim3,
                                                           size_t dim4, const std::string& tag_string) {

    dims_[0] = tpetra_pmap->getLocalNumElements();
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    order_ = 5;
    length_ = (dims_[0] * dims_[1] * dims_[2] * dims_[3]* dims_[4]);
    component_length_ = dims_[1] * dims_[2] * dims_[3] * dims_[4];
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], component_length_);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

// Overloaded 6D array setup
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::data_setup(size_t dim1, size_t dim2, size_t dim3,
                                                           size_t dim4, size_t dim5, const std::string& tag_string) {

    dims_[0] = tpetra_pmap->getLocalNumElements();
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    dims_[5] = dim5;
    order_ = 6;
    length_ = (dims_[0] * dims_[1] * dims_[2] * dims_[3] * dims_[4] * dims_[5]);
    component_length_ = dims_[1] * dims_[2] * dims_[3] * dims_[4] * dims_[5];
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], component_length_);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

// Overloaded 7D array setup
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::data_setup(size_t dim1, size_t dim2, size_t dim3,size_t dim4, size_t dim5, size_t dim6,
                                                           const std::string& tag_string) {
    
    dims_[0] = tpetra_pmap->getLocalNumElements();
    dims_[1] = dim1;
    dims_[2] = dim2;
    dims_[3] = dim3;
    dims_[4] = dim4;
    dims_[5] = dim5;
    dims_[6] = dim6;
    order_ = 7;
    length_ = (dims_[0] * dims_[1] * dims_[2] * dims_[3] * dims_[4] * dims_[5] * dims_[6]);
    component_length_ = dims_[1] * dims_[2] * dims_[3] * dims_[4] * dims_[5] * dims_[6];
    // Create host ViewCArray
    set_mpi_type();
    this_array_ = TArray1D(tag_string, dims_[0], component_length_);
    tpetra_vector   = Teuchos::rcp(new MV(tpetra_pmap, this_array_));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::set_mpi_type() {
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
        printf("Your entered TpetraDFArray type is not a supported type for MPI communications and is being set to int\n");
        mpi_datatype_ = MPI_INT;
    }
}

//1D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i) const {
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in TpetraDFArray 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDFArray 1D!");
    return this_array_.d_view(i,0);
}

//2D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j) const {
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in TpetraDFArray 2D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDFArray 2D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDFArray 2D!");
    return this_array_.d_view(i,j);
}

// 3D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k) const {
    assert(order_ == 3 && "Tensor order (rank) does not match constructor in TpetraDFArray 3D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDFArray 3D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDFArray 3D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDFArray 3D!");
    return this_array_.d_view(i, j + (k * dims_[1]));
}

// 4D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l) const {
    assert(order_ == 4 && "Tensor order (rank) does not match constructor in TpetraDFArray 4D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDFArray 4D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDFArray 4D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDFArray 4D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in TpetraDFArray 4D!");
    return this_array_.d_view(i, j + (k * dims_[1])
                              + (l * dims_[1] * dims_[2]));
}

// 5D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m) const {
    assert(order_ == 5 && "Tensor order (rank) does not match constructor in TpetraDFArray 5D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDFArray 5D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDFArray 5D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDFArray 5D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in TpetraDFArray 5D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in TpetraDFArray 5D!");
    return this_array_.d_view(i, j + (k * dims_[1])
                         + (l * dims_[1] * dims_[2])
                         + (m * dims_[1] * dims_[2] * dims_[3]));
}

// 6D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n) const {
    assert(order_ == 6 && "Tensor order (rank) does not match constructor in TpetraDFArray 6D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDFArray 6D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDFArray 6D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDFArray 6D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in TpetraDFArray 6D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in TpetraDFArray 6D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in TpetraDFArray 6D!");
    return this_array_.d_view(i, j + (k * dims_[1])
                         + (l * dims_[1] * dims_[2])
                         + (m * dims_[1] * dims_[2] * dims_[3])
                         + (n * dims_[1] * dims_[2] * dims_[3] * dims_[4]));
}

// 7D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n, size_t o) const {
    assert(order_ == 7 && "Tensor order (rank) does not match constructor in TpetraDFArray 7D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDFArray 7D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDFArray 7D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDFArray 7D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in TpetraDFArray 7D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in TpetraDFArray 7D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in TpetraDFArray 7D!");
    assert(o >= 0 && o < dims_[6] && "o is out of bounds in TpetraDFArray 7D!");
    return this_array_.d_view(i, j + (k * dims_[1])
                         + (l * dims_[1] * dims_[2])
                         + (m * dims_[1] * dims_[2] * dims_[3])
                         + (n * dims_[1] * dims_[2] * dims_[3] * dims_[4])
                         + (o * dims_[1] * dims_[2] * dims_[3] * dims_[4] * dims_[5]));
}

// Return global index corresponding to the input local (on this process/rank) index for the sub map this vector comms from
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
long long int TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::getSubMapGlobalIndex(int local_index) const {
    long long int global_index = tpetra_comm_pmap->getGlobalElement(local_index);
    return global_index;
}

// Return global index corresponding to the input local (on this process/rank) index
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
long long int TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::getMapGlobalIndex(int local_index) const {
    long long int global_index = tpetra_pmap->getGlobalElement(local_index);
    return global_index;
}

// Return global index corresponding to the input local (on this process/rank) index for the sub map this vector comms from
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
int TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::getSubMapLocalIndex(long long int global_index) const {
    int local_index = tpetra_comm_pmap->getLocalElement(global_index);
    return local_index;
}

// Return global index corresponding to the input local (on this process/rank) index
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
int TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::getMapLocalIndex(long long int global_index) const {
    int local_index = tpetra_pmap->getLocalElement(global_index);
    return local_index;
}

//1D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T& TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::host(size_t i) const {
    assert(order_ == 1 && "Tensor order (rank) does not match constructor in TpetraDFArray 1D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDFArray 1D!");
    return this_array_.h_view(i,0);
}

//2D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T& TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::host(size_t i, size_t j) const {
    assert(order_ == 2 && "Tensor order (rank) does not match constructor in TpetraDFArray 2D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDFArray 2D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDFArray 2D!");
    return this_array_.h_view(i,j);
}

// 3D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T& TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::host(size_t i, size_t j, size_t k) const {
    assert(order_ == 3 && "Tensor order (rank) does not match constructor in TpetraDFArray 3D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDFArray 3D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDFArray 3D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDFArray 3D!");
    return this_array_.h_view(i, j + (k * dims_[1]));
}

// 4D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T& TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::host(size_t i, size_t j, size_t k, size_t l) const {
    assert(order_ == 4 && "Tensor order (rank) does not match constructor in TpetraDFArray 4D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDFArray 4D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDFArray 4D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDFArray 4D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in TpetraDFArray 4D!");
    return this_array_.h_view(i, j + (k * dims_[1])
                              + (l * dims_[1] * dims_[2]));
}

// 5D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T& TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::host(size_t i, size_t j, size_t k, size_t l,
                               size_t m) const {
    assert(order_ == 5 && "Tensor order (rank) does not match constructor in TpetraDFArray 5D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDFArray 5D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDFArray 5D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDFArray 5D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in TpetraDFArray 5D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in TpetraDFArray 5D!");
    return this_array_.h_view(i, j + (k * dims_[1])
                         + (l * dims_[1] * dims_[2])
                         + (m * dims_[1] * dims_[2] * dims_[3]));
}

// 6D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T& TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::host(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n) const {
    assert(order_ == 6 && "Tensor order (rank) does not match constructor in TpetraDFArray 6D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDFArray 6D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDFArray 6D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDFArray 6D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in TpetraDFArray 6D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in TpetraDFArray 6D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in TpetraDFArray 6D!");
    return this_array_.h_view(i, j + (k * dims_[1])
                         + (l * dims_[1] * dims_[2])
                         + (m * dims_[1] * dims_[2] * dims_[3])
                         + (n * dims_[1] * dims_[2] * dims_[3] * dims_[4]));
}

// 7D
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
T& TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::host(size_t i, size_t j, size_t k, size_t l,
                               size_t m, size_t n, size_t o) const {
    assert(order_ == 7 && "Tensor order (rank) does not match constructor in TpetraDFArray 7D!");
    assert(i >= 0 && i < dims_[0] && "i is out of bounds in TpetraDFArray 7D!");
    assert(j >= 0 && j < dims_[1] && "j is out of bounds in TpetraDFArray 7D!");
    assert(k >= 0 && k < dims_[2] && "k is out of bounds in TpetraDFArray 7D!");
    assert(l >= 0 && l < dims_[3] && "l is out of bounds in TpetraDFArray 7D!");
    assert(m >= 0 && m < dims_[4] && "m is out of bounds in TpetraDFArray 7D!");
    assert(n >= 0 && n < dims_[5] && "n is out of bounds in TpetraDFArray 7D!");
    assert(o >= 0 && o < dims_[6] && "o is out of bounds in TpetraDFArray 7D!");
    return this_array_.h_view(i, j + (k * dims_[1])
                         + (l * dims_[1] * dims_[2])
                         + (m * dims_[1] * dims_[2] * dims_[3])
                         + (n * dims_[1] * dims_[2] * dims_[3] * dims_[4])
                         + (o * dims_[1] * dims_[2] * dims_[3] * dims_[4] * dims_[5]));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>& TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::operator= (const TpetraDFArray& temp) {
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        for (int iter = 0; iter < temp.order_; iter++){
            dims_[iter] = temp.dims_[iter];
        } // end for

        if(temp.order_==1){
            dims_[1] = 1;
        }

        global_dim1_ = temp.global_dim1_;
        order_ = temp.order_;
        length_ = temp.length_;
        component_length_ = temp.component_length_;
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
size_t TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::size() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::extent() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::dims(size_t i) const {
    assert(i < order_ && "TpetraDFArray order (rank) does not match constructor, dim[i] does not exist!");
    assert(i >= 0 && dims_[i]>0 && "Access to TpetraDFArray dims is out of bounds!");
    return dims_[i];
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
size_t TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::global_dim() const {
    return global_dim1_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::order() const {
    return order_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::device_pointer() const {
    return this_array_.d_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::host_pointer() const {
    return this_array_.h_view.data();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
Kokkos::DualView <T**, Layout, ExecSpace, MemoryTraits> TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_dual_view() const {
  return this_array_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::update_host() {

    this_array_.template modify<typename TArray1D::execution_space>();
    this_array_.template sync<typename TArray1D::host_mirror_space>();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::update_device() {

    this_array_.template modify<typename TArray1D::host_mirror_space>();
    this_array_.template sync<typename TArray1D::execution_space>();
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::own_comm_setup(TpetraPartitionMap<ExecSpace,MemoryTraits> &other_pmap) {
    own_comms = true;
    tpetra_comm_pmap = other_pmap.tpetra_map;
    comm_pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_comm_pmap);
    int local_offset = tpetra_pmap->getLocalElement((tpetra_comm_pmap->getMinGlobalIndex()));
    tpetra_sub_vector = Teuchos::rcp(new MV(*tpetra_vector, tpetra_comm_pmap, local_offset));
    submap_size_ = tpetra_comm_pmap->getLocalNumElements();
    importer = Teuchos::rcp(new Tpetra::Import<tpetra_LO, tpetra_GO>(tpetra_comm_pmap, tpetra_pmap));
}

//requires both tpetra_pmap and other_pmap to be contiguous and for other_pmap to be a subset of tpetra_pmap on every process
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::own_comm_setup(Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> other_pmap) {
    own_comms = true;
    tpetra_comm_pmap = other_pmap;
    comm_pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_comm_pmap);
    int local_offset = tpetra_pmap->getLocalElement((tpetra_comm_pmap->getMinGlobalIndex()));
    tpetra_sub_vector = Teuchos::rcp(new MV(*tpetra_vector, tpetra_comm_pmap, local_offset));
    submap_size_ = tpetra_comm_pmap->getLocalNumElements();
    importer = Teuchos::rcp(new Tpetra::Import<tpetra_LO, tpetra_GO>(tpetra_comm_pmap, tpetra_pmap));
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::perform_comms() {
    if(own_comms){
        tpetra_vector->doImport(*tpetra_sub_vector, *importer, Tpetra::INSERT);
    }
    else{}

}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::print() const {
        std::ostream &out = std::cout;
        Teuchos::RCP<Teuchos::FancyOStream> fos;
        fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(out));
        tpetra_vector->describe(*fos,Teuchos::VERB_EXTREME);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::repartition_vector() {

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
    
    TArray1D this_array_temp = TArray1D(this_array_.d_view.label(), dims_[0], component_length_);
    Teuchos::RCP<MV> temp_tpetra_vector = Teuchos::rcp(new MV(tpetra_pmap, this_array_temp));
    problem_adapter->applyPartitioningSolution(*tpetra_vector, temp_tpetra_vector, problem->getSolution());
    
    // std::ostream &out = std::cout;
    // Teuchos::RCP<Teuchos::FancyOStream> fos;
    // fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(out));
    // temp_tpetra_vector->describe(*fos,Teuchos::VERB_EXTREME);
    // temp_tpetra_vector->getMap()->describe(*fos,Teuchos::VERB_EXTREME);
    
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>(*(temp_tpetra_vector->getMap())));
    tpetra_vector = temp_tpetra_vector;
    // *partitioned_node_coords_distributed = Xpetra::toTpetra<real_t, LO, GO, node_type>(*xpartitioned_node_coords_distributed);

    // Teuchos::RCP<Tpetra::Map<LO, GO, node_type>> partitioned_map = Teuchos::rcp(new Tpetra::Map<LO, GO, node_type>(*(partitioned_node_coords_distributed->getMap())));

    Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> partitioned_map_one_to_one;
    partitioned_map_one_to_one = Tpetra::createOneToOne<tpetra_LO, tpetra_GO, tpetra_node_type>(tpetra_pmap);
    temp_tpetra_vector = Teuchos::rcp(new MV(partitioned_map_one_to_one, num_dim));

    Tpetra::Import<tpetra_LO, tpetra_GO> importer_one_to_one(tpetra_vector->getMap(), partitioned_map_one_to_one);
    temp_tpetra_vector->doImport(*tpetra_vector, importer_one_to_one, Tpetra::INSERT);
    // node_coords_distributed = partitioned_node_coords_one_to_one_distributed;
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>(*partitioned_map_one_to_one));
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    own_comms = false; //reset submap setup now that full map is different
    dims_[0] = tpetra_pmap->getLocalNumElements();
    length_ = (dims_[0] * component_length_);

    //copy new partitioned vector into another one constructed with our managed dual view
    this_array_temp = TArray1D(this_array_.d_view.label(), dims_[0], component_length_);
    tpetra_vector = Teuchos::rcp(new MV(tpetra_pmap, this_array_temp));
    tpetra_vector->assign(*temp_tpetra_vector);
    this_array_ = this_array_temp;

    //for whatever reason, when using one process the device contains the updated data, when using several the host does
    //so we need this if block
    if(this_array_.template need_sync<typename TArray1D::execution_space>()){
        this_array_.template sync<typename TArray1D::execution_space>();
    }
    else{
        this_array_.template sync<typename TArray1D::host_mirror_space>();
    }

}

// Return size of the submap
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
size_t TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::submap_size() const {
    return submap_size_;
}

//MPI_Barrier wrapper
//template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
//void TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::barrier(MPI_Comm comm) {
//    MPI_Barrier(comm); 
//}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
TpetraDFArray<T,Layout,ExecSpace,MemoryTraits>::~TpetraDFArray() {}

////////////////////////////////////////////////////////////////////////////////
// End of TpetraDFArray
////////////////////////////////////////////////////////////////////////////////

/////////////////////////
// TpetraCRSMatrix:  CRS Matrix Tpetra wrapper.
/////////////////////////
template <typename T, typename Layout = tpetra_array_layout, typename ExecSpace = tpetra_execution_space, typename MemoryTraits = tpetra_memory_traits>
class TpetraCRSMatrix {

    // this is manage
    using  TArray1D = RaggedRightArrayKokkos<T, Kokkos::LayoutRight, ExecSpace>;
    using  TArray1D_Host = RaggedRightArrayKokkos<T, Kokkos::LayoutRight, HostSpace>;
    using  row_map_type = Kokkos::View<size_t*, Kokkos::LayoutRight, ExecSpace>;
    using  input_row_graph_type = RaggedRightArrayKokkos<long long int, Kokkos::LayoutRight, ExecSpace>;
    using  input_row_map_type = DCArrayKokkos<size_t, Layout, ExecSpace>;
    using  values_array = Kokkos::View<T*, Kokkos::LayoutRight, ExecSpace, MemoryTraits>;
    using  global_indices_array = Kokkos::View<tpetra_GO*, Layout, ExecSpace, MemoryTraits>;
    using  indices_array = Kokkos::View<tpetra_LO*, Layout, ExecSpace, MemoryTraits>;

    size_t dim1_;
    size_t global_dim1_;
    size_t column_map_size_;
    size_t length_;
    MPI_Comm mpi_comm_;
    MPI_Datatype mpi_datatype_;
    TArray1D this_array_;
    row_map_type mystrides_;
    row_map_type start_index_;
    indices_array crs_local_indices_;
    
    // Trilinos type definitions
    typedef Tpetra::CrsMatrix<real_t, tpetra_LO, tpetra_GO> MAT; //stands for matrix
    typedef const Tpetra::CrsMatrix<real_t, tpetra_LO, tpetra_GO> const_MAT;
    typedef Tpetra::MultiVector<real_t, tpetra_LO, tpetra_GO> MV;
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
    TpetraPartitionMap<ExecSpace, MemoryTraits> pmap;
    TpetraPartitionMap<ExecSpace, MemoryTraits> column_pmap;
    TpetraPartitionMap<ExecSpace, MemoryTraits> comm_pmap;
    Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> tpetra_pmap;
    Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> tpetra_column_pmap;
    Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>> tpetra_comm_pmap;
    Teuchos::RCP<MAT>       tpetra_crs_matrix;

    TpetraCRSMatrix();
    
    //Copy Constructor
    KOKKOS_INLINE_FUNCTION
    TpetraCRSMatrix(const TpetraCRSMatrix<T, Layout, ExecSpace,MemoryTraits> &temp){
        *this = temp;
    }

    //CRS matrix constructor for banded matrix case
    // TpetraCRSMatrix(size_t dim1, size_t dim2,
    //                 const std::string& tag_string = DEFAULTSTRINGARRAY, MPI_Comm mpi_comm = MPI_COMM_WORLD);
    
    // //CRS row distributed matrix constructor for rectangular matrix
    // TpetraCRSMatrix(size_t global_dim1, size_t dim2, const std::string& tag_string = DEFAULTSTRINGARRAY, MPI_Comm mpi_comm = MPI_COMM_WORLD);

    // Constructor that takes local data in a matar ragged type and build map from rows of that (unfinished)
    TpetraCRSMatrix(size_t dim1, input_row_map_type input_strides, DCArrayKokkos<tpetra_GO,Layout,ExecSpace,MemoryTraits> crs_graph,
                    TArray1D input_values, const std::string& tag_string = DEFAULTSTRINGARRAY, MPI_Comm mpi_comm = MPI_COMM_WORLD);

    // Constructor that takes local data in a matar ragged type and a row partition map (avoids multiple copies partition map data)
    TpetraCRSMatrix(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap, input_row_map_type input_strides,
                    input_row_graph_type crs_graph, TArray1D input_values,
                    const std::string& tag_string = DEFAULTSTRINGARRAY);


    KOKKOS_INLINE_FUNCTION
    T& operator()(size_t i, size_t j) const;

    // T& host(size_t i, size_t j) const;
    
    KOKKOS_INLINE_FUNCTION
    TpetraCRSMatrix& operator=(const TpetraCRSMatrix& temp);

    // GPU Method
    // Method that returns size
    KOKKOS_INLINE_FUNCTION
    size_t size() const;

    KOKKOS_INLINE_FUNCTION
    long long int getColumnMapGlobalIndex(int local_index) const;

    KOKKOS_INLINE_FUNCTION
    long long int getMapGlobalIndex(int local_index) const;

    KOKKOS_INLINE_FUNCTION
    int getColumnMapLocalIndex(long long int local_index) const;

    KOKKOS_INLINE_FUNCTION
    int getMapLocalIndex(long long int local_index) const;

    // Host Method
    // Method that returns size
    KOKKOS_INLINE_FUNCTION
    size_t extent() const;

    KOKKOS_INLINE_FUNCTION
    size_t dim1() const;

    size_t global_dim() const;
 
    // Method returns the raw device pointer of the Kokkos DualView
    KOKKOS_INLINE_FUNCTION
    T* device_pointer() const;

    // Method returns the raw host pointer of the Kokkos DualView
    // KOKKOS_INLINE_FUNCTION
    // T* host_pointer() const;

    // Method returns kokkos dual view
    KOKKOS_INLINE_FUNCTION
    Kokkos::View <T**, Layout, ExecSpace, MemoryTraits> get_kokkos_view() const;

    // // Method that update host view
    // void update_host();

    // Method that update device view
    void update_device();

    //print vector data
    void print() const;

    // Deconstructor
    virtual KOKKOS_INLINE_FUNCTION
    ~TpetraCRSMatrix ();
}; // End of TpetraCRSMatrix


// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::TpetraCRSMatrix(): tpetra_pmap(NULL){
    length_ = 0;
    for (int i = 0; i < 7; i++) {
        dim1_ = 0;
    }
}

// // Constructor that takes local data in a matar ragged type
// template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
// TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::TpetraCRSMatrix(size_t global_dim1, size_t dim2, const std::string& tag_string, MPI_Comm mpi_comm) {
//     mpi_comm_ = mpi_comm;
//     global_dim1_ = global_dim1;
//     Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
//     tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) global_dim1, 0, teuchos_comm));
//     pmap = TpetraPartitionMap<tpetra_GO,Layout,ExecSpace,MemoryTraits>(tpetra_pmap);
//     dim1_ = tpetra_pmap->getLocalNumElements();
//     //construct strides that are constant
//     mystrides_ = row_map_type("mystrides_",dim1_);
//     for(int irow = 0; irow < dim1_; irow++){
//         mystrides_(irow) = dim2;
//     }
//     this_array_ = input_values;
//     global_indices_array input_crs_graph = crs_graph.get_kokkos_dual_view().d_view;

    
//     //build column map for the global conductivity matrix
//     Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type> > colmap;
//     const Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type> > dommap = tpetra_pmap;

//     Tpetra::Details::makeColMap<tpetra_LO, tpetra_GO, tpetra_node_type>(colmap, tpetra_pmap, input_crs_graph.get_kokkos_dual_view().d_view, nullptr);
//     tpetra_column_pmap = colmap;
//     size_t nnz = input_crs_graph.size();

//     //debug print
//     //std::cout << "DOF GRAPH SIZE ON RANK " << myrank << " IS " << nnz << std::endl;
    
//     //local indices in the graph using the constructed column map
//     crs_local_indices_ = indices_array("crs_local_indices", nnz);
    
//     //row offsets with compatible template arguments
//         row_map_type row_offsets_pass("row_offsets", dim1_ + 1);
//         for(int ipass = 0; ipass < dim1_ + 1; ipass++){
//             row_offsets_pass(ipass) = input_values.start_index_(ipass);
//         }

//     size_t entrycount = 0;
//     for(int irow = 0; irow < dim1_; irow++){
//         for(int istride = 0; istride < mystrides_(irow); istride++){
//             crs_local_indices_(entrycount) = tpetra_column_pmap->getLocalElement(crs_graph(entrycount));
//             entrycount++;
//         }
//     }
    
//     //sort values and indices
//     Tpetra::Import_Util::sortCrsEntries<row_map_type, indices_array, values_array>(row_offsets_pass, crs_local_indices_.d_view, this_array_.get_kokkos_view());

//     tpetra_crs_matrix = Teuchos::rcp(new MAT(tpetra_pmap, tpetra_column_pmap, start_index_.d_view, crs_local_indices_.d_view, this_array_.get_kokkos_view()));
//     tpetra_crs_matrix->fillComplete();
// }

// Constructor that takes local data in a matar ragged type
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::TpetraCRSMatrix(size_t dim1, input_row_map_type input_strides, DCArrayKokkos<tpetra_GO,Layout,ExecSpace,MemoryTraits> crs_graph,
                                                                  TArray1D input_values, const std::string& tag_string, MPI_Comm mpi_comm) {
    mpi_comm_ = mpi_comm;
    global_dim1_ = dim1;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
    tpetra_pmap = Teuchos::rcp(new Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type>((long long int) dim1, 0, teuchos_comm));
    pmap = TpetraPartitionMap<ExecSpace,MemoryTraits>(tpetra_pmap);
    dim1_ = tpetra_pmap->getLocalNumElements();
    mystrides_ = input_strides;
    this_array_ = input_values;
    global_indices_array input_crs_graph = crs_graph.get_kokkos_dual_view().d_view;

    
    //build column map for the global conductivity matrix
    Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type> > colmap;
    const Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type> > dommap = tpetra_pmap;

    Tpetra::Details::makeColMap<tpetra_LO, tpetra_GO, tpetra_node_type>(colmap, tpetra_pmap, input_crs_graph.get_kokkos_dual_view().d_view, nullptr);
    tpetra_column_pmap = colmap;
    size_t nnz = input_crs_graph.size();

    //debug print
    //std::cout << "DOF GRAPH SIZE ON RANK " << myrank << " IS " << nnz << std::endl;
    
    //local indices in the graph using the constructed column map
    crs_local_indices_ = indices_array("crs_local_indices", nnz);
    
    //row offsets with compatible template arguments
        row_map_type row_offsets_pass("row_offsets", dim1_ + 1);
        for(int ipass = 0; ipass < dim1_ + 1; ipass++){
            row_offsets_pass(ipass) = input_values.start_index_(ipass);
        }

    size_t entrycount = 0;
    for(int irow = 0; irow < dim1_; irow++){
        for(int istride = 0; istride < mystrides_(irow); istride++){
            crs_local_indices_(entrycount) = tpetra_column_pmap->getLocalElement(crs_graph(entrycount));
            entrycount++;
        }
    }
    
    //sort values and indices
    Tpetra::Import_Util::sortCrsEntries<row_map_type, indices_array, values_array>(row_offsets_pass, crs_local_indices_.d_view, this_array_.get_kokkos_view());

    tpetra_crs_matrix = Teuchos::rcp(new MAT(tpetra_pmap, tpetra_column_pmap, start_index_.d_view, crs_local_indices_.d_view, this_array_.get_kokkos_view()));
    tpetra_crs_matrix->fillComplete();
}

// Constructor that takes local data in a matar ragged type
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::TpetraCRSMatrix(TpetraPartitionMap<ExecSpace,MemoryTraits> &input_pmap, input_row_map_type input_strides,
                                                                  input_row_graph_type crs_graph, TArray1D input_values,
                                                                  const std::string& tag_string) {
    mpi_comm_ = input_pmap.mpi_comm_;
    global_dim1_ = input_pmap.num_global_;
    Teuchos::RCP<const Teuchos::Comm<int>> teuchos_comm = Teuchos::rcp(new Teuchos::MpiComm<int>(mpi_comm_));
    pmap = input_pmap;
    tpetra_pmap = pmap.tpetra_map;
    dim1_ = tpetra_pmap->getLocalNumElements();
    mystrides_ = input_strides.get_kokkos_dual_view().d_view;
    this_array_ = input_values;
    global_indices_array input_crs_graph = crs_graph.get_kokkos_view();

    
    //build column map for the global conductivity matrix
    Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type> > colmap;
    const Teuchos::RCP<const Tpetra::Map<tpetra_LO, tpetra_GO, tpetra_node_type> > dommap = tpetra_pmap;

    Tpetra::Details::makeColMap<tpetra_LO, tpetra_GO, tpetra_node_type>(colmap, tpetra_pmap, input_crs_graph, nullptr);
    tpetra_column_pmap = colmap;
    size_t nnz = crs_graph.size();

    //debug print
    //std::cout << "DOF GRAPH SIZE ON RANK " << myrank << " IS " << nnz << std::endl;
    
    //local indices in the graph using the constructed column map
    crs_local_indices_ = indices_array("crs_local_indices", nnz);
    
    //row offsets with compatible template arguments
        row_map_type row_offsets_pass("row_offsets", dim1_ + 1);
        for(int ipass = 0; ipass < dim1_ + 1; ipass++){
            row_offsets_pass(ipass) = input_values.start_index_(ipass);
        }

    size_t entrycount = 0;
    for(int irow = 0; irow < dim1_; irow++){
        for(int istride = 0; istride < mystrides_(irow); istride++){
            crs_local_indices_(entrycount) = tpetra_column_pmap->getLocalElement(input_crs_graph(entrycount));
            entrycount++;
        }
    }
    
    //sort values and indices
    Tpetra::Import_Util::sortCrsEntries<row_map_type, indices_array, values_array>(row_offsets_pass, crs_local_indices_, this_array_.get_kokkos_view());

    tpetra_crs_matrix = Teuchos::rcp(new MAT(tpetra_pmap, tpetra_column_pmap, input_values.start_index_, crs_local_indices_, this_array_.get_kokkos_view()));
    tpetra_crs_matrix->fillComplete();
}

//select MPI datatype
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::set_mpi_type() {
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
        printf("Your entered TpetraCRSMatrix type is not a supported type for MPI communications and is being set to int\n");
        mpi_datatype_ = MPI_INT;
    }
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T& TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::operator()(size_t i, size_t j) const {
    assert(i >= 0 && i < dim1_ && "i is out of bounds in TpetraCRSMatrix!");
    assert(j >= 0 && j < mystrides_(i) && "j is out of bounds in TpetraCRSMatrix!");
    return this_array_(i,j);
}

// Return global index corresponding to the input local (on this process/rank) index for the sub map this vector comms from
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
long long int TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::getColumnMapGlobalIndex(int local_index) const {
    long long int global_index = tpetra_column_pmap->getGlobalElement(local_index);
    return global_index;
}

// Return global index corresponding to the input local (on this process/rank) index
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
long long int TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::getMapGlobalIndex(int local_index) const {
    long long int global_index = tpetra_pmap->getGlobalElement(local_index);
    return global_index;
}

// Return global index corresponding to the input local (on this process/rank) index for the sub map this vector comms from
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
int TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::getColumnMapLocalIndex(long long int global_index) const {
    int local_index = tpetra_column_pmap->getLocalElement(global_index);
    return local_index;
}

// Return global index corresponding to the input local (on this process/rank) index
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
int TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::getMapLocalIndex(long long int global_index) const {
    int local_index = tpetra_pmap->getLocalElement(global_index);
    return local_index;
}

// template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
// T& TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::host(size_t i, size_t j) const {
//     assert(i >= 0 && i < dim1_ && "i is out of bounds in TpetraCRSMatrix");
//     assert(j >= 0 && j < mystrides_(i) && "j is out of bounds in TpetraCRSMatrix");
//     return this_array_.h_view(i,j);
// }

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>& TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::operator= (const TpetraCRSMatrix& temp) {
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        dim1_ = temp.dim1_;
        mystrides_ = temp.mystrides_;
        start_index_ = temp.start_index_;
        crs_local_indices_ = temp.crs_local_indices_;
        global_dim1_ = temp.global_dim1_;
        length_ = temp.length_;
        this_array_ = temp.this_array_;
        mpi_comm_ = temp.mpi_comm_;
        mpi_datatype_ = temp.mpi_datatype_;
        tpetra_crs_matrix = temp.tpetra_crs_matrix;
        pmap = temp.pmap;
        column_pmap = temp.column_pmap;
        comm_pmap = temp.comm_pmap;
        tpetra_pmap = temp.tpetra_pmap;
        tpetra_column_pmap = temp.tpetra_column_pmap;
        tpetra_comm_pmap = temp.tpetra_comm_pmap;
        importer = temp.importer;
        own_comms = temp.own_comms;
        column_map_size_ = temp.column_map_size_;
    }
    
    return *this;
}

// Return size
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::size() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::extent() const {
    return length_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
size_t TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::dim1() const {
    return dim1_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
size_t TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::global_dim() const {
    return global_dim1_;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
T* TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::device_pointer() const {
    return this_array_.pointer();
}

// template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
// KOKKOS_INLINE_FUNCTION
// T* TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::host_pointer() const {
//     return this_array_.h_view.data();
// }

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
Kokkos::View <T**, Layout, ExecSpace, MemoryTraits> TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::get_kokkos_view() const {
  return this_array_.get_kokkos_view();
}

// template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
// void TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::update_host() {

//     this_array_.template modify<typename TArray1D::execution_space>();
//     this_array_.template sync<typename TArray1D::host_mirror_space>();
// }

// template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
// void TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::update_device() {

//     this_array_.template modify<typename TArray1D::host_mirror_space>();
//     this_array_.template sync<typename TArray1D::execution_space>();
// }

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::print() const {
        std::ostream &out = std::cout;
        Teuchos::RCP<Teuchos::FancyOStream> fos;
        fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(out));
        tpetra_crs_matrix->describe(*fos,Teuchos::VERB_EXTREME);
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
TpetraCRSMatrix<T,Layout,ExecSpace,MemoryTraits>::~TpetraCRSMatrix() {}

////////////////////////////////////////////////////////////////////////////////
// End of TpetraCRSMatrix
////////////////////////////////////////////////////////////////////////////////

/////////////////////////
/* TpetraTpetraCommunicationPlan:  Class storing relevant data and functions to perform comms between two different Tpetra MATAR MPI types.
                       The object for this class should not be reconstructed if the same comm plan is needed repeatedly; the setup is expensive.
                       The comms routines such as execute_comms can be called repeatedly to avoid repeated setup of the plan.*/
/////////////////////////
template <typename T, typename Layout = tpetra_array_layout, typename ExecSpace = tpetra_execution_space, typename MemoryTraits = tpetra_memory_traits>
class TpetraCommunicationPlan {
    
protected:
    TpetraDFArray<T, Layout, ExecSpace, MemoryTraits> destination_vector_;
    TpetraDFArray<T, Layout, ExecSpace, MemoryTraits> source_vector_;

    /*forward comms means communicating data to a vector that doesn't have a unique distribution of its global
      indices amongst processes from a vector that does have a unique distribution amongst processes.
      An example of forward comms in a finite element application would be communicating ghost data from 
      the vector of local data.

      reverse comms means communicating data to a vector that has a unique distribution of its global
      indices amongst processes from a vector that does not have a unique distribution amongst processes.
      An example of reverse comms in a finite element application would be communicating force contributions from ghost
      indices via summation to the entries of the uniquely owned vector that stores final tallies of forces.
    */
    bool reverse_comms_flag; //default is false
    Teuchos::RCP<Tpetra::Import<tpetra_LO, tpetra_GO>> importer; // tpetra comm object
    Teuchos::RCP<Tpetra::Export<tpetra_LO, tpetra_GO>> exporter; // tpetra reverse comm object

public:
    
    enum combine_mode { INSERT, SUM, ABSMAX, REPLACE, MIN, ADD_REPLACE };
    combine_mode combine_mode_;

    TpetraCommunicationPlan();

    //Copy Constructor
    TpetraCommunicationPlan(const TpetraCommunicationPlan<T, Layout, ExecSpace,MemoryTraits> &temp){
        *this = temp;
    }
    
    TpetraCommunicationPlan(TpetraDFArray<T, Layout, ExecSpace, MemoryTraits> destination_vector,
                            TpetraDFArray<T, Layout, ExecSpace, MemoryTraits> source_vector, bool reverse_comms=false, combine_mode mode=INSERT);

    KOKKOS_INLINE_FUNCTION
    TpetraCommunicationPlan& operator=(const TpetraCommunicationPlan& temp);

    // Deconstructor
    virtual KOKKOS_INLINE_FUNCTION
    ~TpetraCommunicationPlan ();

    void execute_comms();
}; // End of TpetraCommunicationPlan


// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraCommunicationPlan<T,Layout,ExecSpace,MemoryTraits>::TpetraCommunicationPlan() {
    
}

// Overloaded 1D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraCommunicationPlan<T,Layout,ExecSpace,MemoryTraits>::TpetraCommunicationPlan(TpetraDFArray<T, Layout, ExecSpace, MemoryTraits> destination_vector,
                            TpetraDFArray<T, Layout, ExecSpace, MemoryTraits> source_vector, bool reverse_comms, combine_mode mode) {
    combine_mode_ = mode;
    reverse_comms_flag = reverse_comms;
    destination_vector_ = destination_vector;
    source_vector_ = source_vector;

    //setup Tpetra comm object
    if(reverse_comms){
        // create export object; completes setup
        exporter = Teuchos::rcp(new Tpetra::Export<tpetra_LO, tpetra_GO>(source_vector_.tpetra_pmap, destination_vector_.tpetra_pmap));
    }
    else{
        // create import object; completes setup
        importer = Teuchos::rcp(new Tpetra::Import<tpetra_LO, tpetra_GO>(source_vector_.tpetra_pmap, destination_vector_.tpetra_pmap));
    }
}


template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
TpetraCommunicationPlan<T,Layout,ExecSpace,MemoryTraits>& TpetraCommunicationPlan<T,Layout,ExecSpace,MemoryTraits>::operator= (const TpetraCommunicationPlan& temp) {
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        reverse_comms_flag = temp.reverse_comms_flag;
        combine_mode_ = temp.combine_mode_;
        destination_vector_ = temp.destination_vector_;
        source_vector_ = temp.source_vector_;
        if(reverse_comms_flag){
            exporter = temp.exporter;
        }
        else{
            importer = temp.importer;
        }
    }
    
    return *this;
}

//perform comms
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraCommunicationPlan<T,Layout,ExecSpace,MemoryTraits>::execute_comms(){
    if(reverse_comms_flag){
        destination_vector_.tpetra_vector->doExport(*(source_vector_.tpetra_vector), *exporter, Tpetra::INSERT, true);\
        if(destination_vector_.this_array_.template need_sync<typename decltype(destination_vector_)::TArray1D::execution_space>()){
            destination_vector_.this_array_.template sync<typename decltype(destination_vector_)::TArray1D::execution_space>();
        }
        else{
            destination_vector_.this_array_.template sync<typename decltype(destination_vector_)::TArray1D::host_mirror_space>();
        }
    }
    else{
        destination_vector_.tpetra_vector->doImport(*(source_vector_.tpetra_vector), *importer, Tpetra::INSERT);
        if(destination_vector_.this_array_.template need_sync<typename decltype(destination_vector_)::TArray1D::execution_space>()){
            destination_vector_.this_array_.template sync<typename decltype(destination_vector_)::TArray1D::execution_space>();
        }
        else{
            destination_vector_.this_array_.template sync<typename decltype(destination_vector_)::TArray1D::host_mirror_space>();
        }
    }
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
TpetraCommunicationPlan<T,Layout,ExecSpace,MemoryTraits>::~TpetraCommunicationPlan() {}

////////////////////////////////////////////////////////////////////////////////
// End of TpetraCommunicationPlan
////////////////////////////////////////////////////////////////////////////////

/////////////////////////
/* TpetraTpetraLRCommunicationPlan:  Class storing relevant data and functions to perform comms between two different Tpetra MATAR MPI types.
                       The object for this class should not be reconstructed if the same comm plan is needed repeatedly; the setup is expensive.
                       The comms routines such as execute_comms can be called repeatedly to avoid repeated setup of the plan.*/
/////////////////////////
template <typename T, typename Layout = Kokkos::LayoutRight, typename ExecSpace = tpetra_execution_space, typename MemoryTraits = tpetra_memory_traits>
class TpetraLRCommunicationPlan {
    
protected:
    TpetraDCArray<T, Layout, ExecSpace, MemoryTraits> destination_vector_;
    TpetraDCArray<T, Layout, ExecSpace, MemoryTraits> source_vector_;

    /*forward comms means communicating data to a vector that doesn't have a unique distribution of its global
      indices amongst processes from a vector that does have a unique distribution amongst processes.
      An example of forward comms in a finite element application would be communicating ghost data from 
      the vector of local data.

      reverse comms means communicating data to a vector that has a unique distribution of its global
      indices amongst processes from a vector that does not have a unique distribution amongst processes.
      An example of reverse comms in a finite element application would be communicating force contributions from ghost
      indices via summation to the entries of the uniquely owned vector that stores final tallies of forces.
    */
    bool reverse_comms_flag; //default is false
    Teuchos::RCP<Tpetra::Import<tpetra_LO, tpetra_GO>> importer; // tpetra comm object
    Teuchos::RCP<Tpetra::Export<tpetra_LO, tpetra_GO>> exporter; // tpetra reverse comm object

public:
    
    enum combine_mode { INSERT, SUM, ABSMAX, REPLACE, MIN, ADD_REPLACE };
    combine_mode combine_mode_;

    TpetraLRCommunicationPlan();

    //Copy Constructor
    TpetraLRCommunicationPlan(const TpetraLRCommunicationPlan<T, Layout, ExecSpace,MemoryTraits> &temp){
        *this = temp;
    }
    
    TpetraLRCommunicationPlan(TpetraDCArray<T, Layout, ExecSpace, MemoryTraits> destination_vector,
                            TpetraDCArray<T, Layout, ExecSpace, MemoryTraits> source_vector, bool reverse_comms=false, combine_mode mode=INSERT);

    KOKKOS_INLINE_FUNCTION
    TpetraLRCommunicationPlan& operator=(const TpetraLRCommunicationPlan& temp);

    // Deconstructor
    virtual KOKKOS_INLINE_FUNCTION
    ~TpetraLRCommunicationPlan ();

    void execute_comms();
}; // End of TpetraLRCommunicationPlan


// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraLRCommunicationPlan<T,Layout,ExecSpace,MemoryTraits>::TpetraLRCommunicationPlan() {
    
}

// Overloaded 1D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
TpetraLRCommunicationPlan<T,Layout,ExecSpace,MemoryTraits>::TpetraLRCommunicationPlan(TpetraDCArray<T, Layout, ExecSpace, MemoryTraits> destination_vector,
                            TpetraDCArray<T, Layout, ExecSpace, MemoryTraits> source_vector, bool reverse_comms, combine_mode mode) {
    combine_mode_ = mode;
    reverse_comms_flag = reverse_comms;
    destination_vector_ = destination_vector;
    source_vector_ = source_vector;

    //setup Tpetra comm object
    if(reverse_comms){
        // create export object; completes setup
        exporter = Teuchos::rcp(new Tpetra::Export<tpetra_LO, tpetra_GO>(source_vector_.tpetra_pmap, destination_vector_.tpetra_pmap));
    }
    else{
        // create import object; completes setup
        importer = Teuchos::rcp(new Tpetra::Import<tpetra_LO, tpetra_GO>(source_vector_.tpetra_pmap, destination_vector_.tpetra_pmap));
    }
}


template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
TpetraLRCommunicationPlan<T,Layout,ExecSpace,MemoryTraits>& TpetraLRCommunicationPlan<T,Layout,ExecSpace,MemoryTraits>::operator= (const TpetraLRCommunicationPlan& temp) {
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        reverse_comms_flag = temp.reverse_comms_flag;
        combine_mode_ = temp.combine_mode_;
        destination_vector_ = temp.destination_vector_;
        source_vector_ = temp.source_vector_;
        if(reverse_comms_flag){
            exporter = temp.exporter;
        }
        else{
            importer = temp.importer;
        }
    }
    
    return *this;
}

//perform comms
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
void TpetraLRCommunicationPlan<T,Layout,ExecSpace,MemoryTraits>::execute_comms(){
    if(reverse_comms_flag){
        destination_vector_.tpetra_vector->doExport(*(source_vector_.tpetra_vector), *exporter, Tpetra::INSERT, true);
        if(destination_vector_.this_array_.template need_sync<typename decltype(destination_vector_)::TArray1D::execution_space>()){
            destination_vector_.this_array_.template sync<typename decltype(destination_vector_)::TArray1D::execution_space>();
        }
        else{
            destination_vector_.this_array_.template sync<typename decltype(destination_vector_)::TArray1D::host_mirror_space>();
        }
    }
    else{
        destination_vector_.tpetra_vector->doImport(*(source_vector_.tpetra_vector), *importer, Tpetra::INSERT);
        if(destination_vector_.this_array_.template need_sync<typename decltype(destination_vector_)::TArray1D::execution_space>()){
            destination_vector_.this_array_.template sync<typename decltype(destination_vector_)::TArray1D::execution_space>();
        }
        else{
            destination_vector_.this_array_.template sync<typename decltype(destination_vector_)::TArray1D::host_mirror_space>();
        }
    }
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
TpetraLRCommunicationPlan<T,Layout,ExecSpace,MemoryTraits>::~TpetraLRCommunicationPlan() {}

////////////////////////////////////////////////////////////////////////////////
// End of TpetraLRCommunicationPlan
////////////////////////////////////////////////////////////////////////////////

} // end namespace

#endif // end if have MPI
#endif // end if TRILINOS_INTERFACE

#endif // TPETRA_WRAPPER_TYPES_H

