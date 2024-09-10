#ifndef COMMUNICATION_PLAN_H
#define COMMUNICATION_PLAN_H
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
#include "partition_map.h"

namespace mtr
{

/////////////////////////
/* CommunicationPlan:  Class storing relevant data and functions to perform comms between two different MATAR MPI types.
                       The object for this class should not be reconstructed if the same comm plan is needed repeatedly; the setup is expensive.
                       The comms routines such as execute_comms can be called repeatedly to avoid repeated setup of the plan.*/
/////////////////////////
template <typename T, typename Layout = DefaultLayout, typename ExecSpace = DefaultExecSpace, typename MemoryTraits = void>
class CommunicationPlan {

    // this is manage
    using TArray1D = Kokkos::DualView <T*, Layout, ExecSpace, MemoryTraits>;
    
protected:

public:
    
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

    CommunicationPlan();

    //Copy Constructor
    CommunicationPlan(const CommunicationPlan<T, Layout, ExecSpace,MemoryTraits> &temp){
        *this = temp;
    }
    
    CommunicationPlan(bool reverse_comms);

    KOKKOS_INLINE_FUNCTION
    CommunicationPlan& operator=(const CommunicationPlan& temp);

    // Deconstructor
    KOKKOS_INLINE_FUNCTION
    ~CommunicationPlan ();

    virtual execute_comms(){}
}; // End of CommunicationPlan


// Default constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CommunicationPlan<T,Layout,ExecSpace,MemoryTraits>::CommunicationPlan() {
    
}

// Overloaded 1D constructor
template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
CommunicationPlan<T,Layout,ExecSpace,MemoryTraits>::CommunicationPlan(bool reverse_comms) {
    reverse_comms_flag = reverse_comms;
}


template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
CommunicationPlan<T,Layout,ExecSpace,MemoryTraits>& CommunicationPlan<T,Layout,ExecSpace,MemoryTraits>::operator= (const CommunicationPlan& temp) {
    
    // Do nothing if the assignment is of the form x = x
    if (this != &temp) {
        reverse_comms_flag = reverse_comms_flag;
    }
    
    return *this;
}

template <typename T, typename Layout, typename ExecSpace, typename MemoryTraits>
KOKKOS_INLINE_FUNCTION
CommunicationPlan<T,Layout,ExecSpace,MemoryTraits>::~CommunicationPlan() {}

////////////////////////////////////////////////////////////////////////////////
// End of CommunicationPlan
////////////////////////////////////////////////////////////////////////////////

} // end namespace

#endif // end if have MPI

#endif // COMMUNICATION_PLAN_H
