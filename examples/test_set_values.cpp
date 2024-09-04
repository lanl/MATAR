/**********************************************************************************************
 � 2020. Triad National Security, LLC. All rights reserved.
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
#include <stdio.h>
#include "matar.h"

using namespace mtr; // matar namespace

int main(int argc, char* argv[])
{
    // Test RaggedRightArrayKokkos
    Kokkos::initialize(argc, argv);
    {
        // Create a CArrayKokkos for strides
        CArrayKokkos<size_t, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<0>> strides(3);
        
        // Set up strides (this is just an example, adjust as needed)
        Kokkos::parallel_for("SetStrides", 1, KOKKOS_LAMBDA(const int&) {
            strides(0) = 3;  // dim0
            strides(1) = 9;  // total elements in dim1
            strides(2) = 18; // total elements overall
        });

        // Create RaggedRightArrayKokkos using the new constructor
        RaggedRightArrayKokkos<double, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<0>, Kokkos::LayoutRight> ragged_array(strides, "MyRaggedArray");

        // Use the ragged array...
        ragged_array.set_values(3.14);
        ragged_array.print();
    }
    
    // Test RaggedDownArrayKokkos
    {
        // Create a CArrayKokkos for strides
        CArrayKokkos<size_t, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<0>> strides(3);
        
        // Set up strides (this is just an example, adjust as needed)
        Kokkos::parallel_for("SetStrides", 1, KOKKOS_LAMBDA(const int&) {
            strides(0) = 2;  // dim0
            strides(1) = 6;  // total elements in dim1
            strides(2) = 12; // total elements overall
        });

        // Create RaggedRightArrayKokkos using the new constructor
        RaggedDownArrayKokkos<double, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace, Kokkos::MemoryTraits<0>, Kokkos::LayoutRight> ragged_array(strides, "MyRaggedArray");

        // Use the ragged array...
        ragged_array.set_values(5.67);
        ragged_array.print();
    }
    Kokkos::finalize();
    return 0;
}
