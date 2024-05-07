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
#include <iostream>
#include "inherited_inits.hpp"

int main()
{
    // Kokkos GPU test

    Kokkos::initialize();
    {
        int num_parent = 2; // number of materials
        Parent1D parent("parent", num_parent); // Initialize Kokkos View on the GPU of type material, size num_parent
        auto     h_parent = Kokkos::create_mirror_view(parent); // Create a host view of the Kokkos View

        AllocateHost(h_parent, 0, BABY2_SIZE); // Function performed on Host to do raw Kokkos allocation of baby2 GPU space inside of Host data structure
        AllocateHost(h_parent, 1, BABY1_SIZE); // Function performed on Host to do raw Kokkos allocation of baby1 GPU space inside of Host data structure

        Kokkos::deep_copy(parent, h_parent); // deep copy Host data (allocated above) to the GPU Kokkos View. GPU View now has the class space allocated

        InitChildModels(parent, 0, baby2{}); // Kokkos Function to create new instances of the baby2 model on the GPU
        InitChildModels(parent, 1, baby1{ 1.4, 1.0 }); // Kokkos Function to create new instances of the baby1 models on the GPU

        // Model test, also shows a Kokkos reduction
        double value_1;
        Kokkos::parallel_reduce(
        "CheckValues",
        num_parent,
        KOKKOS_LAMBDA(const int idx, real_t & lsum)
        {
             lsum += parent(idx).child->math(2.0, 4.0);
        }
            , value_1);

        printf("value %f\n", value_1);

        ClearDeviceModels(parent); // Kokkos Function to call deconstructors of objects on the GPU

        FreeHost(h_parent); // Function performed on Host to free the allocated GPU classes inside of the Host mirror
    }
    Kokkos::finalize();

    printf("--- finished ---\n");

    return 0;
}
