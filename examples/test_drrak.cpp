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
#include <array>
#include <variant>
#include <chrono>

#include "matar.h"

using namespace mtr; // matar namespace




// =============================================================
//
// Main function
//
int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {
       
        // -----------------------
        // DDynamicRaggedRightArray
        // -----------------------

        printf("\nDual Dynamic Ragged Right Array test 1D \n");
        DRaggedRightArrayKokkos<int> ddrrak1D;

        // testing ragged initialized with CArrayKokkos for strides
        CArrayKokkos<size_t> some_strides(4);

        // create a lower-triangular array
        RUN({
            some_strides(0) = 1;
            some_strides(1) = 3;
            some_strides(2) = 5;
            some_strides(3) = 7;
        });


        ddrrak1D = DRaggedRightArrayKokkos<int>(some_strides, "test_1D");







        // printf("\nDual Dynamic Ragged Right Array test 2D \n");
        // DRaggedRightArrayKokkos<int> ddrrak2D;

        // // testing ragged initialized with CArrayKokkos for strides
        // CArrayKokkos<size_t> some_strides(4);

        // // create a lower-triangular array
        // RUN({
        //     some_strides(0) = 1;
        //     some_strides(1) = 3;
        //     some_strides(2) = 5;
        //     some_strides(3) = 7;
        // });


        // ddrrak2D = DDynamicRaggedRightArrayKokkos<int>(some_strides, 9, "test_2D");







        // printf("\nDual Dynamic Ragged Right Array test 3D \n");
        // DDynamicRaggedRightArrayKokkos<int> ddrrak3D;

        // // testing ragged initialized with CArrayKokkos for strides
        // CArrayKokkos<size_t> some_strides(4);

        // // create a lower-triangular array
        // RUN({
        //     some_strides(0) = 1;
        //     some_strides(1) = 3;
        //     some_strides(2) = 5;
        //     some_strides(3) = 7;
        // });


        // ddrrak3D = DDynamicRaggedRightArrayKokkos<int>(some_strides, 3, 3, "test_3D");

        // Kokkos::parallel_for("DDRRAKTest", size_i, KOKKOS_LAMBDA(const int i) {
        //     for (int j = 0; j < (i % size_j) + 1; j++) {
        //         ddrrak.stride(i)++;
        //         ddrrak(i, j) = j;
        //         // printf("(%i) stride is %d\n", i, j);
        //     }
        // });
        // Kokkos::fence();

        // printf("\ntesting macro FOR_ALL\n");

        // // testing MATAR FOR_ALL loop
        // DDynamicRaggedRightArrayKokkos<int> my_ddyn_ragged(size_i, size_j);
        // FOR_ALL(i, 0, size_i, {
        //     for (int j = 0; j <= (i % size_j); j++) {
        //         my_ddyn_ragged.stride(i)++;
        //         my_ddyn_ragged(i, j) = j;
        //         printf(" ddyn_ragged_right error = %i \n", my_ddyn_ragged(i, j) - ddrrak(i, j));
        //     } // end for
        // }); // end parallel for
        Kokkos::fence();

       

    } // end of kokkos scope

    Kokkos::finalize();

    printf("\nfinished\n\n");

    return 0;
}

