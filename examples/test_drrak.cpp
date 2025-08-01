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
        // DRaggedRightArray Scalar with CArrayKokkos
        // -----------------------

        printf("\nDRaggedRightArray test 1D \n");
        DRaggedRightArrayKokkos<int> drrak1D;

        // testing ragged initialized with CArrayKokkos for strides
        int num_strides = 3;
        CArrayKokkos<size_t> some_strides(num_strides, "test_1D_strides");

        FOR_ALL(i, 0, num_strides, {
            some_strides(i) = i+1;
        });

        Kokkos::fence();

        drrak1D = DRaggedRightArrayKokkos<int>(some_strides, "test_1D");
        drrak1D.update_host();

        std::cout << "Array length: " << drrak1D.size() << std::endl;
        FOR_ALL(i, 0, num_strides,{
            for(int j = 0; j < drrak1D.stride(i); j++) {
                drrak1D(i, j) = j;
            }
        });

        drrak1D.update_host();

        for(int i = 0; i < num_strides; i++) {
            for(int j = 0; j < drrak1D.stride_host(i); j++) {
                if(drrak1D.host(i, j) != j) {
                    printf("Error: drrak1D(i, j) = %d, expected %d\n", drrak1D.host(i, j), j);
                }
            }
        }
        std::cout << "test_1D passed" << std::endl;
        Kokkos::fence();


        // -----------------------
        // DRaggedRightArray Scalar with DCArrayKokkos
        // -----------------------

        printf("\nDRaggedRightArray test 1D \n");
        DRaggedRightArrayKokkos<int> drrak1D_d;

        // testing ragged initialized with CArrayKokkos for strides
        num_strides = 3;
        DCArrayKokkos<size_t> some_strides_d(num_strides, "test_1D_strides_d");

        for(int i = 0; i < num_strides; i++){
            some_strides_d.host(i) = i+1;
        }

        some_strides_d.update_device();

        Kokkos::fence();

        drrak1D_d = DRaggedRightArrayKokkos<int>(some_strides_d, "test_1D_dual");
        drrak1D_d.update_host();

        std::cout << "Array length: " << drrak1D_d.size() << std::endl;

        for(int i = 0; i < num_strides; i++){
            if(drrak1D_d.stride_host(i) != i+1){
                printf("Error: drrak1D_d.stride_host(i) = %zu, expected %d\n", drrak1D_d.stride_host(i), i+1);
            }
        }


        FOR_ALL(i, 0, num_strides,{
            for(int j = 0; j < drrak1D_d.stride(i); j++) {
                drrak1D_d(i, j) = j;
            }
        });

        drrak1D_d.update_host();

        for(int i = 0; i < num_strides; i++) {
            for(int j = 0; j < drrak1D_d.stride_host(i); j++) {
                if(drrak1D_d.host(i, j) != j) {
                    printf("Error: drrak1D_d(i, j) = %d, expected %d\n", drrak1D_d.host(i, j), j);
                }
            }
        }
        std::cout << "test_1D dual input passed" << std::endl;
        Kokkos::fence();


        // -----------------------
        // DRaggedRightArray Vector
        // -----------------------

        printf("\nDRaggedRightArray test 2D \n");
        DRaggedRightArrayKokkos<int> drrak2D;
        size_t dim2D = 3;

        drrak2D = DRaggedRightArrayKokkos<int>(some_strides, dim2D, "test_2D");

        FOR_ALL(i, 0, num_strides,{
            for(int j = 0; j < drrak2D.stride(i); j++) {
                for(int k = 0; k < dim2D; k++) {

                    drrak2D(i, j, k) = j+k;
                }
            }
        });
        Kokkos::fence();
        drrak2D.update_host();
        Kokkos::fence();

        for(int i = 0; i < num_strides; i++) {
            for(int j = 0; j < drrak2D.stride_host(i); j++) {
                for(int k = 0; k < dim2D; k++) {
                    if(drrak2D.host(i, j, k) != j+k) {
                        printf("Error: drrak2D(i, j, k) = %d, expected %d\n", drrak2D.host(i, j, k), j+k);
                    }
                }
            }
        }
        std::cout << "test_2D passed" << std::endl;
        Kokkos::fence();


        // -----------------------
        // DRaggedRightArray Tensor
        // -----------------------

        printf("\nDRaggedRightArray test 3D \n");
        DRaggedRightArrayKokkos<int> drrak3D;


        drrak3D = DRaggedRightArrayKokkos<int>(some_strides, dim2D, dim2D, "test_3D");


        FOR_ALL(i, 0, num_strides,{
            for(int j = 0; j < drrak3D.stride(i); j++) {
                for(int k = 0; k < dim2D; k++) {
                    for(int l = 0; l < dim2D; l++) {
                        drrak3D(i, j, k, l) = j+k+l;
                    }
                }
            }
        });

        drrak3D.update_host();

        for(int i = 0; i < num_strides; i++) {
            for(int j = 0; j < drrak3D.stride_host(i); j++) {
                for(int k = 0; k < dim2D; k++) {
                    for(int l = 0; l < dim2D; l++) {
                        if(drrak3D.host(i, j, k, l) != j+k+l) {
                            printf("Error: drrak3D(i, j, k, l) = %d, expected %d\n", drrak3D.host(i, j, k, l), j+k+l);
                        }
                    }
                }
            }
        }
        std::cout << "test_3D passed" << std::endl;
        Kokkos::fence();

    } // end of kokkos scope

    Kokkos::finalize();

    printf("\nfinished\n\n");

    return 0;
}

