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

using namespace mtr;  // matar namespace

// =============================================================
//
// RaggedRightArrayofVectorsKokkos example: row i holds stride(i)
// vectors, each of length vector_dim, accessed as array(i, j, k)
//
int main(int argc, char* argv[]) {
    int status = 0;
    Kokkos::initialize(argc, argv);
    {
        const size_t num_rows   = 4;
        const size_t vector_dim = 3;

        // strides are 1, 2, 3, 4 vectors per row; filled on device
        CArrayKokkos<size_t> strides(num_rows, "strides");
        FOR_ALL(i, 0, num_rows, {
            strides(i) = i + 1;
        });
        Kokkos::fence();

        RaggedRightArrayofVectorsKokkos<double> rrav(strides, vector_dim, "rrav");

        printf("RaggedRightArrayofVectorsKokkos: %zu rows, vector_dim %zu, %zu entries\n",
               num_rows, vector_dim, rrav.size());

        // stride() reads device memory, so it is only called inside device loops
        FOR_ALL(i, 0, num_rows, {
            for (size_t j = 0; j < rrav.stride(i); j++) {
                for (size_t k = 0; k < vector_dim; k++) {
                    rrav(i, j, k) = 100.0 * i + 10.0 * j + k;
                }
            }
        });
        Kokkos::fence();

        RUN({
            printf("row 2, vector 1 = (%g, %g, %g)\n", rrav(2, 1, 0), rrav(2, 1, 1), rrav(2, 1, 2));
        });
        Kokkos::fence();

        double loc_sum = 0.0;
        double sum     = 0.0;
        FOR_REDUCE_SUM(i, 0, num_rows,
                       loc_sum, {
            for (size_t j = 0; j < rrav.stride(i); j++) {
                for (size_t k = 0; k < vector_dim; k++) {
                    loc_sum += rrav(i, j, k);
                }
            }
        }, sum);

        // the same sum computed on the host from the known strides
        double expected      = 0.0;
        size_t expected_size = 0;
        for (size_t i = 0; i < num_rows; i++) {
            for (size_t j = 0; j < i + 1; j++) {
                for (size_t k = 0; k < vector_dim; k++) {
                    expected += 100.0 * i + 10.0 * j + k;
                    expected_size++;
                }
            }
        }

        printf("sum = %g (expected %g)\n", sum, expected);

        // views must be destroyed before Kokkos::finalize, so report via status
        status = (sum == expected && rrav.size() == expected_size) ? 0 : 1;
        printf(status == 0 ? "PASSED\n" : "FAILED\n");
    }
    Kokkos::finalize();
    return status;
}
