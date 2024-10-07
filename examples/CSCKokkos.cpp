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
#include <math.h>
#include <matar.h>
#include <limits.h>

using namespace mtr; // matar namespace

int main(int argc, char* argv[])
{
    Kokkos::initialize(); {
        size_t nnz  = 6;
        size_t dim1 = 3;
        size_t dim2 = 10;
        CArrayKokkos<size_t> starts(dim2 + 1);
        CArrayKokkos<size_t> rows(nnz);
        CArrayKokkos<int>    array(nnz + 1);
        RUN({
            starts(1) = 1;
            starts(2) = 2;
            starts(3) = 3;
            starts(4) = 4;
            starts(5) = 5;
            starts(6) = 6;
            starts(7) = 6;
            starts(8) = 6;
            starts(9) = 6;

            rows(0) = 0;
            rows(1) = 0;
            rows(2) = 1;
            rows(3) = 1;
            rows(4) = 2;
            rows(5) = 2;

            array(0) = 1;
            array(1) = 2;
            array(2) = 3;
            array(3) = 4;
            array(4) = 5;
            array(5) = 6;
            array(6) = 0;
        });

        /*
        |1 2 2 0 0 0 0 0 0 0|
        |0 0 3 4 0 0 0 0 0 0|
        |0 0 0 0 5 6 0 0 0 0|
        */

        const std::string s = "hello";
        // Testing = op
        auto pre_A   = CSCArrayKokkos<int>(array, starts, rows, dim1, dim2, s);
        auto A       = pre_A;
        int* values  = A.pointer();
        auto a_start = A.get_starts();
        int  total   = 0;

        RUN({
            printf("This matix is %ld x %ld \n", A.dim1(), A.dim2());
        });

        RUN({
            printf("nnz : %ld \n", A.nnz());
        });

        int loc_total = 0;
        loc_total += 0; // Get rid of warning
        FOR_REDUCE_SUM(i, 0, nnz,
                       loc_total, {
                loc_total += values[i];
        }, total);
        printf("Sum of nnz from pointer method %d\n", total);
        total = 0;
        FOR_REDUCE_SUM(i, 0, nnz,
                       loc_total, {
                loc_total += a_start[i];
        }, total);
        printf("Sum of start indices form .get_starts() %d\n", total);
        total = 0;

        FOR_REDUCE_SUM(i, 0, dim1,
                       j, 0, dim2 - 1,
            loc_total, {
                loc_total += A(i, j);
        }, total);
        printf("Sum of nnz in array notation %d\n", total);
    } Kokkos::finalize();
    return 0;
}
