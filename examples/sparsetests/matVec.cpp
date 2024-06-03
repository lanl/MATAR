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
#include <stdio.h>
#include <math.h>
#include <matar.h>
#include <limits.h>
#include <chrono>
#include <time.h>

#define EXPORT true

using namespace mtr; // matar namespace

void matVec(CArrayKokkos<double>& A, CArrayKokkos<double>& v, CArrayKokkos<double>& b)
{
    size_t n = A.dims(0);
    size_t m = A.dims(1);
    FOR_ALL(i, 0, n,
    {
        for (int j = 0; j < m ; j++) {
            b(i) += A(i, j) * v(j);
        }
    });

    Kokkos::fence();
}

void matVecSparse(CSRArrayKokkos<double>& A, CArrayKokkos<double>& v, CArrayKokkos<double>& b)
{
    size_t m = A.dim2();
    size_t n = A.dim1();
    FOR_ALL(i, 0, n, {
        size_t col;
        for (auto j = A.begin_index(i); j < A.end_index(i); j++) {
            col   = A.get_col_flat(j);
            b(i) += A(i, col) * v(col);
        }
    });
    Kokkos::fence();
}

int main(int argc, char** argv)
{
    Kokkos::initialize(); {
        int    nrows = 55;
        int    ncols = 55;
        size_t n;
        if (argc != 2) {
            printf("Usage is .powerTest <MatrixSize> using default of 5000\n");
            n = 5000;
        }
        else{
            n = (size_t) atoi(argv[1]);
        }
        nrows = n;
        ncols = n;
        CArrayKokkos<double> A(nrows, ncols);

        FOR_ALL(i, 0, nrows,
            j, 0, ncols, {
            A(i, j) = 0.0;
        });
        CArrayKokkos<double> data(3 * nrows);
        CArrayKokkos<size_t> starts(nrows + 1);
        CArrayKokkos<size_t> cols(3 * nrows);
        CArrayKokkos<double> v1(ncols);
        CArrayKokkos<double> v2(ncols);
        CArrayKokkos<double> b1(nrows);
        CArrayKokkos<double> b2(nrows);

        int i;
        i = 0;
        FOR_ALL(i, 0, ncols, {
            v1(i) = 1;
            v2(i) = 1;
            b1(i) = 0;
            b2(i) = 0;
        });
        FOR_ALL(i, 0, nrows, {
            if (i == nrows - 2) {
                A(i, i - 1)     = i;
                A(i, i)         = i;
                A(i, i + 1)     = i;
                data(3 * i)     = i;
                data(3 * i + 1) = i;
                data(3 * i + 2) = i;
                cols(3 * i)     = i - 1;
                cols(3 * i + 1) = i;
                cols(3 * i + 2) = i + 1;
                b1(i) = 0;
                b2(i) = 0;
                starts(i) = 3 * i;
            }
            else if (i == nrows - 1) {
                A(i, i - 2)     = i;
                A(i, i - 1)     = i;
                A(i, i)         = i;
                data(3 * i)     = i;
                data(3 * i + 1) = i;
                data(3 * i + 2) = i;
                cols(3 * i)     = i - 2;
                cols(3 * i + 1) = i - 1;
                cols(3 * i + 2) = i;
                b1(i) = 0;
                b2(i) = 0;
                starts(i) = 3 * i;
            }
            else {
                A(i, i)         = i;
                A(i, i + 1)     = i;
                A(i, i + 2)     = i;
                data(3 * i)     = i;
                data(3 * i + 1) = i;
                data(3 * i + 2) = i;
                cols(3 * i)     = i;
                cols(3 * i + 1) = i + 1;
                cols(3 * i + 2) = i + 2;
                b1(i) = 0;
                b2(i) = 0;
                starts(i) = 3 * i;
            }
        });
        RUN({
            starts(0) = 0;
            starts(nrows) = 3 * nrows;
        });
        CSRArrayKokkos<double> B(data, starts, cols, nrows, ncols);
        auto start = std::chrono::high_resolution_clock::now();

        int j;
        matVec(A, v1, b1);
        Kokkos::fence();
        auto lap1 = std::chrono::high_resolution_clock::now();
        auto lap2 = std::chrono::high_resolution_clock::now();
        matVecSparse(B, v2, b2);
        Kokkos::fence();
        auto lap3  = std::chrono::high_resolution_clock::now();
        auto time1 =  std::chrono::duration_cast<std::chrono::nanoseconds>(lap1 - start);
        auto time2 =  std::chrono::duration_cast<std::chrono::nanoseconds>(lap3 - lap2);

        if (!EXPORT) {
            RUN({ printf("Size: %ld, Dense: %.2e, Sparse: %.2e, %f, %f \n", n, time1.count() * 1e-9, time2.count() * 1e-9, b1(57980), b2(57980) ); });
        }
        else {
            RUN({
                for (int i = 0; i < n; i++) {
                    if (abs(b1(i) - b2(i) > 1e-7)) {
                        printf("b1(%d) - b2(%d) = %.2e\n", i, i, b1(i) - b2(i));
                    }
                } 
                printf("%ld, %.2e, %.2e, %f, %f, %f \n", n, time1.count() * 1e-9, time2.count() * 1e-9, (1e-9 * time1.count()) / (1e-9 * time2.count()), b1(25), b2(25) );
            });
        }
    } Kokkos::finalize();
}
