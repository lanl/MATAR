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
#include <chrono>
#include <time.h>

#define EXPORT true

using namespace mtr; // matar namespace

void matVecSp(CSRArrayKokkos<double>& A, CArrayKokkos<double>& v, CArrayKokkos<double>& b)
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
}

void renormSp(CArrayKokkos<double>& b)
{
    double total     = 0;
    double loc_total = 0;
    int    n = b.dims(0);
    int    i = 0;
    FOR_REDUCE_SUM(i, 0, n,
                loc_total, { loc_total += b(i) * b(i); }
        , total);
    total = 1 / sqrt(total);
    FOR_ALL(i, 0, n,
    { 
        b(i) *= total; 
    });
    // printf("Norm is %f\n", total);
}

void copySp(CArrayKokkos<double>& a, CArrayKokkos<double>& b)
{
    int n = b.dims(0);
    FOR_ALL(i, 0, n,
    { 
        b(i) = a(i);
        a(i) = 0;
    });
}

double innerProdSp(CArrayKokkos<double>& a, CArrayKokkos<double>& b)
{
    double total     = 0;
    double loc_total = 0;
    int    n = b.dims(0);
    FOR_REDUCE_SUM(i, 0, n,
        loc_total, { 
            loc_total += a(i) * b(i); 
        }, total);
    return total;
}

double l1ChangeSp(CArrayKokkos<double>& a, CArrayKokkos<double>& b)
{
    double total     = 0;
    double loc_total = 0;
    int    n = b.dims(0);
    FOR_REDUCE_SUM(i, 0, n,
        loc_total, { 
            loc_total += abs(a(i) - b(i)); 
        }, total);
    return total;
}

double powerIterSp(CSRArrayKokkos<double>& A, CArrayKokkos<double>& v, CArrayKokkos<double>& b, double tol, int max_iter, int& did_converge)
{
    double last_totl = 4 * tol;
    double my_tol    = 2 * tol;
    int    my_iter   = 0;

    while (my_iter < max_iter && my_tol > tol) {
        matVecSp(A, v, b);
        renormSp(b);
        if (my_iter % 100 == 0) {
            my_tol = l1ChangeSp(b, v);
        }
        copySp(b, v);
        my_iter++;
    }
    matVecSp(A, v, b);
    if (!EXPORT) {
        printf("Converged in %d iterations with tol of %f\n", my_iter, my_tol);
    }
    if (my_iter >= max_iter && my_tol > tol) {
        did_converge = 0;
    }
    else{
        did_converge = 1;
    }
    return innerProdSp(v, b);
}

int main(int argc, char** argv)
{
    Kokkos::initialize(); {
        size_t n;
        if (argc != 2) {
            printf("Usage is .powerTest <MatrixSize> using default of 5000\n");
            n = 5000;
        }
        else{
            n = (size_t) atoi(argv[1]);
        }
        CArrayKokkos<double> v(n);
        CArrayKokkos<double> b1(n);
        CArrayKokkos<double> b2(n);
        CArrayKokkos<double> v1(n);
        CArrayKokkos<double> data(3 * n - 2);
        CArrayKokkos<size_t> starts(n + 1);
        CArrayKokkos<size_t> cols(3 * n - 2);
        double eig1   = 0;
        double eig2   = 0;
        double my_tol = n * (1e-09);
        int    t1     = 1;
        int    t2     = 1;
        FOR_ALL(i, 0, n,
        {
            v(i)  = 1;
            v1(i) = 1;
            b1(i) = 0;
            b2(i) = 0;
            if (i == 1) {
                starts(i) = 2;
            }
            else if (i == 0) {
                starts(i) = 0;
            }
            else{
                starts(i) = 2 + 3 * (i - 1);
            }
        });
        RUN({ starts(n) = 3 * n - 2; });
        FOR_ALL(i, 0, n,
                j, 0, n, {
            if (abs(i - j) <= 1) {
                if (i == 0) {
                    data(i + j) = i + 2 * j;
                    cols(i + j) = j;
                }
                else {
                    data(3 * (i - 1) + 3 + j - i) = i + 2 * j;
                    cols(3 * (i - 1) + 3 + j - i) = j;
                }
            }
        });
        CSRArrayKokkos<double> Asp(data, starts, cols, n, n);
        auto start = std::chrono::high_resolution_clock::now();
        eig2 = powerIterSp(Asp, v1, b2, my_tol, 10000000, t2);
        auto lap = std::chrono::high_resolution_clock::now();
        if (!EXPORT) {
            printf("Max eig is %f %f\n", eig1, eig2);
            printf("Sparse took %.2e\n", std::chrono::duration_cast<std::chrono::nanoseconds>(lap - start).count() * 1e-9);
        }
        else {
            printf("%ld, %.2e, %d, %f\n", n, std::chrono::duration_cast<std::chrono::nanoseconds>(lap - start).count() * 1e-9, t2, eig2);
        }
    } Kokkos::finalize();
    return 0;
}
