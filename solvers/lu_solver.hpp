#ifndef LUSOLVER_H
#define LUSOLVER_H
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

#include <cmath>
#include <iostream>

#include "matar.h"
using namespace mtr;

// Precision-scaled singularity guard: a few ulps above zero at the working
// precision (a fixed 1e-15 is meaningless below double precision).
template <typename T>
KOKKOS_INLINE_FUNCTION T lu_tiny() {
    return mtr::impl::epsilon<T>() * T(10);
}

// ---------------------------
// LU decomposition functions
// ---------------------------

// the function is run on the GPU
template <typename T1, typename T2, typename T3>
KOKKOS_FUNCTION int LU_decompose(const T1& A,     // device array A (e.g., DCArrayKokkos <double>)
                                                  // passed in and is sent out in LU decomp format
                                 const T2& perm,  // permutations (e.g., DCArrayKokkos <size_t>)
                                 const T3& vv,    // helper array (e.g., CArrayKokkos <double>)
                                 int& parity) {   // parity (+1 or -1)

    const int n = A.dims(0);  // size of array

    parity = 1;

    // helper variables
    real_t temp;

    // search for the largest element in each row; save the scaling in the
    // temporary array vv and return zero if the array is singular
    for (size_t i = 0; i < n; i++) {
        real_t big = real_t(0);
        for (size_t j = 0; j < n; j++) {
            if ((temp = mtr::impl::fabs(A(i, j))) > big) {
                big = temp;
            }
        }

        if (big == real_t(0)) return (0);

        vv(i) = big;
    }

    // the main loop for the Crout's algorithm
    for (size_t j = 0; j < n; j++) {
        // this is the part a) of the algorithm except for i==j
        for (size_t i = 0; i < j; i++) {
            real_t sum = A(i, j);

            for (size_t k = 0; k < i; k++) {
                sum -= A(i, k) * A(k, j);
            }

            A(i, j) = sum;
        }

        // initialize for the search for the largest pivot element
        real_t big  = real_t(0);
        size_t imax = j;

        // this is the part a) for i==j and part b) for i>j + pivot search
        for (size_t i = j; i < n; i++) {
            real_t sum = A(i, j);

            for (size_t k = 0; k < j; k++) {
                sum -= A(i, k) * A(k, j);
            }

            A(i, j) = sum;

            // is the figure of merit for the pivot better than the best so far?
            if ((temp = vv(i) * mtr::impl::fabs(sum)) >= big) {
                big  = temp;
                imax = i;
            }
        }  // end for i

        // interchange rows, if needed, change parity and the scale factor
        if (imax != j) {
            for (size_t k = 0; k < n; k++) {
                temp       = A(imax, k);
                A(imax, k) = A(j, k);
                A(j, k)    = temp;
            }

            parity   = -(parity);
            vv(imax) = vv(j);
        }

        // store the index
        perm(j) = imax;
        // if the pivot element is zero, the array is singular but for some
        // applications a tiny number is desirable instead

        if (A(j, j) == real_t(0)) {
            A(j, j) = lu_tiny<real_t>();
        }
        // finally, divide by the pivot element

        if (j < n - 1) {
            temp = real_t(1) / real_t(A(j, j));
            for (size_t i = j + 1; i < n; i++) {
                A(i, j) *= temp;
            }  // end for i
        }  // end if j

    }  // end for j

    return (1);
}  // end function

// -------------------------------
// LU back substitution functions
// -------------------------------

// this function is run on the GPU
template <typename T1, typename T2, typename T3>
KOKKOS_FUNCTION void LU_backsub(const T1& A,     // input array A (e.g., DCArrayKokkos <double>) in LU decomp format
                                const T2& perm,  // permutations (e.g., DCArrayKokkos <size_t>)
                                const T3& b) {   // RHS and is answer x to Ax=B (e.g., DCArrayKokkos <double>)

    const int n = A.dims(0);  // size of array

    int ii = -1;

    // First step of backsubstitution; the only wrinkle is to unscramble
    // the permutation order. Note: the algorithm is optimized for a
    // possibility of large amount of zeroes in b

    for (size_t i = 0; i < n; i++) {
        size_t ip = perm(i);

        real_t sum = b(ip);
        b(ip)      = b(i);

        if (ii >= 0) {
            for (size_t j = ii; j < i; j++) {
                sum -= A(i, j) * b(j);
            }
        } else if (sum > real_t(0)) {
            ii = i;  // a nonzero element encounted
        }

        b(i) = sum;
    }  // end loop i

    // the second step
    for (int i = n - 1; i >= 0; i--) {
        real_t sum = b(i);
        for (size_t j = i + 1; j < n; j++) {
            sum -= A(i, j) * b(j);
        }  // end j

        b(i) = sum / real_t(A(i, i));
    }  // end loop i

}  // end if

// ------------------
// LU invert function
// ------------------
template <typename T1, typename T2, typename T3, typename T4>
KOKKOS_INLINE_FUNCTION void LU_invert(T1& A,        // input array, e.g., DCArrayKokkos <double>
                                      T2& perm,     // permutations, e.g., DCArrayKokkos <size_t>
                                      T3& inv_mat,  // inverse array, e.g., DCArrayKokkos <double>
                                      T4& col) {    // tmp array, e.g., DCArrayKokkos <double>

    const size_t n = A.dims(0);  // size of array

    for (size_t j = 0; j < n; j++) {
        for (size_t i = 0; i < n; i++) {
            col(i) = 0.0;
        }  // end for i

        col(j) = 1.0;
        LU_backsub(A, perm, col);

        for (size_t i = 0; i < n; i++) {
            inv_mat(i, j) = col(i);
        }  // end for i

    }  // end for j

    return;

}  // end function

// -----------------------
// LU determinant function
//  Input:  A filled in LUPDecompose; N - dimension.
//  Output: determinate of original A array
// -----------------------
template <typename T>
KOKKOS_INLINE_FUNCTION real_t LU_determinant(T& A,                // input array, e.g., DCArrayKokkos <double>
                                             const int parity) {  // parity (+1 0r -1)

    const int n = A.dims(0);  // size of array

    real_t res = real_t(parity);

    for (size_t j = 0; j < n; j++) {
        res *= A(j, j);
    }  // end j

    return (res);

}  // end function

// ============================================
//  GPU kernals
// ============================================

// the function is run from the host, and kernals
// inside this function are run on the GPU
template <typename T1, typename T2, typename T3>
int LU_decompose_host(T1& A,          // device array A (e.g., DCArrayKokkos <double>) passed in and is sent
                                      // out in LU decomp format
                      T2& perm,       // permutations (e.g., DCArrayKokkos <size_t>)
                      T3& vv,         // helper array (e.g., CArrayKokkos <double>)
                      int& parity) {  // parity (+1 or -1)

    const int n = A.dims(0);  // size of array

    CArrayKokkos<real_t> temp_scalar(1);  // persistant scalar on device

    parity = 1;

    // STEP 1:
    // search for the largest element in each row; save the scaling in the
    // temporary array vv and return zero if the array is singular
    FOR_FIRST(i, 0, n, {
        real_t max_val     = real_t(0);
        real_t max_val_lcl = real_t(0);

        FOR_REDUCE_MAX_SECOND(j, 0, n,
                              max_val_lcl, {
            max_val_lcl = mtr::impl::fmax(max_val_lcl, real_t(mtr::impl::fabs(A(i, j))));
        }, max_val);  // end parallel j

        vv(i) = max_val;
    });  // end for

    // if the largest value in the array row is 0, then exit
    real_t min_val     = real_t(0);
    real_t min_val_lcl = real_t(0);
    FOR_REDUCE_MIN(i, 0, n,
                   min_val_lcl, {
        min_val_lcl = mtr::impl::fmin(real_t(vv(i)), min_val_lcl);
    }, min_val);
    if (min_val < lu_tiny<real_t>()) return (0);  // singular array as all row values are 0

    // STEP 2:
    // the main loop for the Crout's algorithm
    for (size_t j = 0; j < n; j++) {
        // this is the part a) of the algorithm except for i==j
        for (size_t i = 0; i < j; i++) {
            real_t sum     = real_t(0);
            real_t sum_lcl = real_t(0);

            FOR_REDUCE_SUM(k, 0, i,
                           sum_lcl, {
                sum_lcl -= A(i, k) * A(k, j);
            }, sum);  // end parallel k

            RUN({ A(i, j) = sum + A(i, j); });

        }  // end i

        // this is the part a) for i==j and part b) for i>j
        // loop is from i=j to i<n
        for (size_t i = j; i < n; i++) {
            real_t sum     = real_t(0);
            real_t sum_lcl = real_t(0);

            FOR_REDUCE_SUM(k, 0, j,
                           sum_lcl, {
                sum_lcl -= A(i, k) * A(k, j);
            }, sum);  // parallel k

            RUN({ A(i, j) = sum + A(i, j); });
        }  // end i

        // initialize the search for the largest pivot element

        real_t max_val     = real_t(0);
        real_t max_val_lcl = real_t(0);
        // loop is from i=j to i<n
        FOR_REDUCE_MAX(i, j, n,
                       max_val_lcl, {
            // is the figure of merit for the pivot better than the best so far?
            if (vv(i) * mtr::impl::fabs(A(i, j)) >= max_val_lcl) {
                max_val_lcl = vv(i) * mtr::impl::fabs(A(i, j));
            }  // end if
        }, max_val);  // end for i
        Kokkos::fence();

        size_t imax     = j;
        size_t imax_lcl = j;
        // loop is from i=j to i<n
        FOR_REDUCE_MAX(i, j, n,
                       imax_lcl, {
            // is the figure of merit for the pivot better than the best so far?
            if (vv(i) * mtr::impl::fabs(A(i, j)) >= max_val) {
                imax_lcl = i;
            }  // end if
        }, imax);  // end for i
        Kokkos::fence();

        // interchange rows, if needed, change parity and the scale factor
        if (imax != j) {
            FOR_ALL(k, 0, n, {
                real_t temp = A(imax, k);
                A(imax, k)  = A(j, k);
                A(j, k)     = temp;
            });

            parity = -parity;
            RUN({ vv(imax) = vv(j); });

        }  // end if

        // store the index
        RUN({ perm(j) = imax; });

        // if the pivot element is zero, the array is singular but for some
        // applications a tiny number is desirable instead
        RUN({
            if (A(j, j) == real_t(0)) {
                A(j, j) = lu_tiny<real_t>();
            }
        });

        // finally, divide by the pivot element
        if (j < n - 1) {
            RUN({ temp_scalar(0) = real_t(1) / real_t(A(j, j)); });

            // loop is from i=j+1 to i<n
            FOR_ALL(i, j + 1, n, {
                A(i, j) *= temp_scalar(0);
            });
            Kokkos::fence();

        }  // end if

    }  // end for j

    perm.update_host();

    return (1);
}

// the function is run from the host, and kernals
// inside this function are run on the GPU
template <typename T1, typename T2, typename T3>
void LU_backsub_host(const T1& A,     // input array A (e.g., DCArrayKokkos <double>) in LU decomp format
                     const T2& perm,  // permutations (e.g., DCArrayKokkos <size_t>)
                     T3& b) {         // RHS and is answer x to Ax=B (e.g., DCArrayKokkos <double>)

    const int n = A.dims(0);  // size of array

    CArrayKokkos<real_t> val(1);  // a helper variable that carries a scalar

    // First step of backsubstitution; the only wrinkle is to unscramble
    // the permutation order. Note, the algorithm is optimized for a
    // possibility of large amount of zeroes in b

    // Forward substitution: solve L x = P b
    for (size_t i = 0; i < n; i++) {
        size_t ip = perm.host(i);

        RUN({
            val(0) = b(ip);
            b(ip)  = b(i);
        });

        real_t sum     = real_t(0);
        real_t sum_lcl = real_t(0);

        // j=0 to j<i
        FOR_REDUCE_SUM(j, 0, i,
                       sum_lcl, {
            sum_lcl -= A(i, j) * b(j);
        }, sum);
        Kokkos::fence();

        RUN({ b(i) = sum + val(0); });

    }  // end loop i

    // the second step
    // Backward substitution: solve U b = x
    for (int i = n - 1; i >= 0; i--) {
        real_t sum     = real_t(0);
        real_t sum_lcl = real_t(0);

        // for j=i+1 to j<N
        FOR_REDUCE_SUM(j, i + 1, n,
                       sum_lcl, {
            sum_lcl -= A(i, j) * b(j);
        }, sum);
        Kokkos::fence();

        RUN({ b(i) = (sum + b(i)) / real_t(A(i, i)); });

    }  // end for i

}  // end LU backsubstitution on lauched from host

// -----------------------
// LU determinant function
//  Input:  A filled in LUPDecompose; N - dimension.
//  Output: determinate of original A array
// -----------------------
template <typename T>
real_t LU_determinant_host(T& A,                // input array (e.g., DCArrayKokkos <double>)
                           const int parity) {  // parity (+1 0r -1)

    const int n = A.dims(0);  // size of array

    real_t res = real_t(parity);
    real_t prod_tally;
    real_t prod_lcl = real_t(1);

    FOR_REDUCE_PRODUCT(j, 0, n,
                       prod_lcl, {
        prod_lcl *= A(j, j);
    }, prod_tally);  // end j

    res *= prod_tally;

    return (res);

}  // end function

// ------------------
// LU invert function
// ------------------
template <typename T1, typename T2, typename T3, typename T4>
void LU_invert_host(T1& A,        // input array (e.g., DCArrayKokkos <double>)
                    T2& perm,     // permutations (e.g., DCArrayKokkos <size_t> )
                    T3& inv_mat,  // inverse array (e.g., DCArrayKokkos <double>)
                    T4& col) {    // tmp array (e.g., DCArrayKokkos <double>)

    const size_t n = A.dims(0);  // size of array

    for (size_t j = 0; j < n; j++) {
        col.set_values(0.0);

        RUN({ col(j) = 1.0; });

        LU_backsub_host(A, perm, col);

        FOR_ALL(i, 0, n, {
            inv_mat(i, j) = col(i);
        });  // end for i

    }  // end for j

    return;

}  // end function

// Solve for x in Ax = b using LU
// A[n,n]
// b[n], note answer, x, is returned in b
template <typename T1, typename T2, typename T3, typename T4>
int LU_solver_host(T1& A,     // e.g., DCArrayKokkos <double>
                   T2& b,     // e.g., DCArrayKokkos <double>
                   T3& perm,  // permutations (e.g., DCArrayKokkos <size_t>)
                   T4& vv,    // e.g., CArrayKokkos <double>
                   int& parity) {
    int singular = 0;
    parity       = 0;
    singular     = LU_decompose_host(A, perm, vv, parity);  // A is returned as the LU array

    if (singular == 0) {
        printf("ERROR: array is singluar \n");
        return 0;
    }

    LU_backsub_host(A, perm, b);  // note: answer is sent back in b

    return singular;
}

#endif  // LUSOLVER
