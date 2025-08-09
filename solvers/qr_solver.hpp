#ifndef QR_H
#define QR_H
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

 //////////////////////////

#include <iostream>
#include <stdio.h>
#include <cmath>


#include "matar.h"
using namespace mtr;


// Transpose matrix
void transpose_host(const DCArrayKokkos <double> &A,
                    DCArrayKokkos <double> &At) {

    size_t m = A.dims(0);
    size_t n = A.dims(1);

printf("here taking transpose\n");
    FOR_ALL(i, 0, m,
            j, 0, n, {
        At(j, i) = A(i,j);
    });

    return;

} // end function


// Back substitution to solve Rx = y
void backsub_host(const DCArrayKokkos <double> &R, 
                  const DCArrayKokkos <double> &y,
                  DCArrayKokkos <double> &x) {
    
    size_t n = R.dims(0);
    
    for (int i = n - 1; i >= 0; --i) {
        
        RUN({
            x(i) = y(i);
        });

        double sum = 0.0;
        double sum_lcl = 0.0;

        FOR_REDUCE_SUM(j, i + 1, n, 
                       sum_lcl, {
             sum_lcl -= R(i,j) * x(j);
        }, sum);

        RUN({
            x(i) += sum;
            x(i) /= R(i,i);
        });

    } // end for i

    return;

} // end function

// QR Decomposition using Modified Gram-Schmidt
void QR_decompose_host(const DCArrayKokkos <double> &A, 
                       DCArrayKokkos <double> &Q, 
                       DCArrayKokkos <double> &R) {


    const size_t m = A.dims(0);
    const size_t n = A.dims(1);

    Q.set_values(0.0);
    R.set_values(0.0);

    DCArrayKokkos <double> v(n,m,"v");

    // Copy columns of A to v, and taking transpose
    transpose_host(A,v);


    for (size_t i = 0; i < n; ++i) {
printf("loop i = %zu \n", i);
        // find the norm of a column in matrix v for row i
        double tally = 0.0;
        double tally_lcl = 0.0;


        FOR_REDUCE_SUM(j, 0, m, 
                       tally_lcl, {
            tally_lcl += v(i,j) * v(i,j);
        }, tally);

        RUN({
            R(i,i) = sqrt(tally); // row i norm
        });
        // done with norm calc

        FOR_ALL(j, 0, m, {
            Q(j,i) = v(i,j)/R(i,i);
        });


        FOR_FIRST(jj, i+1, n, {
printf("loop j = %d and n=%zu and i+1=%zu \n", jj, n, i+1);
            R(i,jj) = 0.0;

            double sum=0;
            double sum_lcl = 0;

            FOR_REDUCE_SUM_SECOND(k, 0, m,
                                  sum_lcl, {
                sum_lcl += Q(k,i) * A(k,jj);
            }, sum);
            R(i,jj) = sum;

           teamMember.team_barrier();

printf("v(i,k) reduce k to m \n");
            FOR_SECOND(k, 0, m,{
                v(jj,k) -= sum * Q(k,i);
            });

printf("bottom of j loop \n");

            teamMember.team_barrier();

        }); // end parallel j

    } // end for i
}

// Solve for x in Ax = b using QR
// A[m,n]
// x[n]
// b[m]
void QR_solver_host(const DCArrayKokkos <double> & A, 
                    DCArrayKokkos <double> &b,
                    DCArrayKokkos <double> &x) {
    
    const size_t m = A.dims(0);
    const size_t n = A.dims(1);

    DCArrayKokkos <double> Q(m,n,"Q");
    DCArrayKokkos <double> R(n,n,"R");
    DCArrayKokkos <double> y(n,"y");

printf("qr decomp call\n");
    QR_decompose_host(A, Q, R);

printf("calc y\n");
    // Compute Q^t * b
    FOR_FIRST(i, 0, n, {

        double sum = 0.0;
        double sum_lcl = 0.0;

        // Q[m,n] so Q^t[n,m] b[m]
        FOR_REDUCE_SUM_SECOND(j, 0, m, 
                              sum_lcl, {
            sum_lcl += Q(j,i) * b(j);
        }, sum);
        y(i) = sum;

    }); // end parallel i

printf("backsub call\n");

    // Solve R x = y
    backsub_host(R, y, x);
}





//////////////////////////


#endif // QR

/*
#include <math.h>

// Perform QR factorization (Modified Gram-Schmidt)
void qr_factorization(int m, int n, double *A, double *Q, double *R) {
    // Copy A into Q (we'll turn Q into orthonormal columns)
    for (int i = 0; i < m * n; i++)
        Q[i] = A[i];

    // Initialize R to zero
    for (int i = 0; i < n * n; i++)
        R[i] = 0.0;

    for (int k = 0; k < n; k++) {
        // Compute norm of k-th column of Q
        double norm = 0.0;
        for (int i = 0; i < m; i++)
            norm += Q[i * n + k] * Q[i * n + k];
        norm = sqrt(norm);

        R[k * n + k] = norm;

        // Normalize k-th column
        for (int i = 0; i < m; i++)
            Q[i * n + k] /= norm;

        // Orthogonalize remaining columns
        for (int j = k + 1; j < n; j++) {
            double dot = 0.0;
            for (int i = 0; i < m; i++)
                dot += Q[i * n + k] * Q[i * n + j];

            R[k * n + j] = dot;

            for (int i = 0; i < m; i++)
                Q[i * n + j] -= dot * Q[i * n + k];
        }
    }
}

// Multiply Q^t * b (Q is m-by-n, b is length m, result is length n)
void multiply_QT_b(int m, int n, double *Q, double *b, double *y) {
    for (int j = 0; j < n; j++) {
        y[j] = 0.0;
        for (int i = 0; i < m; i++)
            y[j] += Q[i * n + j] * b[i];
    }
}

// Back-substitution for R x = y (R is n-by-n upper triangular)
void back_substitution(int n, double *R, double *y, double *x) {
    for (int i = n - 1; i >= 0; i--) {
        double sum = y[i];
        for (int j = i + 1; j < n; j++)
            sum -= R[i * n + j] * x[j];
        x[i] = sum / R[i * n + i];
    }
}

// Utility: print matrix
void print_matrix(const char *name, double *M, int rows, int cols) {
    printf("%s =\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("%10.6f ", M[i * cols + j]);
        printf("\n");
    }
}

// Utility: print vector
void print_vector(const char *name, double *v, int len) {
    printf("%s = [", name);
    for (int i = 0; i < len; i++)
        printf(" %10.6f", v[i]);
    printf(" ]\n");
}

int main() {
    // Example system: A x = b
    int m = 3; // rows
    int n = 2; // cols
    double A[6] = {
        1, 2,
        3, 4,
        5, 6
    };
    double b[3] = {1, 0, 1};

    double Q[6], R[4], y[2], x[2];

    // Step 1: QR factorization
    qr_factorization(m, n, A, Q, R);

    // Step 2: Compute y = Qᵀ b
    multiply_QT_b(m, n, Q, b, y);

    // Step 3: Solve R x = y
    back_substitution(n, R, y, x);

    // Print results
    print_matrix("Q", Q, m, n);
    print_matrix("R", R, n, n);
    print_vector("y", y, n);
    print_vector("x", x, n);

    return 0;
}
*/


//////////////////////////////////


/*
// Dot product
double dot_host(
    const DCArrayKokkos <double> &a,
    const DCArrayKokkos <double> &b) {
    
    double result = 0.0;
    double sum_lcl = 0.0;

    FOR_REDUCE_SUM(i, 0, a.size(), 
                   sum_lcl, {
        sum_lcl += a(i) * b(i);
    }, result);

    return result;

} // end function



// Norm
double norm_vec_host(const DCArrayKokkos <double> &v) {
    return sqrt(dot_host(v, v));
} // end function



// Multiply matrix and vector
void mat_vec_multiply_host(
        const DCArrayKokkos <double> &A, 
        const DCArrayKokkos <double> &x,
        DCArrayKokkos <double> &result) {
    
    size_t m = A.dims(0), n = x.size();

    FOR_FIRST(i, 0, m, {

        double sum = 0.0;
        double sum_lcl = 0.0;

        FOR_REDUCE_SUM(j, 0, n,
                       sum_lcl, {
            sum_lcl += A(i,j) * x(j);
        }, sum);

        result(i) = sum;

    }); // end for i

    return;

} // end function
*/