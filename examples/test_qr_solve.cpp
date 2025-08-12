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
 
#include <chrono>   // for timing
#include "cramers_rule.hpp"
#include "qr_solver.hpp"

bool verbose = false;

int test_qr_diagonal();
int test_qr_upper();
int test_qr_ragged();
int test_qr_heat_transfer(size_t num_vals);
int test_qr_hilbert(size_t num);
int test_qr_nonsquare();


int main(int argc, char *argv[]){

    Kokkos::initialize(argc, argv);
    {  

        std::cout << "\ntesting MATAR QR solver \n\n";

        int singular;

        std::cout << "\nRunning test_qr_diagonal\n\n";
        singular = test_qr_diagonal();

        std::cout << "\nRunning test_qr_upper\n\n";
        singular = test_qr_upper();

        std::cout << "\nRunning test_qr_ragged\n\n";
        singular = test_qr_ragged();

        std::cout << "\nRunning test_qr_heat_transfer\n\n";
        singular = test_qr_heat_transfer(10);

        std::cout << "\nRunning test_qr_hilbert(3)\n\n";
        singular = test_qr_hilbert(3);

        std::cout << "\nRunning test_qr_hilbert(4)\n\n";
        singular = test_qr_hilbert(4);

        std::cout << "\nRunning test_qr_nonsquare()\n\n";
        singular = test_qr_nonsquare();

    } // end of kokkos scope


    Kokkos::finalize();

    return 1;

 } // end function



// --------------
// --- test 1 ---
// --------------
int test_qr_diagonal(){

    size_t num = 3;
    DCArrayKokkos <double> A(num, num, "A");
    DCArrayKokkos <double> b(num, "b");
    DCArrayKokkos <double> x(num, "x");


    A.set_values(0.0);
    b.set_values(0.0);

    // run the host functions
    RUN({
        A(2,0) = 1;
        A(1,1) = 2;
        A(0,2) = 3;

        b(0) = 3;
        b(1) = 2;
        b(2) = 1;
    });
    A.update_host();
    b.update_host();

    printf("A = \n");
    for(size_t i=0; i<num; i++){
        for(size_t j=0; j<num; j++){
            printf("%f ", A.host(i,j));
        }
        printf("\n");
    }
    printf("\n");

    QR_solver_host(A, b, x); 
    x.update_host();

    printf("host executed routines \n");
    for(size_t i=0; i<num; i++){
        printf("x = %f \n", x.host(i));
    } // end for
    printf("exact = [1,1,1]^T \n\n");


    return 1;

} // end test 1


// --------------
// --- test 2 ---
// --------------
int test_qr_upper(){
    size_t num = 3;
    DCArrayKokkos <double> A(num, num, "A");
    DCArrayKokkos <double> b(num, "b");
    DCArrayKokkos <double> x(num, "x");

    A.set_values(0.0);
    b.set_values(0.0);

    RUN({
        A(0,0) = 1;
        A(0,1) = 2;
        A(0,2) = 3;
        A(1,1) = 4;
        A(1,2) = 5;
        A(2,2) = 6;

        b(0) = 14;
        b(1) = 23;
        b(2) = 18;
    });
    A.update_host();
    b.update_host();
    
    printf("A = \n");
    for(size_t i=0; i<num; i++){
        for(size_t j=0; j<num; j++){
            printf("%f ", A.host(i,j));
        }
        printf("\n");
    }
    printf("\n");


    QR_solver_host(A, b, x); 
    x.update_host();

    printf("host executed routines \n");
    for(size_t i=0; i<num; i++){
        printf("x = %f \n", x.host(i));
    } // end for
    printf("exact = [1,2,3]^T \n\n");

    return 1;

} // end function



// --------------
// --- test 3 ---
// --------------
int test_qr_ragged(){
    size_t num = 3;
    DCArrayKokkos <double> A(num, num, "A");
    DCArrayKokkos <double> b(num, "b");
    DCArrayKokkos <double> x(num, "x");

    A.set_values(0.0);
    b.set_values(0.0);

    RUN({
        A(0,2) = 6;

        A(1,0) = 1;
        A(1,1) = 2;
        A(1,2) = 3;

        A(2,1) = 4;
        A(2,2) = 5;

        b(0) = 18;
        b(1) = 14;
        b(2) = 23;
    });
    A.update_host();
    b.update_host();

    printf("A = \n");
    for(size_t i=0; i<num; i++){
        for(size_t j=0; j<num; j++){
            printf("%f ", A.host(i,j));
        }
        printf("\n");
    }
    printf("\n");


    QR_solver_host(A, b, x); 
    x.update_host();
    
    printf("host executed routines \n");
    for(size_t i=0; i<num; i++){
        printf("x = %f \n", x.host(i));
    } // end for
    printf("exact = [1,2,3]^T \n\n");


    return 1;

} // end function



// --------------
// --- test 4 ---
// --------------
int test_qr_heat_transfer(size_t num_vals){

    int singular; 
    int parity;

    DCArrayKokkos <double> M(num_vals,num_vals, "M");
    DCArrayKokkos <double> T_bc(num_vals, "T_bc");
    DCArrayKokkos <double> T_field(num_vals, "T_bc");
    M.set_values(0.0);
    T_bc.set_values(0.0);


    RUN({
        M(0,0) = 2.0;
        M(0,1) = -1.0;

        M(num_vals-1, num_vals-2) = -1.0;
        M(num_vals-1, num_vals-1) = 2.0;

        // boundary conditions
        T_bc(0) = 10.0;
        T_bc(num_vals-1) = 1.0;
    });
    
    DO_ALL(i, 1, num_vals-2, {
        M(i, i-1) = -1.0;
        M(i,i) = 2.0;
        M(i, i+1) = -1.0;
    });
    M.update_host();
    T_bc.update_host();

    if(verbose){
        printf("M = \n");
        for(size_t i=0; i<num_vals; i++){
        for(size_t j=0; j<num_vals; j++){
            printf("%f ", M.host(i,j));
        }
        printf("\n");
        }
        printf("\n");
    }

    // start timer
    auto time_1 = std::chrono::high_resolution_clock::now();

    QR_solver_host(M, T_bc, T_field); 
    T_field.update_host();

    // end timer
    auto time_2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration <double, std::milli> ms = time_2 - time_1;
    std::cout << "runtime of parallel heat transfer solve = " << ms.count() << "ms\n\n";

    if(verbose){
        printf("host executed routines \n");
        for(size_t i=0; i<num_vals; i++){
            printf("Temp_field = %f \n", T_field.host(i));
        } // end for
    }

    return 1;

} // end function


// --------------
// --- test 5 ---
// --------------
int test_qr_hilbert(size_t num){

    DCArrayKokkos <double> A(num, num, "A");
    DCArrayKokkos <double> b(num, "b");
    DCArrayKokkos <double> x(num, "x");

    A.set_values(0.0);
    b.set_values(0.0);

    // run the host functions
    FOR_ALL(i, 0, num, 
            j, 0, num, {
            
            A(i,j) = 1.0 / ((double)i+1 + (double)j+1 - 1.0);
    });
    FOR_ALL(i, 0, num,  {
            b(i) = (double)i + 1.0;
    });
    A.update_host();
    b.update_host();


    printf("\n");
    printf("A = \n");
    for(size_t i=0; i<num; i++){
        for(size_t j=0; j<num; j++){
            printf("%f ", A.host(i,j));
        }
        printf("\n");
    }
    printf("\n");


    // --------
    // for exact solution
    DCArrayKokkos <double> A_inverse(num, num);
    DCArrayKokkos <double> eye(num, num);
    eye.set_values(0.0);
    DCArrayKokkos <double> x_exact(num);


    // solve for inverse using Cramer's rule
    if(num==3){
        RUN({
            double det = invert_3x3(A, A_inverse);
        });
    }
    else if(num==4){
        RUN({
            double det = invert_4x4(A, A_inverse);
        });
    } else {
        printf("matrix size not supported in this test case \n");
        return 0;
    } 
    // end if

    FOR_FIRST(i, 0, num, {
        FOR_SECOND(j, 0, num, {   
            double sum = 0;
            double sum_lcl = 0;
            FOR_REDUCE_SUM_THIRD(k, 0 , num, 
                                sum_lcl, {
                sum_lcl += A_inverse(i, k)*A(k, j);
            }, sum);
            eye(i,j) = sum;
        }); // end parallel j
    }); // end parallel i
    
    A_inverse.update_host();
    eye.update_host();

    printf("A_inverse = \n");
    for(size_t i=0; i<num; i++){
        for(size_t j=0; j<num; j++){
            printf("%f ", A_inverse.host(i,j));
        }
        printf("\n");
    }
    printf("\n");

    printf("eye = A_inverse*A = \n");
    for(size_t i=0; i<num; i++){
        for(size_t j=0; j<num; j++){
            printf("%f ", eye.host(i,j));
        }
        printf("\n");
    }
    printf("\n");


    FOR_FIRST(i, 0, num, {
        double sum = 0;
        double sum_lcl = 0;
        FOR_REDUCE_SUM_SECOND(j, 0 , num, 
                              sum_lcl, {
            sum_lcl += A_inverse(i, j)*b(j);
        }, sum);
        x_exact(i) = sum;
    });
    x_exact.update_host();
    A_inverse.update_host();

    printf("exact solution: \n");
    for(size_t i=0; i<num; i++){
        printf("x = %f \n", x_exact.host(i));
    } // end for
    // -----


    // --------
    // QR solve


    QR_solver_host(A, b, x); 
    x.update_host();

    printf("host executed routines \n");
    for(size_t i=0; i<num; i++){
        printf("x = %f \n", x.host(i));
    } // end for


    return 1;

} // end function


// --------------
// --- test 6 ---
// --------------
int test_qr_nonsquare(){

    size_t m = 3;
    size_t n = 2;
    DCArrayKokkos <double> A(m, n, "A");
    DCArrayKokkos <double> b(m, "b");
    DCArrayKokkos <double> x(n, "x");


    A.set_values(0.0);
    b.set_values(0.0);

    // run the host functions
    // This corresponds to linear regression for data points:
    //  (1,1), (2,2), (3,2)
    RUN({
        A(0,0) = 1;
        A(0,1) = 1;
        A(1,0) = 1;
        A(1,1) = 2;
        A(2,0) = 1;
        A(2,1) = 3;

        b(0) = 1;
        b(1) = 2;
        b(2) = 2;
    });
    A.update_host();
    b.update_host();

    printf("A = \n");
    for(size_t i=0; i<m; i++){
        for(size_t j=0; j<n; j++){
            printf("%f ", A.host(i,j));
        }
        printf("\n");
    }
    printf("\n");

    QR_solver_host(A, b, x); 
    x.update_host();

    printf("host executed routines \n");
    for(size_t i=0; i<n; i++){
        printf("x = %f \n", x.host(i));
    } // end for
    printf("exact = [0.6667,0.5]^T \n\n");


    return 1;

} // end test 6