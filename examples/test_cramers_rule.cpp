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
#include <iostream>
#include <iomanip>
#include <cmath>

#include "cramers_rule.hpp"

// comment and uncomment to test other MATAR data types
using TestArray = CArrayKokkos <double>;


//using TestArray = DCArrayKokkos <double>;  

// Helper function to multiply matrices
KOKKOS_INLINE_FUNCTION
void multiply_matrices(const TestArray& A, const TestArray& B, const TestArray& C, const size_t n) {
    for(size_t i=0; i<n;i++) 
    for(size_t j=0; j<n;j++) {
        C(i, j) = 0.0;
        for (size_t k = 0; k < n; k++) {
            C(i, j) += A(i, k) * B(k, j);
        } // end for k
    };
} // end function

// Helper function to check if matrix is identity
KOKKOS_INLINE_FUNCTION
bool is_identity(const TestArray& A, size_t n, double tol = 1e-10) {

   bool result = true;

    for(size_t i=0; i<n;i++) 
    for(size_t j=0; j<n;j++) {
            
        double expected = (i == j) ? 1.0 : 0.0;
        if (fabs(A(i, j) - expected) > tol) {
            result(0) = false;
        } // end if
    }

    return true;
} // end function

// Helper to compare two values
KOKKOS_INLINE_FUNCTION
bool close_enough(double a, double b, double tol = 1e-10) {
    return fabs(a - b) < tol;
}


//==============================================================================
// TEST 2x2 MATRICES
//==============================================================================
void test_2x2() {
    std::cout << "========================================\n";
    std::cout << "TESTING 2x2 MATRICES\n";
    std::cout << "========================================\n\n";

    // Test 1: Simple matrix with known determinant and inverse
    {
        std::cout << "Test 1: Simple 2x2 matrix\n";

        TestArray A(2, 2);
        TestArray A_inv(2, 2);
        TestArray result(2, 2);

        RUN({
            // Initialize matrix
            A(0, 0) = 4.0;  A(0, 1) = 7.0;
            A(1, 0) = 2.0;  A(1, 1) = 6.0;

            // Test determinant (expected: 4*6 - 7*2 = 10)
            double det = det_2x2(A);
            printf("  Determinant: %f (expected: 10.0)\n", det);
            printf("  Det test: ");
            if (close_enough(det, 10.0)) {
                printf("PASSED\n\n");
            } else {
                printf("FAILED\n\n");
                Kokkos::abort("ERROR: det_2x2 failed test\n");
            }

            // Test inversion
            invert_2x2(A, A_inv, det);
            multiply_matrices(A, A_inv, result, 2);
            
            bool is_id = is_identity(result, 2);
            printf("  A * A^(-1) inversion test: ");
            if (is_id) {
                printf("PASSED\n\n");
            } else {
                printf("FAILED\n\n");
                Kokkos::abort("ERROR: invert_2x2 failed test\n");
            }
        });
    }

    // Test 2: Identity matrix
    {
        std::cout << "Test 2: 2x2 Identity matrix\n";
        
        TestArray I(2, 2);
        
        RUN({
            I(0, 0) = 1.0;  I(0, 1) = 0.0;
            I(1, 0) = 0.0;  I(1, 1) = 1.0;
            
            double det = det_2x2(I);
            printf("  Det(I): %f (expected: 1.0)\n", det);
            printf("  Det test: ");
            if (close_enough(det, 1.0)) {
                printf("PASSED\n\n");
            } else {
                printf("FAILED\n\n");
                Kokkos::abort("ERROR: det_2x2 identity test failed\n");
            }
        });
    }

    // Test 3: Rotation matrix
    {
        std::cout << "Test 3: Rotation matrix (45 degrees)\n";
        
        double angle = M_PI / 4.0;
        TestArray R(2, 2);
        TestArray R_inv(2, 2);
        TestArray result(2, 2);
        
        RUN({
            R(0, 0) = cos(angle);  R(0, 1) = -sin(angle);
            R(1, 0) = sin(angle);  R(1, 1) =  cos(angle);
        
            double det = det_2x2(R);
            printf("  Det(R): %f (expected: 1.0)\n", det);
            printf("  Det test: ");
            if (close_enough(det, 1.0)) {
                printf("PASSED\n\n");
            } else {
                printf("FAILED\n\n");
                Kokkos::abort("ERROR: det_2x2 rotation test failed\n");
            }

            invert_2x2(R, R_inv, det);
            multiply_matrices(R, R_inv, result, 2);
            
            bool is_id = is_identity(result, 2);
            printf("  R * R^(-1) inversion test: ");
            if (is_id) {
                printf("PASSED\n\n");
            } else {
                printf("FAILED\n\n");
                Kokkos::abort("ERROR: invert_2x2 rotation test failed\n");
            }
        });
    }
}

//==============================================================================
// TEST 3x3 MATRICES
//==============================================================================
void test_3x3() {
    std::cout << "========================================\n";
    std::cout << "TESTING 3x3 MATRICES\n";
    std::cout << "========================================\n\n";

    // Test 1: Simple matrix
    {
        std::cout << "Test 1: Simple 3x3 matrix\n";
        
        TestArray A(3, 3);  
        TestArray A_inv(3, 3);
        TestArray result(3, 3);

        RUN({
            A(0, 0) = 1.0;  A(0, 1) = 2.0;  A(0, 2) = 3.0;
            A(1, 0) = 0.0;  A(1, 1) = 1.0;  A(1, 2) = 4.0;
            A(2, 0) = 5.0;  A(2, 1) = 6.0;  A(2, 2) = 0.0;
            
            // Test determinant (expected: 1.0)
            // det = 1*(1*0 - 4*6) - 2*(0*0 - 4*5) + 3*(0*6 - 1*5)
            //     = -24 + 40 - 15 = 1
            double det = det_3x3(A);
            printf("  Determinant: %f (expected: 1.0)\n", det);
            printf("  Det test: ");
            if (close_enough(det, 1.0)) {
                printf("PASSED\n\n");
            } else {
                printf("FAILED\n\n");
                Kokkos::abort("ERROR: det_3x3 failed test\n");
            }

            // Test inversion
            double det_check = invert_3x3(A, A_inv);
            printf("  Det from invert function: %f\n", det_check);
            
            multiply_matrices(A, A_inv, result, 3);
            
            bool is_id = is_identity(result, 3);
            printf("  A * A^(-1) inversion test: ");
            if (is_id) {
                printf("PASSED\n\n");
            } else {
                printf("FAILED\n\n");
                Kokkos::abort("ERROR: invert_3x3 failed test\n");
            }
        });
    }

    // Test 2: Identity matrix
    {
        std::cout << "Test 2: 3x3 Identity matrix\n";
        
        TestArray I(3, 3);
        TestArray I_inv(3, 3);

        RUN({
            I(0, 0) = 1.0;  I(0, 1) = 0.0;  I(0, 2) = 0.0;
            I(1, 0) = 0.0;  I(1, 1) = 1.0;  I(1, 2) = 0.0;
            I(2, 0) = 0.0;  I(2, 1) = 0.0;  I(2, 2) = 1.0;
            
            double det = det_3x3(I);
            printf("  Det(I): %f (expected: 1.0)\n", det);
            printf("  Det test: ");
            if (close_enough(det, 1.0)) {
                printf("PASSED\n\n");
            } else {
                printf("FAILED\n\n");
                Kokkos::abort("ERROR: det_3x3 identity test failed\n");
            }
            
            invert_3x3(I, I_inv, det);
            
            bool is_id = is_identity(I_inv, 3);
            printf("  I^(-1) inversion test: ");
            if (is_id) {
                printf("PASSED\n\n");
            } else {
                printf("FAILED\n\n");
                Kokkos::abort("ERROR: invert_3x3 identity test failed\n");
            }
        });
    }

    // Test 3: Diagonal matrix
    {
        std::cout << "Test 3: 3x3 Diagonal matrix\n";
        
        TestArray D(3, 3);
        TestArray D_inv(3, 3);
        TestArray result(3, 3);

        RUN({
            D(0, 0) = 2.0;  D(0, 1) = 0.0;  D(0, 2) = 0.0;
            D(1, 0) = 0.0;  D(1, 1) = 3.0;  D(1, 2) = 0.0;
            D(2, 0) = 0.0;  D(2, 1) = 0.0;  D(2, 2) = 4.0;
            
            double det = det_3x3(D);
            printf("  Det(D): %f (expected: 24.0)\n", det);
            printf("  Det test: ");
            if (close_enough(det, 24.0)) {
                printf("PASSED\n\n");
            } else {
                printf("FAILED\n\n");
                Kokkos::abort("ERROR: det_3x3 diagonal test failed\n");
            }
            
            invert_3x3(D, D_inv, det);
            
            // Check diagonal elements (should be 1/d_ii)
            bool correct_diag = 
                close_enough(D_inv(0, 0), 0.5) &&
                close_enough(D_inv(1, 1), 1.0/3.0) &&
                close_enough(D_inv(2, 2), 0.25);
            printf("  Diagonal inverse test: ");
            if (correct_diag) {
                printf("PASSED\n");
            } else {
                printf("FAILED\n");
                Kokkos::abort("ERROR: invert_3x3 diagonal elements test failed\n");
            }
            
            multiply_matrices(D, D_inv, result, 3);
            
            bool is_id = is_identity(result, 3);
            printf("  D * D^(-1) inversion test: ");
            if (is_id) {
                printf("PASSED\n\n");
            } else {
                printf("FAILED\n\n");
                Kokkos::abort("ERROR: invert_3x3 full diagonal test failed\n");
            }
        });
    }

    // Test 4: Jacobian-like matrix (common in FEM)
    {
        std::cout << "Test 4: Jacobian-like matrix\n";
        
        TestArray J(3, 3);
        TestArray J_inv(3, 3);
        TestArray result(3, 3);

        RUN({
            J(0, 0) = 0.5;   J(0, 1) = 0.25;  J(0, 2) = 0.0;
            J(1, 0) = 0.25;  J(1, 1) = 0.5;   J(1, 2) = 0.0;
            J(2, 0) = 0.0;   J(2, 1) = 0.0;   J(2, 2) = 1.0;
            
            double det = det_3x3(J);
            printf("  Det(J): %f\n", det);
            
            invert_3x3(J, J_inv, det);
            multiply_matrices(J, J_inv, result, 3);
            
            bool is_id = is_identity(result, 3);
            printf("  J * J^(-1) inversion test: ");
            if (is_id) {
                printf("PASSED\n\n");
            } else {
                printf("FAILED\n\n");
                Kokkos::abort("ERROR: invert_3x3 Jacobian test failed\n");
            }
        });
    }
}

//==============================================================================
// TEST 4x4 MATRICES
//==============================================================================
void test_4x4() {
    std::cout << "========================================\n";
    std::cout << "TESTING 4x4 MATRICES\n";
    std::cout << "========================================\n\n";

    // Test 1: Simple matrix
    {
        std::cout << "Test 1: Simple 4x4 matrix\n";
        
        TestArray A(4, 4);
        TestArray A_inv(4, 4);
        TestArray result(4, 4);

        RUN({
            A(0, 0) = 1.0;  A(0, 1) = 0.0;  A(0, 2) = 2.0;  A(0, 3) = -1.0;
            A(1, 0) = 3.0;  A(1, 1) = 0.0;  A(1, 2) = 0.0;  A(1, 3) = 5.0;
            A(2, 0) = 2.0;  A(2, 1) = 1.0;  A(2, 2) = 4.0;  A(2, 3) = -3.0;
            A(3, 0) = 1.0;  A(3, 1) = 0.0;  A(3, 2) = 5.0;  A(3, 3) = 0.0;
            
            double det = det_4x4(A);
            printf("  Determinant: %f\n", det);
            
            double det_check = invert_4x4(A, A_inv);
            printf("  Det from invert function: %f\n", det_check);
            
            multiply_matrices(A, A_inv, result, 4);
            
            bool is_id = is_identity(result, 4);
            printf("  A * A^(-1) inversion test: ");
            if (is_id) {
                printf("PASSED\n\n");
            } else {
                printf("FAILED\n\n");
                Kokkos::abort("ERROR: invert_4x4 failed test\n");
            }
        });
    }

    // Test 2: Identity matrix
    {
        std::cout << "Test 2: 4x4 Identity matrix\n";
        
        TestArray I(4, 4);
        TestArray I_inv(4, 4);

        RUN({
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    I(i, j) = (i == j) ? 1.0 : 0.0;
                }
            }
            
            double det = det_4x4(I);
            printf("  Det(I): %f (expected: 1.0)\n", det);
            printf("  Det test: ");
            if (close_enough(det, 1.0)) {
                printf("PASSED\n\n");
            } else {
                printf("FAILED\n\n");
                Kokkos::abort("ERROR: det_4x4 identity test failed\n");
            }
            
            invert_4x4(I, I_inv, det);
            
            bool is_id = is_identity(I_inv, 4);
            printf("  I^(-1) inversion test: ");
            if (is_id) {
                printf("PASSED\n\n");
            } else {
                printf("FAILED\n\n");
                Kokkos::abort("ERROR: invert_4x4 identity test failed\n");
            }
        });
    }

    // Test 3: Diagonal matrix
    {
        std::cout << "Test 3: 4x4 Diagonal matrix\n";
        
        TestArray D(4, 4);
        TestArray D_inv(4, 4);
        TestArray result(4, 4);
        
        RUN({
            D(0, 0) = 2.0;  D(0, 1) = 0.0;  D(0, 2) = 0.0;  D(0, 3) = 0.0;
            D(1, 0) = 0.0;  D(1, 1) = 3.0;  D(1, 2) = 0.0;  D(1, 3) = 0.0;
            D(2, 0) = 0.0;  D(2, 1) = 0.0;  D(2, 2) = 4.0;  D(2, 3) = 0.0;
            D(3, 0) = 0.0;  D(3, 1) = 0.0;  D(3, 2) = 0.0;  D(3, 3) = 5.0;
            
            double det = det_4x4(D);
            printf("  Det(D): %f (expected: 120.0)\n", det);
            printf("  Det test: ");
            if (close_enough(det, 120.0)) {
                printf("PASSED\n\n");
            } else {
                printf("FAILED\n\n");
                Kokkos::abort("ERROR: det_4x4 diagonal test failed\n");
            }

            invert_4x4(D, D_inv, det);
            multiply_matrices(D, D_inv, result, 4);
            
            bool is_id = is_identity(result, 4);
            printf("  D * D^(-1) inversion test: ");
            if (is_id) {
                printf("PASSED\n\n");
            } else {
                printf("FAILED\n\n");
                Kokkos::abort("ERROR: invert_4x4 diagonal test failed\n");
            }
        });
    }

    // Test 4: Block matrix
    {
        std::cout << "Test 4: Block diagonal-ish matrix\n";
        
        TestArray B(4, 4);
        TestArray B_inv(4, 4);
        TestArray result(4, 4);

        RUN({
            B(0, 0) = 1.0;  B(0, 1) = 2.0;  B(0, 2) = 0.0;  B(0, 3) = 0.0;
            B(1, 0) = 3.0;  B(1, 1) = 4.0;  B(1, 2) = 0.0;  B(1, 3) = 0.0;
            B(2, 0) = 0.0;  B(2, 1) = 0.0;  B(2, 2) = 5.0;  B(2, 3) = 6.0;
            B(3, 0) = 0.0;  B(3, 1) = 0.0;  B(3, 2) = 7.0;  B(3, 3) = 8.0;
            
            // det = det([1,2;3,4]) * det([5,6;7,8]) = (-2) * (-2) = 4
            double det = det_4x4(B);
            printf("  Det(B): %f (expected: 4.0, product of 2x2 blocks)\n", det);
            printf("  Det test: ");
            if (close_enough(det, 4.0)) {
                printf("PASSED\n\n");
            } else {
                printf("FAILED\n\n");
                Kokkos::abort("ERROR: det_4x4 block test failed\n");
            }
            
            invert_4x4(B, B_inv, det);
            multiply_matrices(B, B_inv, result, 4);
            
            bool is_id = is_identity(result, 4);
            printf("  B * B^(-1) inversion test: ");
            if (is_id) {
                printf("PASSED\n\n");
            } else {
                printf("FAILED\n\n");
                Kokkos::abort("ERROR: invert_4x4 block test failed\n");
            }
        });
    }
}



//==============================================================================
// MAIN TEST DRIVER
//==============================================================================
int main(int argc, char *argv[]) {

    Kokkos::initialize(argc, argv);
    {  

    std::cout << std::fixed << std::setprecision(8);
    
    std::cout << "\n";
    std::cout << "=========================================\n";
    std::cout << "|  MATRIX DETERMINANT & INVERSE TESTS   |\n";
    std::cout << "=========================================\n";
    std::cout << "\n";

    test_2x2();
    test_3x3();
    test_4x4();

    std::cout << "========================================\n";
    std::cout << "ALL TESTS COMPLETED\n";
    std::cout << "========================================\n";
    }
    Kokkos::finalize();

    return 0;
}

