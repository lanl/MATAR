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
 
 #include "lu_solver.hpp"

 int main(int argc, char *argv[]){

    Kokkos::initialize(argc, argv);
    {  

        std::cout << "testing MATAR LU solver \n\n";

        size_t num_points = 3;
        DCArrayKokkos <double> A(num_points, num_points, "A");
        DCArrayKokkos <double> b(num_points, "b");

        // used for LU problem
        int singular; 
        int parity;
        DCArrayKokkos <size_t> perm (num_points, "perm");
        CArrayKokkos <double> vv(num_points, "vv");   // temp arrary for solver


        // --------------
        // --- test 1 ---
        // --------------
        A.set_values(0.0);
        b.set_values(0.0);

        // run the host functions
        RUN({
            A(0,0) = 1;
            A(1,1) = 2;
            A(2,2) = 3;

            b(0) = 1;
            b(1) = 2;
            b(2) = 3;
        });

        printf("A = \n");
        for(size_t i=0; i<num_points; i++){
            for(size_t j=0; j<num_points; j++){
                printf("%f ", A.host(i,j));
            }
            printf("\n");
        }
        printf("\n");


        singular = 0; 
        parity = 0;
        singular = LU_decompose_host(A, perm, vv, parity);  // A is returned as the LU matrix  
        if(singular==0){
            printf("ERROR: matrix is singluar \n");
            return 0;
        }

        LU_backsub_host(A, perm, b);  // note: answer is sent back in b
        b.update_host();

        printf("host executed routines \n");
        for(size_t i=0; i<num_points; i++){
            printf("x = %f \n", b.host(i));
        } // end for
        printf("exact = [1,1,1]^T \n\n");


        // run the device functions
        A.set_values(0.0);
        RUN({
            A(0,0) = 1;
            A(1,1) = 2;
            A(2,2) = 3;

            b(0) = 1;
            b(1) = 2;
            b(2) = 3;
        });
        A.update_host();
        b.update_host();


        RUN({
            int singular_d = 0; 
            int parity_d = 0;
            singular_d = LU_decompose(A, perm, vv, parity_d);  // A is returned as the LU matrix  
            if(singular_d==0){
                printf("ERROR: matrix is singluar \n");
            }

            LU_backsub(A, perm, b);  // note: answer is sent back in b
        });
        b.update_host();

        printf("device executed routines \n");
        for(size_t i=0; i<num_points; i++){
            printf("x = %f \n", b.host(i));
        } // end for
        printf("exact = [1,1,1]^T \n\n");


        // --------------
        // --- test 2 ---
        // --------------
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
        for(size_t i=0; i<num_points; i++){
            for(size_t j=0; j<num_points; j++){
                printf("%f ", A.host(i,j));
            }
            printf("\n");
        }
        printf("\n");

        singular = 0; 
        parity = 0;
        singular = LU_decompose_host(A, perm, vv, parity);  // A is returned as the LU matrix  
        if(singular==0){
            printf("ERROR: matrix is singluar \n");
            return 0;
        }

        LU_backsub_host(A, perm, b);  // note: answer is sent back in b
        b.update_host();

        printf("host executed routines \n");
        for(size_t i=0; i<num_points; i++){
            printf("x = %f \n", b.host(i));
        } // end for
        printf("exact = [1,2,3]^T \n\n");


        // run the device functions
        A.set_values(0.0);
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
        

        RUN({
            int singular_d = 0; 
            int parity_d = 0;
            singular_d = LU_decompose(A, perm, vv, parity_d);  // A is returned as the LU matrix  
            if(singular_d==0){
                printf("ERROR: matrix is singluar \n");
            }

            LU_backsub(A, perm, b);  // note: answer is sent back in b
        });
        b.update_host();

        printf("device executed routines \n");
        for(size_t i=0; i<num_points; i++){
            printf("x = %f \n", b.host(i));
        } // end for
        printf("exact = [1,2,3]^T \n\n");


        // --------------
        // --- test 3 ---
        // --------------
        A.set_values(0.0);
        b.set_values(0.0);

        RUN({
            A(0,0) = 1;
            A(0,1) = 2;
            A(0,2) = 3;
            A(1,2) = 6;
            A(2,1) = 4;
            A(2,2) = 5;

            b(0) = 14;
            b(1) = 18;
            b(2) = 23;
        });
        A.update_host();
        b.update_host();

        printf("A = \n");
        for(size_t i=0; i<num_points; i++){
            for(size_t j=0; j<num_points; j++){
                printf("%f ", A.host(i,j));
            }
            printf("\n");
        }
        printf("\n");


        singular = 0; 
        parity = 0;
        singular = LU_decompose_host(A, perm, vv, parity);  // A is returned as the LU matrix  
        if(singular==0){
            printf("ERROR: matrix is singluar \n");
            return 0;
        }

        LU_backsub_host(A, perm, b);  // note: answer is sent back in b
        b.update_host();

        printf("host executed routines \n");
        for(size_t i=0; i<num_points; i++){
            printf("x = %f \n", b.host(i));
        } // end for
        printf("exact = [1,2,3]^T \n\n");


        // run the device functions
        A.set_values(0.0);
        RUN({
            A(0,0) = 1;
            A(0,1) = 2;
            A(0,2) = 3;
            A(1,2) = 6;
            A(2,1) = 4;
            A(2,2) = 5;

            b(0) = 14;
            b(1) = 18;
            b(2) = 23;
        });
        A.update_host();
        b.update_host();
        

        RUN({
            int singular_d = 0; 
            int parity_d = 0;
            singular_d = LU_decompose(A, perm, vv, parity_d);  // A is returned as the LU matrix  
            if(singular_d==0){
                printf("ERROR: matrix is singluar \n");
            }

            LU_backsub(A, perm, b);  // note: answer is sent back in b
        });
        b.update_host();

        printf("device executed routines \n");
        for(size_t i=0; i<num_points; i++){
            printf("x = %f \n", b.host(i));
        } // end for
        printf("exact = [1,2,3]^T \n\n");



        // --------------
        // --- test 4 ---
        // --------------
        size_t num_vals = 10;
        DCArrayKokkos <double> M(num_vals,num_vals, "M");
        DCArrayKokkos <double> T_field(num_vals, "T_field");
        M.set_values(0.0);
        T_field.set_values(0.0);

        DCArrayKokkos <size_t> perm_T (num_vals, "perm_T");
        CArrayKokkos <double> vv_T (num_vals, "vv_T");

        RUN({
            M(0,0) = 2.0;
            M(0,1) = -1.0;

            M(num_vals-1, num_vals-2) = -1.0;
            M(num_vals-1, num_vals-1) = 2.0;

            // boundary conditions
            T_field(0) = 10.0;
            T_field(num_vals-1) = 1.0;
        });
        
        DO_ALL(i, 1, num_vals-2, {
            M(i, i-1) = -1.0;
            M(i,i) = 2.0;
            M(i, i+1) = -1.0;
        });
        M.update_host();
        T_field.update_host();

        printf("M = \n");
        for(size_t i=0; i<num_vals; i++){
            for(size_t j=0; j<num_vals; j++){
                printf("%f ", M.host(i,j));
            }
            printf("\n");
        }
        printf("\n");

        singular = 0; 
        parity = 0;
        singular = LU_decompose_host(M, perm_T, vv_T, parity);  // A is returned as the LU matrix  
        if(singular==0){
            printf("ERROR: matrix is singluar \n");
            return 0;
        }

        LU_backsub_host(M, perm_T, T_field);  // note: answer is sent back in b
        T_field.update_host();

        printf("host executed routines \n");
        for(size_t i=0; i<num_vals; i++){
            printf("Temp_field = %f \n", T_field.host(i));
        } // end for


        // run the device functions
        M.set_values(0.0);
        T_field.set_values(0.0);
        RUN({
            M(0,0) = 2.0;
            M(0,1) = -1.0;

            M(num_vals-1, num_vals-2) = -1.0;
            M(num_vals-1, num_vals-1) = 2.0;

            // boundary conditions
            T_field(0) = 10.0;
            T_field(num_vals-1) = 1.0;
        });
        
        DO_ALL(i, 1, num_vals-2, {
            M(i, i-1) = -1.0;
            M(i,i) = 2.0;
            M(i, i+1) = -1.0;
        });
        M.update_host();
        T_field.update_host();


        RUN({
            int singular_d = 0; 
            int parity_d = 0;
            singular_d = LU_decompose(M, perm_T, vv_T, parity_d);  // A is returned as the LU matrix  
            if(singular_d==0){
                printf("ERROR: matrix is singluar \n");
            }

            LU_backsub(M, perm_T, T_field);  // note: answer is sent back in b
        });
        T_field.update_host();

        printf("device executed routines \n");
        for(size_t i=0; i<num_vals; i++){
            printf("Temp_field = %f \n", T_field.host(i));
        } // end for


    } // end of kokkos scope



    Kokkos::finalize();

    return 1;

 } // end function

