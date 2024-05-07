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
// 
// This file contains simple examples to use MATAR
//
// To set the number of OpenMP threads to 6
//    export OMP_NUM_THREADS=6
//
// Data type naming convetions:
//
//  Allocation memory or view existin memory:
//   View  = map data to multi-D or slice out data, no memory allocation on CPU 
//   Blank = allocate memory on the CPU, GPU, or Both depending on specified location
//
//  Index convetion:
//   F = Fortran index convention, fast index varies the quickest
//   C = C index convetion, last index varies the quickest
//   RaggedC = dense in first index, second index varies in size
//   RaggedF  = first index varies in size, second index is dense
//   DynamicRaggedC = dense in first index, second index varies in size dynamically
//   DynamicRaggedF  = first index varies in size dynamically, second index is dense
//   CSR = compressed row storage
//   CSC = compressed column storage
//
//   Matrix = index goes from 1 to N
//   Array  = index goes from 0 to less than N
//
//  Location of type:
//   Host = data always lives on the CPU
//   Device = data is on the GPU if building for a GPU or on a CPU if building for a CPU 
//   Dual = data lives on GPU and on CPU
//   blank = if nothing is specified, data always lives on the CPU
//
//  Putting it all together, a name will be:
//   View/blank C/F Matrix/Array Host/Device/Dual  <data_type>  variable_name (dimensions...)
//
//  Examples that allocate memory:
//   CMatrixHost <int> My4DTensor(3,3,3,3);  // C convetion, indices 1:N, always on the CPU, integers
//   CMatrix <int> My4DTensor(3,3,3,3);      // same as CMatrixHost
//
//   FArrayDevice <double> My2DMatrix(10,10) // F convetion, indices 0:<N, always on the device, double
//   CArrayDual <double> My2DMatrix(10,10)   // F convetion, indices 0:<N, always on the CPU and device, double   
//
//  Examples that view allocated memory:
//   ViewFMatrixHost <double> My2DMatrix(&A[0],10,10)  // View existing data, F convetion, indices 1:N, always on the CPU, double  
//   ViewCArrayDevice <double> My2DMatrix(&A[0],10,10) // View existing data, C convetion, indices 0:<N, always on the Device, double  
//   ViewCArrayDual <double> My2DMatrix(&A[0],10,10)   // View existing data, C convetion, indices 0:<N, always on CPU and device, double  


#include <stdio.h>  // for outputs
#include <chrono>   // for timing

#include "matar.h"

using namespace mtr; // matar namespace 



// main
int main(int argc, char *argv[]) {


    Kokkos::initialize(argc, argv);
    {  

        std::cout << "testing MATAR \n";

        // Matrix examples following the Fortran index convention,
        // indicies go from 1 to N, first index varies the fastest
        FMatrixDevice <real_t> matrix1D(10);    // declare and allocate a 1D matrix of size 10
        FMatrixDevice <real_t> matrix2D(10,10); // declare and allocate a 2D matrix with sizes of 10 x 10

        FMatrixDevice <real_t> matrix3D;        // declare variable and allocate sizes and dimensions later
        matrix3D = FMatrixDevice <real_t> (10,10,10); // allocate dimensions and sizes 

        // Array example following the Fortran index convention,
        // indicies go from 0 to less than N, last index varies the fastest
        FArrayDevice <int> arr3D(10,10,10);


        // Initialize matrix1D
        DO_ALL (i, 1, 10, {
                matrix1D(i) = 1.0;
        }); // end parallel do

        
        // Initialize matrix2D
        DO_ALL (j, 1, 10,
                i, 1, 10, {
                matrix2D(i,j) = 2.0;
        }); // end parallel do

        
        // Initialize matrix3D
        DO_ALL (k, 1, 10,
                j, 1, 10,
                i, 1, 10, {
                matrix3D(i,j,k) = 3.0;

                arr3D(i-1,j-1,k-1) = 1 + i*j*k;
        }); // end parallel do

        
        int result;
        int loc_max;
        DO_REDUCE_MAX(k, 0, 10,
                      j, 0, 10,
                      i, 0, 10,
                      loc_max, {

                if (loc_max < arr3D(i,j,k)){
                    loc_max = arr3D(i,j,k);
                } // end if

        }, result);

        printf("3D reduce MAX %i\n", result);


        // ===============
        

        int N=200;  // array dimensions are NxN

        // A 2D array example following the C index convention
        // indicies go from 0 to less than N, last index varies the fastest
        CArrayDevice <double> A(N,N); // dense array
        CArrayDevice <double> B(N,N);
        CArrayDevice <double> C(N,N);
        CArrayDevice <double> D(N,N);

        CArrayDevice <double> L(N,N); // lower triangular array
        CArrayDevice <double> U(N,N); // upper triangular array
        CArrayDevice <double> x(N);
        CArrayDevice <double> y(N);
        

        auto time_1 = std::chrono::high_resolution_clock::now();

        // ininitialize arrays
        FOR_ALL (i, 0, N,
                 j, 0, N,{

            A(i,j) = 1.0; 
            B(i,j) = 2.0;

        });

        FOR_ALL (i, 0, N,
                 j, 0, N,{

            if (j <=i){
                L(i,j) = 3.0; 
                U(i,j) = 0.0; 
            }
            else{
                L(i,j) = 0.0;
                U(i,j) = 4.0;
            } // end if

        });

        FOR_ALL (i, 0, N, {
            y(i) = 4.0;
        });

        
        // Add two arrays together
        // C = A+B
        FOR_ALL (i, 0, N,
                 j, 0, N,{

            C(i,j) = A(i,j) + B(i,j);

        });

        // Multiply two arrays together
        // D = A*B
        FOR_ALL (i, 0, N,
                 j, 0, N,
                 k, 0, N,{

            D(i,j) = A(i,k)*B(k,j);

        });

        // backwards substitution
        for (int k = N-1; k>=0; k--){

            x(k) = y(k);
            
            int loc_sum;
            int result;
            // calculate dot product
            if(k<N-1){
                REDUCE_SUM(i, k, N-1,
                           loc_sum, {
                        loc_sum += U(k,i)*x(i);
                }, result);
            } // end if
            x(k) -= result;
            x(k) /= U(k,k);
        } // end for k backwards


        // forward substitution
        for (int i = 0; i<N; i++){

            int loc_sum;
            int result;
            // calculate dot product
            if(i-1>0){
                REDUCE_SUM(j, 0, i-1,
                           loc_sum, {
                        loc_sum += L(i,j)*x(j);
                }, result);
            }

            x(i) = (y(i)- result)/U(i,i);
        } // end for i  


        auto time_2 = std::chrono::high_resolution_clock::now();

        std::chrono::duration <double, std::milli> ms = time_2 - time_1;
        std::cout << ms.count() << "ms\n";
 

    } // end of kokkos scope

    Kokkos::finalize();

    return 1;

} //end of main




