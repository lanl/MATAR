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
#include <iostream>
#include <stdio.h>
#include <cmath>


#include "matar.h"
using namespace mtr;


const double TINY = 1e-15;

// ---------------------------
// LU decomposition functions
// ---------------------------


// the function is run on the GPU
KOKKOS_FUNCTION
int LU_decompose(
    const DCArrayKokkos <double> &A, // matrix A passed in and is sent out in LU decomp format
    const DCArrayKokkos <size_t> &perm,  // permutations
    int &parity) {                 // parity (+1 or -1)
                          
    const int n = A.dims(0);  // size of matrix 

    CArrayKokkos <double> vv(n);   // temp arrary for solver

    parity = 1;

    // helper variables
    double temp;
    
    // search for the largest element in each row; save the scaling in the 
    // temporary array vv and return zero if the matrix is singular 
    for(size_t i = 0; i < n; i++) {
        
        double big = 0.;
        for(size_t j = 0; j < n; j++){
            if((temp=fabs(A(i,j))) > big){
                big=temp;
            }
        }
        
        if(big == 0.0) return(0);
        
        vv(i) = big;
    }

    // the main loop for the Crout's algorithm
    for(size_t j = 0; j < n; j++) {
        
        // this is the part a) of the algorithm except for i==j 
        for(size_t i=0;i<j;i++) {
            
            double sum=A(i,j);
            
            for(size_t k=0;k<i;k++){
                sum -= A(i,k)*A(k,j);
            }

            A(i,j) = sum;
        }
    
        // initialize for the search for the largest pivot element
        double big = 0.;
        size_t imax = j;
        
        
        // this is the part a) for i==j and part b) for i>j + pivot search 
        for(size_t i = j; i < n; i++) {
            
            double sum = A(i,j);
            
            for(size_t k=0; k<j; k++){
                sum -= A(i,k)*A(k,j);
            }
            
            A(i,j) = sum;
            
            // is the figure of merit for the pivot better than the best so far?
            if((temp = vv(i)*fabs(sum)) >= big){
                big = temp; 
                imax = i;
            }
        } // end for i

        // interchange rows, if needed, change parity and the scale factor
        if(imax != j) {
            
            for(size_t k = 0; k < n; k++){
                temp = A(imax,k);
                A(imax,k) = A(j,k);
                A(j,k) = temp;
            }
            
            parity = -(parity);
            vv(imax) = vv(j);
        }
        
        // store the index
        perm(j) = imax;
        // if the pivot element is zero, the matrix is singular but for some 
        // applications a tiny number is desirable instead 
        
        if(A(j,j) == 0.0){
            A(j,j) = TINY;
        }
        // finally, divide by the pivot element
        
        if(j<n-1) {
            
            temp=1./A(j,j);
            for(size_t i = j+1; i < n; i++){
                A(i,j)*=temp;
            } // end for i
        } // end if j

    } // end for j
    
    return(1);
} // end function

// the function is run from the host, and kernals
// inside this function are run on the GPU
int LU_decompose_host(
    DCArrayKokkos <double> &A, // matrix A passed in and is sent out in LU decomp format
    DCArrayKokkos <size_t> &perm,  // permutations
    int &parity) {                 // parity (+1 or -1)
                          
    const int n = A.dims(0);  // size of matrix 

    CArrayKokkos <double> vv(n);   // temp arrary for solver

    parity = 1;
    
    // STEP 1:
    // search for the largest element in each row; save the scaling in the 
    // temporary array vv and return zero if the matrix is singular
    FOR_FIRST(i, 0, n, {
        
        double max_val = 0.0;
        double max_val_lcl = 0.0;

        FOR_REDUCE_MAX_SECOND(j, 0, n, 
                              max_val_lcl, {

            max_val_lcl = fmax(max_val_lcl, fabs(A(i,j)) );

        }, max_val); // end parallel j
        Kokkos::fence();
        
        vv(i) = max_val;       

    }); // end for

    // if the largest value in the array row is 0, then exit
    double min_val = 0.0;
    double min_val_lcl = 0.0; 
    FOR_REDUCE_MIN(i, 0, n, 
                   min_val_lcl, {

        min_val_lcl = fmin(vv(i), min_val_lcl);

    }, min_val);
    if(min_val < TINY) return(0); // singular matrix as all row values are 0


    // STEP 2:
    // the main loop for the Crout's algorithm
    for(size_t j = 0; j < n; j++) {
        
        // this is the part a) of the algorithm except for i==j 
        FOR_FIRST(i, 0, j, {
            
            double sum = 0.0;
            double sum_lcl = 0.0;
            
            FOR_REDUCE_SUM_SECOND(k, 0, i, 
                                  sum_lcl, {

                sum_lcl -= A(i,k)*A(k,j);

            }, sum); // end parallel k
            Kokkos::fence();

            sum += A(i,j);

            A(i,j) = sum;
        }); // end parallel
    
        
        // this is the part a) for i==j and part b) for i>j
        // loop is from i=j to i<n 
        FOR_FIRST(i, j, n, {
            
            double sum = 0.0;
            double sum_lcl = 0.0;

            FOR_REDUCE_SUM_SECOND(k, 0, j, 
                                  sum_lcl, {

                sum_lcl -= A(i,k)*A(k,j);

            }, sum); // parallel k
            Kokkos::fence();

            sum += A(i,j);

            A(i,j) = sum;
        }); // end parallel
        Kokkos::fence();

        // initialize the search for the largest pivot element

        double max_val = 0.0;
        double max_val_lcl = 0.0;
        // loop is from i=j to i<n
        FOR_REDUCE_MAX(i, j, n, 
                       max_val_lcl, {
            
            // is the figure of merit for the pivot better than the best so far? 
            if( vv(i)*fabs(A(i,j)) >= max_val_lcl) {
                max_val_lcl = vv(i)*fabs(A(i,j));
            } // end if

        }, max_val); // end for i
        Kokkos::fence();

        size_t imax = j;
        size_t imax_lcl = j;
        // loop is from i=j to i<n
        FOR_REDUCE_MAX(i, j, n, 
                       imax_lcl, {
            
            // is the figure of merit for the pivot better than the best so far? 
            if( vv(i)*fabs(A(i,j)) >= max_val ) {
                imax_lcl = i;
            } // end if

        }, imax); // end for i
        Kokkos::fence();

        // interchange rows, if needed, change parity and the scale factor
        if(imax != j) {
            
            FOR_ALL(k, 0, n, {
                double temp = A(imax,k);
                A(imax,k) = A(j,k);
                A(j,k) = temp;
            });
            
            parity = -parity;
            RUN({
                vv(imax) = vv(j);
            });

        } // end if
        
        // store the index
        RUN({
            perm(j) = imax;
        });


        // if the pivot element is zero, the matrix is singular but for some 
        // applications a tiny number is desirable instead 
        RUN({
            if(A(j,j) == 0.0){
                A(j,j) = TINY;
            }
        });
        
        
        // finally, divide by the pivot element 
        if(j<n-1) {
            // loop is from i=j+1 to i<n
            FOR_ALL(i, j+1, n, {
                A(i,j) *= 1.0/A(j,j);
            });

        } // end if

    } // end for j
    
    return(1);
}


// -------------------------------
// LU back substitution functions 
// -------------------------------

// this function is run on the GPU
KOKKOS_FUNCTION
void LU_backsub(
    const DCArrayKokkos <double> &A,     // input matrix A in LU decomp format
    const DCArrayKokkos <size_t> &perm,  // permutations
    const DCArrayKokkos <double> &b){          // RHS and is answer x to Ax=B

        const int n = A.dims(0);    // size of matrix

        int ii = -1;


        // First step of backsubstitution; the only wrinkle is to unscramble 
        // the permutation order. Note: the algorithm is optimized for a 
        // possibility of large amount of zeroes in b
        
        for(size_t i = 0; i < n; i++) {
           
            size_t ip = perm(i);

            double sum = b(ip);
            b(ip) = b(i);
         
            if(ii >= 0){
                for(size_t j = ii; j<i; j++){
                    sum -= A(i,j)*b(j);
                }
            }
            else if(sum>0){
                ii=i;  // a nonzero element encounted
            }
          
            b(i) = sum;
        } // end loop i
        
        // the second step
        for(int i=n-1; i>=0; i--) {
            
            double sum = b(i);
            for(size_t j=i+1; j<n; j++){
                sum-=A(i,j)*b(j);
            } // end j
       
            b(i)=sum/A(i,i);
        } // end loop i

} // end if



// the function is run from the host, and kernals
// inside this function are run on the GPU
void LU_backsub_host(
    const DCArrayKokkos <double> &A,     // input matrix A in LU decomp format
    const DCArrayKokkos <size_t> &perm,  // permutations
    DCArrayKokkos <double> &b){          // RHS and is answer x to Ax=B

    const int n = A.dims(0);    // size of matrix

    
    // First step of backsubstitution; the only wrinkle is to unscramble 
    // the permutation order. Note, the algorithm is optimized for a 
    // possibility of large amount of zeroes in b 

    // Forward substitution: solve L x = P b
    int ii = -1;
    for(size_t i = 0; i < n; i++) {
           
            size_t ip = perm(i);

            double sum = b(ip);
            double sum_tally = 0.0;
            b(ip) = b(i);
         
            if(ii >= 0){

                double sum_lcl = 0.0;

                // j=ii to j<i
                FOR_REDUCE_SUM(j, ii, i, 
                               sum_lcl, {
                    sum_lcl -= A(i,j)*b(j);
                }, sum_tally);
                Kokkos::fence();
                
                sum += sum_tally;
            }
            else if(sum>0){
                ii=i;  // a nonzero element encounted
            }
          
            b(i) = sum;
    } // end loop i


     // the second step
     // Backward substitution: solve U b = x
    for(int i=n-1; i>=0; i--){

        double sum = 0.0;
        double sum_lcl = 0.0;

        // for j=i+1 to j<N
        FOR_REDUCE_SUM(j, i+1, n, 
                       sum_lcl, {
            sum_lcl -= A(i,j)*b(j);
        }, sum);

        sum += b(i);

        b(i) = sum/A(i,i);
    } // end for i       

} // end LU backsubstitution on lauched from host

#endif // LUSOLVER