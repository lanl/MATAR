#include <iostream>
#include <stdio.h>
#include <iomanip> // Include this header for std::fixed and std::setprecision
#include <stdlib.h>
#include <Kokkos_Core.hpp>
#include <sys/stat.h>
#include <chrono> // for std::chrono functions

#include "matar.h"
#include <vector>

using namespace mtr;
// using namespace std;

#define TINY 1.e-16

// int omp_thread_count() {
//     int n = 0;
//     #pragma omp parallel reduction(+:n)
//     n += 1;
//     return n;
// }


// Timer class for timing function calls
class Timer
{
private:
    // Type aliases to make accessing nested type easier
    using Clock = std::chrono::steady_clock;
    using Second = std::chrono::duration<double, std::ratio<1> >;
    
    std::chrono::time_point<Clock> m_beg { Clock::now() };

public:
    void reset()
    {
        m_beg = Clock::now();
    }
    
    double elapsed() const
    {
        return std::chrono::duration_cast<Second>(Clock::now() - m_beg).count();
    }
};

// Helper function for printing matar matrices
void print_mat_matar(DCArrayKokkos<double> &mat){
    mat.update_host();
    printf("**************************************************\n");

    // ************ WARNING: CUDA MAY NOT LIKE THIS  ************ 
    // RUN({

        for(int i = 0; i < mat.dims(0); i++){
            std::cout<<std::endl;
            for(int j = 0; j < mat.dims(1); j++){
                printf("%.4f    ", mat.host(i,j));
            }
            printf("\n");
        }
        std::cout<<std::endl;
    // });  // end RUN
    printf("\n");
    printf("**************************************************\n");
    printf("\n");
}


// Helper function for printing matar matrices
void print_mat_vec(std::vector<std::vector<double>> &mat){

    printf("**************************************************\n");

    // ************ WARNING: CUDA MAY NOT LIKE THIS  ************ 
    // RUN({

        for(int i = 0; i < mat[0].size(); i++){
            std::cout<<std::endl;
            for(int j = 0; j < mat[1].size(); j++){
                printf("%.4f    ", mat[i][j]);
                // std::cout<<mat.host(i,j)<<"  ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    // });  // end RUN
    printf("\n");
    printf("**************************************************\n");
    printf("\n");
}

/* ------------------------- */
/* LU decomposition function */
/* ------------------------- */

/* 
LU-decomposition according to Crout's algorithm with pivoting. 
Description:
int LU_decompos(
    ViewCArray <real_t> &source_mat,  // matrix to invert
    ViewCArray <int> &indx,           // permutations
    int &parity,                        // parity (+1 0r -1)
    const int n);                       // matrix size                   


Parameters:
source_mat - source matrix (n x n) on input, destination on output;
n - the matrix size;
indx - integer array (size n) to remember permutations;
d - on output, contains +1 or -1 for even or odd permutations number.
vv - temporary array (size n).
Returns: 
0 - the source matrix is singular (invalid for decomposition),
1 - if OK.
*/
int lu_decomp_PORTABLE(
    DCArrayKokkos<double> &source_mat,
    DCArrayKokkos<int> &indx,
    int &parity, 
    const int n)
{

    DCArrayKokkos<double> vv = DCArrayKokkos<double>(n); // temp arrary for solver

    parity = 1;
    /* search for the largest element in each row; save the scaling in the
    temporary array vv and return zero if the matrix is singular */

    CArrayKokkos <double> extra_double_var(2);  // big1 = extra_double_var(0), temp1 = extra_double_var(1)
    DCArrayKokkos <int> extra_int_var(2); // Singular flag and parity int

    RUN({
        extra_int_var(0) = 1; // singular
        extra_int_var(1) = 1; // parity
    });
    RUN({
        for(int i = 0; i < n; i++){
            double *big1 = &extra_double_var(0);
            double *temp1 = &extra_double_var(1);

            *big1 = 0.;
            for(int j = 0; j < n; j++) if((*temp1=fabs(source_mat(i,j))) > *big1) *big1=*temp1;
            
            if (*big1 == 0.) {
                extra_int_var(0) = 0;
            }
            vv(i) = *big1;
        }
    });

    // Return if singular
    extra_int_var.update_host();
    if(extra_int_var.host(0) == 0) return(0);

    /* the main loop for the Crout's algorithm */
    CArrayKokkos <int> int_vars(1);
    CArrayKokkos <double> double_vars(2);
    RUN({
        int *imax = &int_vars(0);

        double *big = &double_vars(0);
        double *temp = &double_vars(1);

        for (int j = 0; j < n; j++) {
            /* this is the part a) of the algorithm except for i==j */
            for (int i = 0; i < j; i++) {
                double sum = source_mat(i, j);

                for (int k = 0; k < i; k++) {
                    sum -= source_mat(i, k) * source_mat(k, j);
                }
                source_mat(i, j) = sum;
            }

            /* initialize for the search for the largest pivot element */
            *big  = 0.;
            *imax = j;

            /* this is the part a) for i==j and part b) for i>j + pivot search */
            for (int i = j; i < n; i++) {
                double sum = source_mat(i, j);

                for (int k = 0; k < j; k++) {
                    sum -= source_mat(i, k) * source_mat(k, j);
                }

                source_mat(i, j) = sum;

                /* is the figure of merit for the pivot better than the best so far? */
                if ((*temp = vv(i) * fabs(sum)) >= *big) {
                    *big = *temp; *imax = i;
                }
            }

            /* interchange rows, if needed, change parity and the scale factor */
            if (*imax != j) {
                for (int k = 0; k < n; k++) {
                    *temp = source_mat(*imax, k);
                    source_mat(*imax, k) = source_mat(j, k);
                    source_mat(j, k)    = *temp;
                }

                extra_int_var(1)   = -(extra_int_var(1));
                vv(*imax) = vv(j);
            }

            /* store the index */
            indx(j) = *imax;
            /* if the pivot element is zero, the matrix is singular but for some
            applications a tiny number is desirable instead */

            if (source_mat(j, j) == 0.) {
                source_mat(j, j) = TINY;
            }
            /* finally, divide by the pivot element */

            if (j < n - 1) {
                *temp = 1. / source_mat(j, j);
                for (int i = j + 1; i < n; i++) {
                    source_mat(i, j) *= *temp;
                }
            }
        }
    });  // end RUN

    extra_int_var.update_host();
    parity = extra_int_var(1);

    return(extra_int_var(0));
}

/* ------------------------- */
/* LU decomposition function */
/* ------------------------- */
int lu_decomp(
    std::vector<std::vector<double>>  &source_mat,
    std::vector<int> &indx,           // permutations
    int &parity,                      // parity (+1 0r -1)
    const int n) // size of matrix
{                      

    int i, imax, j, k; // Indexing 

    real_t big, sum, temp;// useful storage

    real_t vv[n];       // temp arrary for solver

    parity = 1;
    /* search for the largest element in each row; save the scaling in the 
    temporary array vv and return zero if the matrix is singular */
    for(i = 0; i < n; i++) {
        
        big = 0.;
        for(j = 0; j < n; j++) if((temp=fabs(source_mat[i][j])) > big) big=temp;
        
        if(big == 0.) return(0);
        
        vv[i] = big;
    }

    /* the main loop for the Crout's algorithm */
    for(j = 0; j < n; j++) {
        
        /* this is the part a) of the algorithm except for i==j */
        for(i=0;i<j;i++) {
            
            sum=source_mat[i][j];
            
            for(k=0;k<i;k++) sum -= source_mat[i][k]*source_mat[k][j];

            source_mat[i][j] = sum;
        }
    
        /* initialize for the search for the largest pivot element */
        big = 0.;
        imax = j;
        
        /* this is the part a) for i==j and part b) for i>j + pivot search */
        for(i = j; i < n; i++) {
            
            sum = source_mat[i][j];
            
            for(k=0; k<j; k++) sum -= source_mat[i][k]*source_mat[k][j];
            
            source_mat[i][j] = sum;
            
            /* is the figure of merit for the pivot better than the best so far? */
            if((temp = vv[i]*fabs(sum)) >= big) {big = temp; imax = i;}
        }

        /* interchange rows, if needed, change parity and the scale factor */
        if(imax != j) {
            
            for(k = 0; k < n; k++){
                temp = source_mat[imax][k];
                source_mat[imax][k] = source_mat[j][k];
                source_mat[j][k] = temp;
            }
            
            parity = -(parity);
            vv[imax] = vv[j];
        }
        
        /* store the index */
        indx[j] = imax;
        /* if the pivot element is zero, the matrix is singular but for some 
        applications a tiny number is desirable instead */
        
        if(source_mat[j][j] == 0.) source_mat[j][j] = TINY;
        /* finally, divide by the pivot element */
        
        if(j<n-1) {
            
            temp=1./source_mat[j][j];
            for(i = j+1; i < n; i++) source_mat[i][j]*=temp;
        }
    }
    
    return(1);
}


/* ----------------------------- */
/* LU back substitution function */
/* ----------------------------- */
/* 
Back substitution, using LU decomposed matrix.
Description:
void LU_backsub(
    ViewCArray <real_t> &source_mat,  // input matrix
    ViewCArray <int> &indx,           // permutations
    ViewCArray <real_t> &coeffs,      // Least squares coefficents
    const int n);

Parameters:
source_mat - the matrix decomposed by Crout;
n - the matrix size;
indx - permutation order obtained by decomposition algorithm;
coeffs - the vector (size n) to be substituted on input, the result
of the substitution on output.
Note: a and indx are not modified by this routine and could be 
used in multiple calls.
*/
void lu_backsub_PORTABLE(
    DCArrayKokkos<double> &mat, 
    DCArrayKokkos<int> &indx,
    DCArrayKokkos<double> &col_vec,
    int not_used)
{
    
    CArrayKokkos <int> int_vars(3); // int j, ip, ii = -1;
    CArrayKokkos <double> double_vars(1); // double sum;

    RUN({
        int_vars(0) = -1;
        int_vars(1) = -1;
        int_vars(2) = -1;
        double_vars(0) = 0.0;
    });

    /* First step of backsubstitution; the only wrinkle is to unscramble
    the permutation order. Note: the algorithm is optimized for a
    possibility of large amount of zeroes in b */
    RUN({

        int n = mat.dims(0);
        int *j = &int_vars(0);
        int *ip = &int_vars(1);
        int *ii = &int_vars(2);

        double *sum = &double_vars(0);

        for (int i = 0; i < n; i++) {
            *ip = indx(i);

            *sum = col_vec(*ip);
            col_vec(*ip) = col_vec(i);

            if (*ii >= 0) {
                for (*j = *ii; *j < i; (*j)++) {
                    *sum -= mat(i, *j) * col_vec(*j);
                }
            }
            else if (*sum) {
                *ii = i;             /* a nonzero element encounted */
            }
            col_vec(i) = *sum;
        }

        /* the second step */
        for (int i = n - 1; i >= 0; i--) {
            *sum = col_vec(i);

            // std::cout<<"i = "<<i<<std::endl;
            // std::cout<<"j = "<<*j<<std::endl;

            for (*j = i + 1; *j < n; (*j)++) {

                *sum -= mat(i, *j) * col_vec(*j);
            }

            col_vec(i) = *sum / mat(i, i);
        }
    });  // end RUN
}

/* ----------------------------- */
/* LU back substitution function */
/* ----------------------------- */
void LU_backsub(
    std::vector<std::vector<double>> &source_mat,  // input matrix
    std::vector<int> &indx,        // permutations
    std::vector<double> &b,      // Least squares coefficents
    int n) 
{
    
    int i, j, ip, ii = -1;
    double sum;
    
    /* First step of backsubstitution; the only wrinkle is to unscramble 
    the permutation order. Note: the algorithm is optimized for a 
    possibility of large amount of zeroes in b */
    
    for(i = 0; i < n; i++) {
        
        ip = indx[i];
        sum = b[ip];
        b[ip] = b[i];

        
        if(ii >= 0) for(j = ii; j<i; j++) sum -= source_mat[i][j]*b[j];
        else if(sum) ii=i;  /* a nonzero element encounted */
        
        b[i] = sum;
    }
    
    /* the second step */
    for(i=n-1; i>=0; i--) {
        
        sum = b[i];
        for(j=i+1; j<n; j++) sum-=source_mat[i][j]*b[j];
        
        b[i]=sum/source_mat[i][i];
    }    
}


void lu_invert_PORTABLE(
    DCArrayKokkos<double> &mat, 
    DCArrayKokkos<double> &mat_inv, 
    DCArrayKokkos<double> &col_vec,
    DCArrayKokkos<int> &indx )
{
    int n = mat.dims(0);

    for (int j = 0; j < n; j++) {
        // Initialize col to zero
        FOR_ALL(i, 0, n, {
            col_vec(i) = 0.0;
        });

        col_vec(j) = 1.0;
        lu_backsub_PORTABLE(mat, indx, col_vec, n);

        FOR_ALL(i, 0, n, {
            mat_inv(i, j) = col_vec(i);
        });
    }
}

/* ------------------ */
/* LU invert function */
/* ------------------ */
void lu_invert(
    std::vector<std::vector<double>> &source_mat,  // input matrix
    std::vector<std::vector<double>> &inv_mat,     // inverse matrix
    std::vector<double>  &col,         // tmp array
    std::vector<int>  &indx,        // permutations
    int n) {


    for(int j = 0; j < n; j++){
        for(int i = 0; i < n; i++) col[i] = 0.0;
        
        col[j] = 1.0;
        LU_backsub(source_mat, indx, col, n);
        
        for(int i = 0; i < n; i++) inv_mat[i][j] = col[i];
    }
}

// Invert a MATAR matrix using LU decomposition and back substitution
void invert_PORTABLE(
    DCArrayKokkos<double> &mat, 
    DCArrayKokkos<double> &mat_inv)
{
    // Assumes a square matrix
    int matrix_size = mat.dims(0);

    DCArrayKokkos<double> MAT_LU = DCArrayKokkos<double>(matrix_size, matrix_size, "input_matrix_LU_decomp");

    // Copy matrix into MAT_LU
    FOR_ALL (i, 0, matrix_size,
             j, 0, matrix_size,{

        MAT_LU(i,j) = mat(i,j);
    });

    // Intermediate data
    DCArrayKokkos<double> col_vec = DCArrayKokkos<double>(matrix_size, "least_squares_coeffs");
    DCArrayKokkos<int> indx(matrix_size, "permutations");
    int parity = 0;
    
    FOR_ALL (i, 0, matrix_size, {
        indx(i) = 0;
        col_vec(i) = 0.0;
    });

    int singular = 1;
    singular = lu_decomp_PORTABLE(MAT_LU, indx, parity, matrix_size); 

    if(singular == 0) std::cout<<"WARNING: SINGULAR MATRIX"<<std::endl;

    lu_invert_PORTABLE(MAT_LU, mat_inv, col_vec, indx);
}

// Invert a matrix using LU decomposition and back substitution
void invert(
    std::vector<std::vector<double>>& mat, 
    std::vector<std::vector<double>>& mat_inv)
{
    // Assumes a square matrix
    int matrix_size = mat[0].size();

    std::vector<std::vector<double>> MAT_LU(matrix_size, std::vector<double>(matrix_size));

    // Copy matrix into MAT_LU
    for(int i = 0; i < matrix_size; i++){
        for(int j = 0; j < matrix_size; j++){
            MAT_LU[i][j] = mat[i][j];
        }
    }

    // Intermediate data
    std::vector<double> col_vec(matrix_size);
    std::vector<int> indx(matrix_size);

    for(int i = 0; i < matrix_size; i++){
        indx[i] = 0;
        col_vec[i] = 0.0;
    }

    int parity = 0;
 
    int singular = 1;
    singular = lu_decomp(MAT_LU, indx, parity, matrix_size); 

    if(singular == 0) std::cout<<"WARNING: SINGULAR MATRIX"<<std::endl;

    lu_invert(MAT_LU, mat_inv, col_vec, indx, matrix_size);
}

// Matrix multiplication C=A*B, used to verify that A*A^{-1} = I
KOKKOS_FUNCTION
template<typename T1>
void rm_dense_matmul_device( T1& C, T1& A, T1& B)
{
    assert(C.order() == A.order() == B.order() == 2 && "Tensor order (rank) does not match 2 in matrix_multiply");
    assert(C.dims(0) == A.dims(0) == B.dims(1) && "Tensor index 0 length mismatch in in serial_matrix_multiply");

    int i_ext = A.dims(0);
    int j_ext = A.dims(1);
    int k_ext = B.dims(1);

    for(int i = 0; i < i_ext; i++){
        for(int k = 0; k < k_ext; k++){
            for(int j = 0; j < j_ext; j++){
                C(i,k) += A(i,j)*B(j,k);
            }
        }
    }
}

void test_MATAR_invert(int num_tests, bool print_matrices, bool verify, Timer t, int start){
    // Timing variable
    double runtime = 0.0;

    std::cout << "**** Testing LU Inverse using MATAR **** " << std::endl;
    // std::cout << "OpenMP  Num threads =  "<< omp_thread_count() << std::endl;

    for(int i = start; i < num_tests; i++){

        int matrix_size = pow(2,i);
        std::cout << "Matrix inversion size  "<<matrix_size<<"X"<<matrix_size << std::endl;

        // Create and initialize matrix
        DCArrayKokkos<double> MAT = DCArrayKokkos<double>(matrix_size, matrix_size, "input_matrix");
        DCArrayKokkos<double> MAT_INV = DCArrayKokkos<double>(matrix_size, matrix_size, "input_matrix_inverse");

        // Initialize matrices to be non-symmetric
        FOR_ALL (i, 0, MAT.dims(0),
                 j, 0, MAT.dims(1),{
            
            if(i > j){
                MAT(i,j) = (double)i - (double)j;
                MAT_INV(i,j) = (double)i - (double)j;
            }
            if(i < j){
                MAT(i,j) = (double)i + (double)j + 1.0;
                MAT_INV(i,j) = (double)i + (double)j + 1.0;
            }

            if(i == j){
                MAT(i,j) = (double)i + (double)j + 1.0;
                MAT_INV(i,j) = (double)i + (double)j + 1.0;
            }

        });

        // Print matrix
        if(print_matrices){
            std::cout<<"Printing MAT"<<std::endl;
            print_mat_matar(MAT);
        }

        // Reset timer
        t.reset();

        // Invert MAT, store in MAT_INV
        invert_PORTABLE(MAT, MAT_INV);

        // Save runtime for LU inverse
        runtime = t.elapsed();

        if(print_matrices){
            std::cout<<"Printing MAT_INV"<<std::endl;
            print_mat_matar(MAT_INV);
        }


        if(verify){
            DCArrayKokkos<double> MAT_TEST = DCArrayKokkos<double>(matrix_size, matrix_size, "test_matrix");

            RUN({
                int n = MAT_TEST.dims(0);
                for(int i=0; i<n; i++){
                    for(int j=0; j<n; j++){
                        MAT_TEST(i,j) = 0.0;
                    }
                }
                rm_dense_matmul_device(MAT_TEST, MAT, MAT_INV);
            });

            if(print_matrices){
                std::cout<<"Printing MAT_TEST: MAT*MAT_INV"<<std::endl;
                print_mat_matar(MAT_TEST);
            }

            RUN({
                int n = MAT_TEST.dims(0);
                for(int i=0; i<n; i++){
                    for(int j=0; j<n; j++){
                        
                        if(i == j){
                            if( fabs(MAT_TEST(i,j) - 1.0) >= 1E-8) std::cout<<"WRONG IDENTITY"<<std::endl;
                        }
                        else{
                            if(fabs(MAT_TEST(i,j)) >= 1E-8) std::cout<<"WRONG OFF DIAGONAL = "<<MAT_TEST(i,j)<<std::endl;
                        }
                        

                    }
                }
            });
        }

        

        // std::cout<<"FINISHED"<<std::endl;
        std::cout << "Time elapsed: " << runtime << " seconds\n";
        std::cout<<std::endl;
    } // end loop over test
}


void test_SERIAL_invert(int num_tests, bool print_matrices, bool verify, Timer t, int start){
    // Timing variable
    double runtime = 0.0;

    std::cout << "**** Testing LU Inverse without MATAR **** " << std::endl;
    for(int i = start; i < num_tests; i++){

        int matrix_size = pow(2,i);
        std::cout << "Matrix inversion size  "<<matrix_size<<"X"<<matrix_size << std::endl;

        // Create and initialize matrix
        std::vector<std::vector<double>> MAT(matrix_size, std::vector<double>(matrix_size));
        std::vector<std::vector<double>> MAT_INV(matrix_size, std::vector<double>(matrix_size));

        for(int i = 0; i < matrix_size; i++){
            for(int j = 0; j < matrix_size; j++){
                if(i > j){
                MAT[i][j] = (double)i - (double)j;
                MAT_INV[i][j] = (double)i - (double)j;
                }
                if(i < j){
                    MAT[i][j] = (double)i + (double)j + 1.0;
                    MAT_INV[i][j] = (double)i + (double)j + 1.0;
                }

                if(i == j){
                    MAT[i][j] = (double)i + (double)j + 1.0;
                    MAT_INV[i][j] = (double)i + (double)j + 1.0;
                }
            }
        }


        // Print matrix
        if(print_matrices){
            std::cout<<"Printing MAT"<<std::endl;
            print_mat_vec(MAT);
        }

        // Reset timer
        t.reset();

        // Invert MAT, store in MAT_INV
        invert(MAT, MAT_INV);

        // Save runtime for LU inverse
        runtime = t.elapsed();

        if(print_matrices){
            std::cout<<"Printing MAT_INV"<<std::endl;
            print_mat_vec(MAT_INV);
        }

        if(print_matrices){
            std::cout<<"Printing MAT"<<std::endl;
            print_mat_vec(MAT);
        }

        if(verify){

            std::vector<std::vector<double>> MAT_TEST(matrix_size, std::vector<double>(matrix_size));


            for(int i=0; i<matrix_size; i++){
                for(int j=0; j<matrix_size; j++){
                    MAT_TEST[i][j] = 0.0;
                }
            }
            // Multiply matrices    
            // rm_dense_matmul_device(MAT_TEST, MAT, MAT_INV);

            for(int i = 0; i < matrix_size; i++){
                for(int j = 0; j < matrix_size; j++){
                    for(int k = 0; k < matrix_size; k++){
                        MAT_TEST[i][k] += MAT[i][j]*MAT_INV[j][k];
                    }
                }
            }

            if(print_matrices){
                std::cout<<"Printing MAT_TEST: MAT*MAT_INV"<<std::endl;
                print_mat_vec(MAT_TEST);
            }

            int n = matrix_size;
            for(int i=0; i<n; i++){
                for(int j=0; j<n; j++){
                    
                    if(i == j){
                        if( fabs(MAT_TEST[i][j] - 1.0) >= 1E-8) std::cout<<"WRONG IDENTITY"<<std::endl;
                    }
                    else{
                        if(fabs(MAT_TEST[i][j]) >= 1E-8) std::cout<<"WRONG OFF DIAGONAL = "<<MAT_TEST[i][j]<<std::endl;
                    }
                    

                }
            }
        }

        

        // std::cout<<"FINISHED"<<std::endl;
        std::cout << "Time elapsed: " << runtime << " seconds\n";
        std::cout<<std::endl;
    } // end loop over test
}

int main(int argc, char* argv[])
{

    Timer t;

    Kokkos::initialize();
    { // Kokkos scope

    // Number of test to run, increasing the matrix size by 2^{n}
    int num_tests = 9;

    int start = 4;

    // Print out input and inverse matrices
    bool print_matrices = false;

    // Verify that matrix A\dot A^{-1} = I
    bool verify = true;

   
    test_MATAR_invert(num_tests, print_matrices, verify, t, start);
    test_SERIAL_invert(num_tests, print_matrices, verify, t, start);


    } // end Kokkos scope
    Kokkos::finalize();

    return 0;
}
