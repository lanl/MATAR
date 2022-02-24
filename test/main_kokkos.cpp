#include <stdio.h>
#include <array>

#include "matar.h"


// A notional class
class Data {
private:
    u_int nx_;
    u_int ny_;
    u_int nz_;
    
    CArrayKokkos <int> arr3D_;
    
public:
    
    // default constructor
    Data();
    
    // overload constructor to set dimensions
    Data(u_int nx, u_int ny, u_int nz);
    
    void some_fcn();
    
}; // end class Data

Data::Data(){};

Data::Data(u_int nx, u_int ny, u_int nz){
    
    nx_ = nx;
    ny_ = ny;
    nz_ = nz;
    
    arr3D_ = CArrayKokkos <int> (nx_, ny_, nz_);
};


void Data::some_fcn(){
    
    // parallel loop inside a class
    // The KOKKOS_CLASS_LAMBDA is [=, *this]. The *this in the lambda
    // capture gives access to the class data
    Kokkos::parallel_for("3DCArray",
                         Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nx_, ny_, nz_}),
                         KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
                             int idx = (i-1) * nx_ * ny_ + (j-1) * nz_ + (k-1);
                             arr3D_(i, j, k) = idx;
                         });
    Kokkos::fence();
    
    // now using the macros for a parallel loop inside a class
    FOR_ALL_CLASS(i, 0, nx_,
                  j, 0, ny_,
                  k, 0, nz_, {
                      
                  int idx = (i-1) * nx_ * ny_ + (j-1) * nz_ + (k-1);
                  arr3D_(i, j, k) = idx;
                  //printf("\nloop\n");
                  });
    Kokkos::fence();
    
}; // end member function



// Main function
int main() {

    Kokkos::initialize();
    {   

        // -----------------------
        // parameters for examples
        // -----------------------
        u_int size_i, size_j, size_k, size_l;
        size_i = 3; size_j = 4; size_k = 5; size_l = 6;
        
        policy1D Arr_policy_1d = policy1D(0, size_i);
        policy2D Arr_policy_2d = policy2D({0, 0}, {size_i, size_j});
        policy3D Arr_policy_3d = policy3D({0, 0, 0}, {size_i, size_j, size_k});
        policy4D Arr_policy_4d = policy4D({0, 0, 0, 0}, {size_i, size_j, size_k, size_l});
        
        policy1D Mtx_policy_1d = policy1D(1, size_i+1);
        policy2D Mtx_policy_2d = policy2D({1, 1}, {size_i+1, size_j+1});
        policy3D Mtx_policy_3d = policy3D({1, 1, 1}, {size_i+1, size_j+1, size_k+1});
        policy4D Mtx_policy_4d = policy4D({1, 1, 1, 1}, {size_i+1, size_j+1, size_k+1, size_l+1});
        
        
        // -----------------------
        // CArray
        // -----------------------
        
        printf("\n1D CArray\n");
        auto cak1D = CArrayKokkos <int> (size_i);
        
        // a parallel 1D loop
        Kokkos::parallel_for("1DCArray", Arr_policy_1d, KOKKOS_LAMBDA(const int i) {
            cak1D(i) = i;
            //printf("%d) %d\n", i, cak1D(i));
        });
        Kokkos::fence();
        
        // the marco for a parallel 1D loop
        FOR_ALL(i, 0, size_i, {
            cak1D(i) = i;
        });
        
        Kokkos::fence();
        
        // -----------------------
        // FArray
        // -----------------------
        
        printf("\n2D FArray\n");
        auto fak2D = FArrayKokkos <int> (size_i, size_j);
        Kokkos::parallel_for("2DFArray", Arr_policy_2d, KOKKOS_LAMBDA(const int i, const int j) {
            int idx = j * size_i + i;
            fak2D(i, j) = idx;
            //printf("%d) %d\n", idx, fak2D(i, j));
        });
        Kokkos::fence();
        
        // the marco for a parallel 2D nested loop
        FOR_ALL(i, 0, size_i,
                j, 0, size_j,
                {
                int idx = j * size_i + i;
                fak2D(i, j) = idx;
                //printf("%d) %d\n", idx, fak2D(i, j));
                });
        Kokkos::fence();
        
        // -----------------------
        // CMatrix
        // -----------------------
        
        printf("\n3D CMatrix\n");
        auto cmk3D = CMatrixKokkos <int> (size_i, size_j, size_k);
        Kokkos::parallel_for("3DCMatrix", Mtx_policy_3d, KOKKOS_LAMBDA(const int i, const int j, const int k) {
            int idx = (i-1) * size_j * size_k + (j-1) * size_k + (k-1);
            cmk3D(i, j, k) = idx;
            //printf("%d) %d\n", i, cmk3D(i, j, k));
        });
        Kokkos::fence();
        
        
        // the marco for a parallel 3D nested loop
        FOR_ALL(i, 1, size_i,
                j, 1, size_j,
                k, 1, size_k,
                {
                int idx = (i-1) * size_j * size_k + (j-1) * size_k + (k-1);
                cmk3D(i, j, k) = idx;
                //printf("%d) %d\n", i, cmk3D(i, j, k));
                });
        Kokkos::fence();
        
        // -----------------------
        // FMatrix
        // -----------------------
        
        printf("\n4D FMatrix\n");
        auto fmk4D = FMatrixKokkos <int> (size_i, size_j, size_k, size_l);
        Kokkos::parallel_for("4DFMatrix", Mtx_policy_4d, KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
            int idx = (l-1) * size_i * size_j * size_k + (k-1) * size_i * size_j + (j-1) * size_i + (i-1);
            fmk4D(i, j, k, l) = idx;
            //printf("%d) %d\n", idx, fmk4D(i, j, k, l));
        });
        Kokkos::fence();
        
        // -----------------------
        // RaggedRightArray
        // -----------------------
        
        printf("\nDynamic Ragged Right Array\n");
        DynamicRaggedRightArrayKokkos <int> drrak;
        drrak = DynamicRaggedRightArrayKokkos <int> (size_i, size_j);
        
        Kokkos::parallel_for("DRRAKTest", size_i, KOKKOS_LAMBDA(const int i) {
            for (int j = 0; j < (i % size_j) + 1; j++) {
                drrak.stride(i)++;
                drrak(i,j) = j;
                //printf("(%i) stride is %d\n", i, j);
            }
        });
        Kokkos::fence();
        
        // testing MATAR FOR_ALL loop
        DynamicRaggedRightArrayKokkos <int> my_dyn_ragged(size_i, size_j);
        FOR_ALL(i, 0, size_i, {
            for (int j = 0; j <= (i % size_j); j++) {
                my_dyn_ragged.stride(i)++;
                my_dyn_ragged(i,j) = j;
            }// end for
        });// end parallel for
        Kokkos::fence();
        
        
        
        // -----------------------
        // CArray view
        // -----------------------
        
        printf("\nView CArray\n");
        std::array<int, 9> A1d;
        for (int init = 0; init < 9; init++) {
            A1d[init] = init+1;
        }
        policy2D CAKPpol = policy2D({0,0}, {3, 3});
        DViewCArrayKokkos <int> cakp;
        cakp = DViewCArrayKokkos <int> (&A1d[0], 3, 3);
        Kokkos::parallel_for("CAKPTest", CAKPpol, KOKKOS_LAMBDA(const int i, const int j) {
            //printf("%d) %d\n", i * 3 + j, cakp(i, j));
        });
        Kokkos::fence();
        
        // -----------------------
        // CArray inside a class
        // -----------------------
        
        printf("\nCArray in a class\n");
        Data my_data(size_i, size_j, size_k);
        my_data.some_fcn();
        
    } // end of kokkos scope
    
    Kokkos::finalize();
    
    printf("\nfinished\n\n");

    return 0;
}
