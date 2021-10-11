#include <stdio.h>
#include <array>

#include "matar.h"

int main() {

    Kokkos::initialize();
    {   

    // -----------------------
    // parameters for examples
    // -----------------------
    u_int size_i, size_j, size_k, size_l; 
    size_i = 3; size_j = 4; size_k = 5; size_l = 6;
    policy1D standard_policy_1d = policy1D({0}, {size_i});
    policy2D standard_policy_2d = policy2D({0, 0}, {size_i, size_j});
    policy3D standard_policy_3d = policy3D({0, 0, 0}, {size_i, size_j, size_k});
    policy4D standard_policy_4d = policy4D({0, 0, 0, 0}, {size_i, size_j, size_k, size_l});

    // -----------------------
    // CArray
    // -----------------------

    printf("\n1D CArray\n");
    auto cak1D = CArrayKokkos <int> (size_i);
    Kokkos::parallel_for("1DCArray", standard_policy_1d, KOKKOS_LAMBDA(const int i) {
            cak1D(i) = i;
            //printf("%d) %d\n", i, cak1D(i));
        });
    Kokkos::fence();

    // -----------------------
    // FArray
    // -----------------------

    printf("\n2D FArray\n");
    auto fak2D = FArrayKokkos <int> (size_i, size_j);
    Kokkos::parallel_for("2DFArray", standard_policy_2d, KOKKOS_LAMBDA(const int i, const int j) {
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
    Kokkos::parallel_for("3DCMatrix", standard_policy_3d, KOKKOS_LAMBDA(const int i, const int j, const int k) {
            cmk3D(i, j, k) = i * size_j * size_k + j * size_k + k;
            //printf("%d) %d\n", i, cmk3D(i, j, k));
        });
    Kokkos::fence();

    // -----------------------
    // FMatrix
    // -----------------------

    printf("\n4D FMatrix\n");
    auto fmk4D = FMatrixKokkos <int> (size_i, size_j, size_k, size_l);
    Kokkos::parallel_for("4DFMatrix", standard_policy_4d, KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
            int idx = l * size_i * size_j * size_k + k * size_i * size_j + j * size_i + i;
            fmk4D(i, j, k, l) = idx;
            //printf("%d) %d\n", idx, fmk4D(i, j, k, l));
        });
    Kokkos::fence();

    // -----------------------
    // RaggedRightArray
    // -----------------------

    printf("\nRagged Right Array\n");
    //auto 

/*
    Kokkos::parallel_for("DRRAKTest", size_i, KOKKOS_LAMBDA(const int i) {
            for (int j = 0; j < (i % size_j) + 1; j++) {
                drrak.stride(i)++;
                printf("(%i) stride to %d\n", i, j+1);
            } 
    });
  
    std::array<int, 9> A1d;
    for (int init = 0; init < 9; init++) { A1d[init] = init+1; }
    using policy2D = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
    policy2D CAKPpol = policy2D({0,0}, {3, 3});
    DViewCArrayKokkos <int> cakp;
    cakp = DViewCArrayKokkos <int> (&A1d[0], 3, 3);
    Kokkos::parallel_for("CAKPTest", CAKPpol, KOKKOS_LAMBDA(const int i, const int j) {
        //cakp(i, j) = i * 3 + j;
        printf("%d) %d\n", i * 3 + j, cakp(i, j));
    });
*/  

    }   
    Kokkos::finalize();

    return 0;
}
