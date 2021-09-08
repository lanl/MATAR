#include <stdio.h>
#include <array>

#include "matar.h"

int main() {

    Kokkos::initialize();
    {   

    int size_i = 3, size_j = 5;

    DynamicRaggedRightArrayKokkos <int> drrak;
    drrak = DynamicRaggedRightArrayKokkos <int> (size_i, size_j);

    Kokkos::parallel_for("DRRAKTest", size_i, KOKKOS_LAMBDA(const int i) {
            for (int j = 0; j < (i % size_j) + 1; j++) {
                drrak.stride(i)++;
                //printf("(%i) stride to %d\n", i, j+1);
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
    }   
    Kokkos::finalize();

    return 0;
}
