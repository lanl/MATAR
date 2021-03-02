#include <stdio.h>

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
                printf("(%i) stride to %d\n", i, j+1);
            } 
    });

    
    std::array<int, 9> A1d;
    CArrayKokkosPtr <int> cakp;
    cakp = CArrayKokkosPtr <int> (&A1d[0], 9);
    Kokkos::parallel_for("CAKPTest", 9, KOKKOS_LAMBDA(const int i) {
        cakp(i) = i;
    });
    }   
    Kokkos::finalize();

    return 0;
}
