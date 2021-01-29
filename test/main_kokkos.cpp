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

    }   
    Kokkos::finalize();

    return 0;
}
