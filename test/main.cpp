#include <stdio.h>

#include "matar.h"

int main() {

    printf("Hello World\n");

    Kokkos::initialize();
    {   
    auto test = CArrayKokkos <int> (5, 5); 
    }   
    Kokkos::finalize();

    return 0;
}
