#include <stdlib.h>
#include "matar.h"


int main() {

    int si = 5, sj = 3;

    RaggedRightArray <int> rtest;
    rtest = RaggedRightArray <int> (5, 3);

    int count = 0;
    for (int i = 0; i < si; i++) {
        for (int j = 0; j < sj; j++) {
            if ((i+j) % 2 == 0) {
                //rtest.push_back(i);
                size_t sizei = i;
                rtest += sizei;
                rtest(i, j) = 1;
                count++;
                printf("Push Back %d %d on to position %d\n", i, j, count);
            }
        }
    }

    count = 0;

#ifdef HAVE_KOKKOS
    Kokkos::initialize();
    {




    RaggedRightArrayKokkos <int> rktest;
    rktest = RaggedRightArrayKokkos <int> (si, sj);
    Kokkos::parallel_for("RaggedTest", 5, KOKKOS_LAMBDA (const int i) {
        int mycount = 0;
        for (int j = 0; j < sj; j++) {
            if ((i+j) % 2 == 0) {
                mycount++;
            } 
        }
        rktest.build_stride(i) = mycount;
    });
    rktest.stride_finalize();

    Kokkos::parallel_for("RaggedTest", si, KOKKOS_LAMBDA (const int i) {
        printf("%ld\n", rktest.stride(i));
    });
    

    }
    Kokkos::finalize();
#endif
  

    return 0;
}
