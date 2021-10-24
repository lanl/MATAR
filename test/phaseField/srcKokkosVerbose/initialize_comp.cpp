#include <iostream>
#include "initialize_comp.h"


void initialize_comp(SimParameters& sp, CArrayKokkos<double> &comp)
{
    // unpack simimulation parameters needed 
    // for calculations in this function
    int nx       = sp.nn[0];
    int ny       = sp.nn[1];
    int nz       = sp.nn[2];
    int iseed    = sp.iseed;
    double c0    = sp.c0;
    double noise = sp.noise;

    // seed random number generator
    srand(iseed);

    // to hold random number
    double r;

    // temp "comp" array on cpu
    double* temp_comp = new double[nx*ny*nz];

    for (int i = 0; i < nx*ny*nz; ++i) {
        // random number between 0.0 and 1.0
        r = (double) rand()/RAND_MAX;

        // initialize "comp" with stochastic thermal fluctuations
        temp_comp[i] = c0 + (2.0*r - 1.0)*noise;
    }


    // make a dual view (this is used to send temp_comp to the GPU if need be)
    auto temp_comp_dual_view = DViewCArrayKokkos<double>(&temp_comp[0], nx, ny, nz);

    // write "temp_comp_dual_view" to "comp"
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {nx,ny,nz}),
        KOKKOS_LAMBDA(const int i, const int j, const int k){
                comp(i,j,k) = temp_comp_dual_view(i,j,k);
    });


    // deallocate temp_comp
    delete [] temp_comp;
}
