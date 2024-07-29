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
#include "local_free_energy.h"

double calculate_total_free_energy(int* nn, double* delta, double kappa, DCArrayKokkos<double>& comp)
{
    // this function calculates the total free energy of the system.

    // unpack simimulation parameters needed
    // for calculations in this function
    int    nx = nn[0];
    int    ny = nn[1];
    int    nz = nn[2];
    double dx = delta[0];
    double dy = delta[1];
    double dz = delta[2];

    //
    double total_energy = 0.0;
    double loc_sum = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 1, 1, 1 }, { nx - 1, ny - 1, nz - 1 }),
        KOKKOS_LAMBDA(const int i, const int j, const int k, double& loc_sum) {
        // central difference spatial derivative of comp
           double dcdx = (comp(i + 1, j, k) - comp(i - 1, j, k)) / (2.0 * dx);
           double dcdy = (comp(i, j + 1, k) - comp(i, j - 1, k)) / (2.0 * dy);
           double dcdz = (comp(i, j, k + 1) - comp(i, j, k - 1)) / (2.0 * dz);
           loc_sum    +=   comp(i, j, k) * comp(i, j, k) * (1.0 - comp(i, j, k)) * (1.0 - comp(i, j, k))
                         + 0.5 * kappa * (dcdx * dcdx + dcdy * dcdy + dcdz * dcdz);
        }, total_energy);

    return total_energy;
}

void calculate_dfdc(int* nn, DCArrayKokkos<double>& comp, CArrayKokkos<double>& dfdc)
{
    // this function calculates the derivitive of local free energy density (f)
    // with respect to composition (c) (df/dc).

    // unpack simimulation parameters needed
    // for calculations in this function
    int nx = nn[0];
    int ny = nn[1];
    int nz = nn[2];

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 0, 0, 0 }, { nx, ny, nz }),
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
            dfdc(i, j, k) =   4.0 * comp(i, j, k) * comp(i, j, k) * comp(i, j, k)
                            - 6.0 * comp(i, j, k) * comp(i, j, k)
                            + 2.0 * comp(i, j, k);
    });
}
