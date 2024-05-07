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
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <matar.h>

using namespace mtr; // matar namespace

const int    width  = 1000;
const int    height = 1000;
const double temp_tolerance = 0.01;

void initialize(FArrayKokkos<double>& temperature_previous);
void track_progress(int iteration, FArrayKokkos<double>& temperature);

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {
        auto temperature = FArrayKokkos<double>(height + 2, width + 2);
        auto temperature_previous = FArrayKokkos<double>(height + 2, width + 2);

        int    iteration = 1;
        double worst_dt  = 100;
        double max_value;

        // Start measuring time
        auto begin = std::chrono::high_resolution_clock::now();

        // initialize temperature profile
        initialize(temperature_previous);

        while (worst_dt > temp_tolerance) {
            // finite difference
            Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>({ 1, 1 }, { height + 1, width + 1 }),
            KOKKOS_LAMBDA(const int i, const int j) {
                temperature(i, j) = 0.25 * (temperature_previous(i + 1, j)
                                            + temperature_previous(i - 1, j)
                                            + temperature_previous(i, j + 1)
                                            + temperature_previous(i, j - 1));
            });

            // calculate max difference between temperature and temperature_previous
            Kokkos::parallel_reduce(
            Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>({ 1, 1 }, { height + 1, width + 1 }),
            KOKKOS_LAMBDA(const int i, const int j, double& loc_max_value) {
                double value = fabs(temperature(i, j) - temperature_previous(i, j));
                if (value > loc_max_value) {
                    loc_max_value = value;
                }
            }, Kokkos::Max<double>(max_value));
            worst_dt = max_value;

            // update temperature_previous
            Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>({ 1, 1 }, { height + 1, width + 1 }),
            KOKKOS_LAMBDA(const int i, const int j) {
                temperature_previous(i, j) = temperature(i, j);
            });

            // track progress
            if (iteration % 100 == 0) {
                track_progress(iteration, temperature);
            }

            iteration++;
        }

        // Stop measuring time and calculate the elapsed time
        auto end     = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

        printf("Total time was %f seconds.\n", elapsed.count() * 1e-9);
        printf("\nMax error at iteration %d was %f\n", iteration - 1, worst_dt);
    }
    Kokkos::finalize();

    return 0;
}

void initialize(FArrayKokkos<double>& temperature_previous)
{
    // initialize temperature_previous to 0.0
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>({ 0, 0 }, { height + 2, width + 2 }),
        KOKKOS_LAMBDA(const int i, const int j) {
            temperature_previous(i, j) = 0.0;
    });

    // setting the left and right boundary conditions
    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(0, height + 2),
        KOKKOS_LAMBDA(const int i) {
            temperature_previous(i, 0) = 0.0;
            temperature_previous(i, width + 1) = (100.0 / height) * i;
    });

    // setting the top and bottom boundary condition
    Kokkos::parallel_for(
        Kokkos::RangePolicy<>(0, width + 2),
        KOKKOS_LAMBDA(const int j) {
            temperature_previous(0, j) = 0.0;
            temperature_previous(height + 1, j) = (100.0 / width) * j;
    });
}

void track_progress(int iteration, FArrayKokkos<double>& temperature)
{
    int i;

    // make a deep copy of temperature from device to host
    auto temperature_host = create_mirror_view_and_copy(Kokkos::HostSpace(), temperature.get_kokkos_view());
    auto temperature_host_view = ViewFArray<double>(temperature_host.data(), height + 2, width + 2);

    printf("---------- Iteration number: %d ----------\n", iteration);
    for (i = height - 5; i <= height; i++) {
        printf("[%d,%d]: %5.2f  ", i, i, temperature_host_view(i, i));
    }
    printf("\n");
}
