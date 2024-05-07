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
#include <iostream>
#include <stdio.h>
#include <chrono>

#include "sim_parameters.h"
#include "global_arrays.h"
#include "initialize_comp.h"
#include "CH_fourier_spectral_solver.h"
#include "local_free_energy.h"
#include "outputs.h"

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {
        // simulation parameters
        SimParameters sp;
        sp.print();

        // global arrays needed for simulation
        GlobalArrays ga = GlobalArrays(sp.nn);

        // setup initial composition profile
        initialize_comp(sp, ga.comp);

        // initialize solver
        CHFourierSpectralSolver CH_fss(sp);

        // Start measuring time
        auto begin = std::chrono::high_resolution_clock::now();

        // time stepping loop
        for (int iter = 1; iter <= sp.num_steps; iter++) {
            // calculate df/dc
            calculate_dfdc(sp.nn, ga.comp, ga.dfdc);

            // Cahn Hilliard equation solver
            CH_fss.time_march(ga.comp, ga.dfdc);

            // report simulation progress and output vtk files
            if (iter % sp.print_rate == 0) {
                track_progress(iter, sp.nn, ga.comp);

                write_vtk(iter, sp.nn, sp.delta, ga.comp);

                output_total_free_energy(iter, sp.print_rate, sp.num_steps,
                                     sp.nn, sp.delta, sp.kappa,
                                     ga.comp);
            }
        }

        // Stop measuring time and calculate the elapsed time
        auto end     = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        printf("Total time was %f seconds.\n", elapsed.count() * 1e-9);
    }
    Kokkos::finalize();

    return 0;
}
