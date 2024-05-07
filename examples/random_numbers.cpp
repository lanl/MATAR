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
#include "matar.h"
#include <Kokkos_Random.hpp> // for Kokkos random number generator

#define SEED  5374857

// Kokkos provides two random number generator pools one for 64bit states and one for 1024 bit states.
// Choose one.
// using gen_t = Kokkos::Random_XorShift64_Pool<DefaultExecSpace>;
using gen_t = Kokkos::Random_XorShift1024_Pool<DefaultExecSpace>;

int main()
{
    Kokkos::initialize();
    { // kokkos scope
        // Seed random number generator
        gen_t rand_pool(SEED);

        // DCArrayKokkos type to store the random numbers generated on the device
        // and print out on the host
        const int N = 100;
        DCArrayKokkos<int> arr(N);

        // Generate random numbers
        FOR_ALL(i, 0, N, {
            // Get a random number state from the pool for the active thread
            gen_t::generator_type rand_gen = rand_pool.get_state();

            // rand_gen.rand() generates integers from (0,MAX_RAND]
            // rand_gen.rand(END) generates integers from (0,END]
            // rand_gen.rand(START, END) generates integers from (START,END]
            // Note, frand() or drand() can be used in place of rand() to generate floats and
            // doubles, respectively. Please check out Kokkos_Random.hpp for all the other type of
            // scalars that are supported.

            // generate random numbers in the range (0,10]
            arr(i) = rand_gen.rand(10);

            // Give the state back, which will allow another thread to acquire it
            rand_pool.free_state(rand_gen);
        }); // end FOR_ALL

        // update host
        arr.update_host();

        for (int i = 0; i < N; i++) {
            printf(" %d", arr.host(i));
        }
        printf("\n");
    } // end kokkos scope
    Kokkos::finalize();

    return 0;
}
