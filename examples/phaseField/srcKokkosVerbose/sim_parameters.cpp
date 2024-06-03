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
#include <fstream>
#include <limits>

#include "sim_parameters.h"

SimParameters::SimParameters()
{
    // set default simulation parameters
    this->nn[0]    = 32;          // nx
    this->nn[1]    = 32;          // ny
    this->nn[2]    = 32;          // nz
    this->delta[0] = 1.0;         // dx
    this->delta[1] = 1.0;         // dy
    this->delta[2] = 1.0;         // dz
    this->dt = 5.0E-2;            // dt
    this->num_steps  = 1000;      // total number of time steps
    this->print_rate = 100;       // time step interval for output file
    this->iseed = 456;            // random number seed
    this->kappa = 1.0;            // gradient energy coefficient
    this->M     = 1.0;            // mobility
    this->c0    = 5.0E-1;         // critical composition
    this->noise = 5.0E-3;         // noise term for thermal fluctuations

    // set number of dimensions
    set_ndim();
}

void SimParameters::set_ndim()
{
    ndim = 0;
    for (int i = 0; i < 3; i++) {
        if (nn[i] > 1) {
            ++ndim;
        }
    }
}

void SimParameters::print()
{
    std::cout << " nx = " << nn[0] << std::endl;
    std::cout << " ny = " << nn[1] << std::endl;
    std::cout << " nz = " << nn[2] << std::endl;
    std::cout << " dx = " << delta[0] << std::endl;
    std::cout << " dy = " << delta[1] << std::endl;
    std::cout << " dz = " << delta[2] << std::endl;
    std::cout << " dt = " << dt << std::endl;
    std::cout << " num_steps = " << num_steps << std::endl;
    std::cout << " print_rate = " << print_rate << std::endl;
    std::cout << " iseed = " << iseed << std::endl;
    std::cout << " kappa = " << kappa << std::endl;
    std::cout << " M = " << M << std::endl;
    std::cout << " c0 = " << c0 << std::endl;
    std::cout << " noise = " << noise << std::endl;
}
