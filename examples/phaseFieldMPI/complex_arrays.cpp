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
#include "complex_arrays.h"
#include "mpi.h"

ComplexArrays::ComplexArrays(const SimParameters& sp, const std::array<int, 3>& loc_nn_img, const std::array<int, 3>& loc_start_index) :
    comp_img(loc_nn_img[2], loc_nn_img[1], loc_nn_img[0], 2),
    dfdc_img(loc_nn_img[2], loc_nn_img[1], loc_nn_img[0], 2),
    kpow2(loc_nn_img[2], loc_nn_img[1], loc_nn_img[0]),
    denominator(loc_nn_img[2], loc_nn_img[1], loc_nn_img[0]),
    fs(sp.nn, loc_nn_img, loc_start_index, sp.delta)
{
    // set values of kpow2
    set_kpow2();

    // set values of denominator
    set_denominator(sp);
}

void ComplexArrays::set_kpow2()
{
    // calculate kpow2
    FOR_ALL_CLASS(k, 0, kpow2.dims(0),
                  j, 0, kpow2.dims(1),
                  i, 0, kpow2.dims(2), {
        kpow2(k, j, i) =   fs.kx(i) * fs.kx(i)
                         + fs.ky(j) * fs.ky(j)
                         + fs.kz(k) * fs.kz(k);
    });
}

void ComplexArrays::set_denominator(const SimParameters& sp)
{
    double dt    = sp.dt;
    double M     = sp.M;
    double kappa = sp.kappa;

    // calculate denominator_
    FOR_ALL_CLASS(k, 0, denominator.dims(0),
                  j, 0, denominator.dims(1),
                  i, 0, denominator.dims(2), {
        denominator(k, j, i) = 1.0 + (dt * M * kappa * kpow2(k, j, i) * kpow2(k, j, i));
    });
}
