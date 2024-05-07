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

#include "fourier_space.h"
#include "mpi.h"

FourierSpace::FourierSpace(const std::array<int, 3>&    glob_nn_real,
                           const std::array<int, 3>&    loc_nn_cmplx,
                           const std::array<int, 3>&    loc_start_index,
                           const std::array<double, 3>& delta) :
    kx(loc_nn_cmplx[0]),
    ky(loc_nn_cmplx[1]),
    kz(loc_nn_cmplx[2])
{
    // set values of kx, ky, and kz
    set_kx_ky_kz(glob_nn_real, loc_nn_cmplx, loc_start_index, delta);
}

void FourierSpace::set_kx_ky_kz(const std::array<int, 3>& glob_nn_real,
    const std::array<int, 3>&    loc_nn_cmplx,
    const std::array<int, 3>&    loc_start_index,
    const std::array<double, 3>& delta)
{
    int nx = glob_nn_real[0];
    int ny = glob_nn_real[1];
    int nz = glob_nn_real[2];
    int xstart = loc_start_index[0];
    int ystart = loc_start_index[1];
    int zstart = loc_start_index[2];
    double dx = delta[0];
    double dy = delta[1];
    double dz = delta[2];

    // calculate kx
    FOR_ALL_CLASS(i, 0, kx.dims(0), {
        int ti;
        ti = i + xstart;
        if (ti > nx / 2) {
            ti = ti - nx;
        }
        kx(i) = (double(ti) * twopi) / (double(nx) * dx);
    });

    // calculate ky
    FOR_ALL_CLASS(j, 0, ky.dims(0), {
        int tj;
        tj = j + ystart;
        if (tj > ny / 2) {
            tj = tj - ny;
        }
        ky(j) = (double(tj) * twopi) / (double(ny) * dy);
    });

    // calculate kz
    FOR_ALL_CLASS(k, 0, kz.dims(0), {
        int tk;
        tk = k + zstart;
        if (tk > nz / 2) {
            tk = tk - nz;
        }
        kz(k) = (double(tk) * twopi) / (double(nz) * dz);
    });
}
