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

FourierSpace::FourierSpace(int* nn, double* delta)
{
    // initialize class data
    nn_    = nn;
    nx_    = nn[0];
    ny_    = nn[1];
    nz_    = nn[2];
    delta_ = delta;
    dx_    = delta[0];
    dy_    = delta[1];
    dz_    = delta[2];

    // kx_, ky_, and kz_ initialization
    kx_ = CArrayKokkos<double>(nx_);
    ky_ = CArrayKokkos<double>(ny_);
#ifdef IN_PLACE_FFT
    kz_ = CArrayKokkos<double>(nz_);
#elif OUT_OF_PLACE_FFT
    nz21_ = nz_ / 2 + 1;
    kz_   = CArrayKokkos<double>(nz21_);
#endif

    // set values of kx_, ky_, and kz_
    set_kx_ky_kz_();
}

void FourierSpace::set_kx_ky_kz_()
{
    // calculate kx_
    FOR_ALL_CLASS(i, 0, nx_, {
        int ti;
        ti = i;
        if (ti > nx_ / 2) {
            ti = ti - nx_;
        }
        kx_(i) = (float(ti) * twopi_) / (nx_ * dx_);
    });

    // calculate ky_
    FOR_ALL_CLASS(j, 0, ny_, {
        int tj;
        tj = j;
        if (tj > ny_ / 2) {
            tj = tj - ny_;
        }
        ky_(j) = (float(tj) * twopi_) / (ny_ * dy_);
    });

    // calculate kz_ for in-place-fft
#ifdef IN_PLACE_FFT
    FOR_ALL_CLASS(k, 0, nz_, {
        int tk;
        tk = k;
        if (tk > nz_ / 2) {
            tk = tk - nz_;
        }
        kz_(k) = (float(tk) * twopi_) / (nz_ * dz_);
    });
#elif OUT_OF_PLACE_FFT
    FOR_ALL_CLASS(k, 0, nz21_, {
        int tk;
        tk     = k;
        kz_(k) = (float(tk) * twopi_) / (nz_ * dz_);
    });
#endif
}

CArrayKokkos<double>& FourierSpace::get_kx()
{
    return kx_;
}

CArrayKokkos<double>& FourierSpace::get_ky()
{
    return ky_;
}

CArrayKokkos<double>& FourierSpace::get_kz()
{
    return kz_;
}
