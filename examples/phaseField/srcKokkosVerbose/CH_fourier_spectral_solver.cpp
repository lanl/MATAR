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
#include "CH_fourier_spectral_solver.h"
#include "fourier_space.h"
#ifdef IN_PLACE_FFT
  #include "fft_manager_in_place.h"
#elif OUT_OF_PLACE_FFT
  #include "fft_manager_out_of_place.h"
#endif

CHFourierSpectralSolver::CHFourierSpectralSolver(SimParameters& sp)
{
    // set simulation parameters
    nn_    = sp.nn;
    nx_    = nn_[0];
    ny_    = nn_[1];
    nz_    = nn_[2];
    delta_ = sp.delta;
    dx_    = delta_[0];
    dy_    = delta_[1];
    dz_    = delta_[2];
    ndim_  = sp.ndim;
    dt_    = sp.dt;
    M_     = sp.M;
    kappa_ = sp.kappa;

    // set dimensions for nn_img_
    nn_img_[0] = nx_;
    nn_img_[1] = ny_;
#ifdef IN_PLACE_FFT
    nn_img_[2] = nz_;
#elif OUT_OF_PLACE_FFT
    nz21_ = nz_ / 2 + 1;
    nn_img_[2] = nz21_;
#endif

    // initialize arrays needed for simulation
    comp_img_    = CArrayKokkos<double>(nn_img_[0], nn_img_[1], nn_img_[2], 2);
    dfdc_img_    = CArrayKokkos<double>(nn_img_[0], nn_img_[1], nn_img_[2], 2);
    kpow2_       = CArrayKokkos<double>(nn_img_[0], nn_img_[1], nn_img_[2]);
    denominator_ = CArrayKokkos<double>(nn_img_[0], nn_img_[1], nn_img_[2]);

    // set values of kpow2_
    set_kpow2_();

    // set values of denominator_
    set_denominator_();
}

void CHFourierSpectralSolver::set_kpow2_()
{
    // get fourier space
    FourierSpace fs = FourierSpace(nn_, delta_);
    auto kx = fs.get_kx();
    auto ky = fs.get_ky();
    auto kz = fs.get_kz();

    // calculate kpow2_
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 0, 0, 0 }, { nn_img_[0], nn_img_[1], nn_img_[2] }),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            kpow2_(i, j, k) =  kx(i) * kx(i)
                              + ky(j) * ky(j)
                              + kz(k) * kz(k);
    });
}

void CHFourierSpectralSolver::set_denominator_()
{
    // calculate denominator_
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 0, 0, 0 }, { nn_img_[0], nn_img_[1], nn_img_[2] }),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            denominator_(i, j, k) = 1.0 + (dt_ * M_ * kappa_ * kpow2_(i, j, k) * kpow2_(i, j, k));
    });
}

void CHFourierSpectralSolver::time_march(DCArrayKokkos<double>& comp, CArrayKokkos<double>& dfdc)
{
    // initialize fft manager
#ifdef IN_PLACE_FFT
    static FFTManagerInPlace fft_manager = FFTManagerInPlace(nn_);
#elif OUT_OF_PLACE_FFT
    static FFTManagerOutOfPlace fft_manager = FFTManagerOutOfPlace(nn_);
#endif

    // get foward fft of comp
    fft_manager.perform_forward_fft(comp.device_pointer(), comp_img_.pointer());

    // get foward fft of dfdc
    fft_manager.perform_forward_fft(dfdc.pointer(), dfdc_img_.pointer());

    // solve Cahn Hilliard equation in fourier space
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 0, 0, 0 }, { nn_img_[0], nn_img_[1], nn_img_[2] }),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            comp_img_(i, j, k, 0) =   (comp_img_(i, j, k, 0) - (dt_ * M_ * kpow2_(i, j, k)) * dfdc_img_(i, j, k, 0))
                                    / (denominator_(i, j, k));

            comp_img_(i, j, k, 1) =   (comp_img_(i, j, k, 1) - (dt_ * M_ * kpow2_(i, j, k)) * dfdc_img_(i, j, k, 1))
                                    / (denominator_(i, j, k));
    });

    // get backward fft of comp_img
    fft_manager.perform_backward_fft(comp_img_.pointer(), comp.device_pointer());

    // normalize after inverse fft
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 0, 0, 0 }, { nx_, ny_, nz_ }),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            comp(i, j, k) = comp(i, j, k) / double(nx_ * ny_ * nz_);
    });
}
