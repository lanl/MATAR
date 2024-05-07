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
#ifdef IN_PLACE_FFT

# include "fft_manager_in_place.h"

FFTManagerInPlace::FFTManagerInPlace(int* nn)
{
    nn_   = nn;
    nx_   = nn_[0];
    ny_   = nn_[1];
    nz_   = nn_[2];
    data_ = CArrayKokkos<double>(nx_, ny_, nz_, 2);

    // calculate ndim
    ndim_ = 0;
    for (int i = 0; i < 3; i++) {
        if (nn_[i] > 1) {
            ++ndim_;
        }
    }

    // initialize fft
    #ifdef HAVE_CUDA
    fftc_cufft_init_in_place_();
    #else
    fftc_fftw_init_in_place_();
    #endif
}

void FFTManagerInPlace::prep_for_forward_fft_(double* input)
{
    // this function writes the data in "input" array to "data_" array
    // in order to ready "data_" for in-place forward fft.

    // create view of input
    auto input_view = ViewCArrayKokkos<double>(input, nx_, ny_, nz_);

    // write input to data for in-place forward fft
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 0, 0, 0 }, { nx_, ny_, nz_ }),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            data_(i, j, k, 0) = input_view(i, j, k);
            data_(i, j, k, 1) = 0.0;
    });
}

void FFTManagerInPlace::get_forward_fft_result_(double* output)
{
    // this function writes the result of in-place forward fft
    // in "data_" array into "output" array.

    // create view of output
    auto output_view = ViewCArrayKokkos<double>(output, nx_, ny_, nz_, 2);

    // write data to output after in-place fft
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 0, 0, 0 }, { nx_, ny_, nz_ }),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            output_view(i, j, k, 0) = data_(i, j, k, 0);
            output_view(i, j, k, 1) = data_(i, j, k, 1);
    });
}

void FFTManagerInPlace::prep_for_backward_fft_(double* input)
{
    // this function writes the data in "input" array to "data_" array
    // in order to ready "data_" for in-place backward fft.

    // create view of input
    auto input_view = ViewCArrayKokkos<double>(input, nx_, ny_, nz_, 2);

    // write input to data for in-place fft
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 0, 0, 0 }, { nx_, ny_, nz_ }),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            data_(i, j, k, 0) = input_view(i, j, k, 0);
            data_(i, j, k, 1) = input_view(i, j, k, 1);
    });
}

void FFTManagerInPlace::get_backward_fft_result_(double* output)
{
    // this function writes the result of in-place backward fft
    // in "data_" array into "output" array.

    // create view of output
    auto output_view = ViewCArrayKokkos<double>(output, nx_, ny_, nz_);

    // write data to output after in-place fft
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ 0, 0, 0 }, { nx_, ny_, nz_ }),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            output_view(i, j, k) = data_(i, j, k, 0);
    });
}

void FFTManagerInPlace::perform_forward_fft(double* input, double* output)
{
    // this function performs forward fft on "input" array and
    // writes the result to "output" array.
    // it calls the appropriate function to perform the forward in-place fft
    // either using OPENMP or CUDA.

    // prep for forward fft
    prep_for_forward_fft_(input);

    // perform foward fft
    isign_ = -1;
    #ifdef HAVE_CUDA
    fftc_cufft_in_place_(data_.pointer(), nn_, &ndim_, &isign_);
    #else
    fftc_fftw_in_place_(data_.pointer(), nn_, &ndim_, &isign_);
    #endif

    // get result after performing foward fft
    get_forward_fft_result_(output);
}

void FFTManagerInPlace::perform_backward_fft(double* input, double* output)
{
    // this function performs backward fft on "input" array and
    // writes the result to "output" array.
    // it calls the appropriate function to perform the backward in-place fft
    // either using OPENMP or CUDA.

    // prep for backward fft
    prep_for_backward_fft_(input);

    // perform backward fft
    isign_ = 1;
    #ifdef HAVE_CUDA
    fftc_cufft_in_place_(data_.pointer(), nn_, &ndim_, &isign_);
    #else
    fftc_fftw_in_place_(data_.pointer(), nn_, &ndim_, &isign_);
    #endif

    // get result after performing backward fft
    get_backward_fft_result_(output);
}

#endif
