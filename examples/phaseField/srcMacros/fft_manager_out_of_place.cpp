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
#ifdef OUT_OF_PLACE_FFT

#include "fft_manager_out_of_place.h"

FFTManagerOutOfPlace::FFTManagerOutOfPlace(int* nn)
{
    nn_   = nn;
    nx_   = nn_[0];
    ny_   = nn_[1];
    nz_   = nn_[2];
    nz21_ = nz_ / 2 + 1;

    // initialize fft
    #ifdef HAVE_CUDA
    fftc_cufft_init_out_of_place_();
    #else
    fftc_fftw_init_out_of_place_();
    #endif
}

void FFTManagerOutOfPlace::perform_forward_fft(double* input, double* output)
{
    // this function performs forward fft on "input" array and
    // writes the result to "output" array.
    // it calls the appropriate function to perform the forward out-of-place fft
    // either using OPENMP or CUDA.

    // perform foward fft
    isign_ = -1;
    #ifdef HAVE_CUDA
    fftc_cufft_out_of_place_(input, output, nn_, &ndim_, &isign_);
    #else
    fftc_fftw_out_of_place_(input, output, nn_, &ndim_, &isign_);
    #endif
}

void FFTManagerOutOfPlace::perform_backward_fft(double* input, double* output)
{
    // this function performs backward fft on "input" array and
    // writes the result to "output" array.
    // it calls the appropriate function to perform the backward out-of-place fft
    // either using OPENMP or CUDA.

    // perform backward fft
    isign_ = 1;
    #ifdef HAVE_CUDA
    fftc_cufft_out_of_place_(input, output, nn_, &ndim_, &isign_);
    #else
    fftc_fftw_out_of_place_(input, output, nn_, &ndim_, &isign_);
    #endif
}

#endif
