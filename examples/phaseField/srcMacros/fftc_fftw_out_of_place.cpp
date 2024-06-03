/**********************************************************************************************
 � 2020. Triad National Security, LLC. All rights reserved.
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
#ifdef HAVE_OPENMP

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <complex.h>

#include <fftw3.h>
// #ifdef FFTW_OMP
#include <omp.h>
// #endif

// ----------------------------------------------------------------------
// FFTW

static void fft_forward_fftw(double* input, double* output, int nn[3])
{
    static fftw_plan plan;
    if (!plan) {
        plan = fftw_plan_dft_r2c_3d(nn[0], nn[1], nn[2],
            (double*) input, (fftw_complex*) output,
                            FFTW_ESTIMATE);
    }

    fftw_execute_dft_r2c(plan, (double*) input, (fftw_complex*) output);
}

static void fft_backward_fftw(double* input, double* output, int nn[3])
{
    static fftw_plan plan;
    if (!plan) {
        plan = fftw_plan_dft_c2r_3d(nn[0], nn[1], nn[2],
            (fftw_complex*) input, (double*) output,
                            FFTW_ESTIMATE);
    }

    fftw_execute_dft_c2r(plan, (fftw_complex*) input, (double*) output);
}

void fftc_fftw_out_of_place_(double input[], double output[], int nn[], int* ndim, int* isign)
{
    if (*isign == -1) {
        fft_forward_fftw(input, output, nn);
    }
    else {
        fft_backward_fftw(input, output, nn);
    }
}

void fftc_fftw_init_out_of_place_(void)
{
// #ifdef FFTW_OMP
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());
// #endif
}

#endif
#endif
