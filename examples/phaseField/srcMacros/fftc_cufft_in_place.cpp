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
#ifdef HAVE_CUDA

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <complex.h>

#include <cufft.h>

// ----------------------------------------------------------------------
// CUFFT

static void fft_cufft_forward(double* data, int nn[3])
{
    int stride = 2 * nn[0] * nn[1] * nn[2];
    int rc, i;

    static cufftHandle planZ2Z;
//  typedef cuComplex cufftComplex;
    typedef cuDoubleComplex cufftDoubleComplex;
    if (!planZ2Z) {
        cufftPlan3d(&planZ2Z, nn[0], nn[1], nn[2], CUFFT_Z2Z);
    }

// #pragma acc data copy(data[0:batch*stride])
    {
//      printf("data1 %p\n", data);
// #pragma acc host_data use_device(data)
        {
//      printf("data2 %p\n", data);
            rc = cufftExecZ2Z(planZ2Z, (cufftDoubleComplex*) data,
                (cufftDoubleComplex*) data,
                        CUFFT_FORWARD);
            assert(rc == CUFFT_SUCCESS);
        }
    }
}

static void fft_cufft_backward(double* data, int nn[3])
{
    int stride = 2 * nn[0] * nn[1] * nn[2];
    int rc, i;

    static cufftHandle planZ2Z;
//  typedef cuComplex cufftComplex;
    typedef cuDoubleComplex cufftDoubleComplex;
    if (!planZ2Z) {
        cufftPlan3d(&planZ2Z, nn[0], nn[1], nn[2], CUFFT_Z2Z);
    }

// #pragma acc data copy(data[0:batch*stride])
    {
// #pragma acc host_data use_device(data)
        rc = cufftExecZ2Z(planZ2Z, (cufftDoubleComplex*) data,
            (cufftDoubleComplex*) data,
                      CUFFT_INVERSE);
        assert(rc == CUFFT_SUCCESS);
    }
}

// ----------------------------------------------------------------------

void fftc_cufft_in_place_(double data[], int nn[], int* ndim, int* isign)
{
    // assert(*ndim == 3);
    if (*isign == -1) {
        fft_cufft_forward(data, nn);
    }
    else {
        fft_cufft_backward(data, nn);
    }
}

void fftc_cufft_init_in_place_(void)
{
}

#endif
#endif
