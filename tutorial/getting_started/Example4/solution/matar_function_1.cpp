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
#include <matar.h>
#include <stdio.h>
#include <math.h>

using namespace mtr; // matar namespace

/*
  Functions or subroutines that will be called from fortran should include "_" at the end of name.
  Example: Given a C++ definded subroutine as "subroutineName_",
           it should be called in fortran as "call subroutineName(...)"
  Also the functions or subroutines should be decleared with extern "C"
*/

extern "C" void matar_initialize_();
extern "C" void matar_finalize_();

extern "C" void square_array_elements_(double* array, int* nx, int* ny);
extern "C" void sum_array_elements_(double* array, int* nx, int* ny, double* sum_of_elements);

void matar_initialize_()
{
    MATAR_INITIALIZE();
}

void matar_finalize_()
{
    MATAR_FINALIZE();
}

void square_array_elements_(double* array, int* nx, int* ny)
{
    // define private copys of nx and ny
    // this enables kokkos to copy stack variables
    // if used in kokkos kernal
    int nx_ = *nx;
    int ny_ = *ny;

    // create ViewFMatrixDual since array is fortran allocated
    auto array_2D_dual_view = ViewFMatrixDual<double>(array, nx_, ny_);

    // Note: DO_ALL is a macro that expands to a loop over the elements of the array following the 
    // optimal layout for column major order which is the default for fortran arrays
    DO_ALL(j, 1, ny_,
           i, 1, nx_, {
        array_2D_dual_view(i, j) = pow(array_2D_dual_view(i, j), 2);
    });

    array_2D_dual_view.update_host();
}

void sum_array_elements_(double* array, int* nx, int* ny, double* sum_of_elements)
{
    // define private copys of nx and ny
    int nx_ = *nx;
    int ny_ = *ny;

    // create ViewFMatrixDual since array is fortran allocated
    auto array_2D_dual_view = ViewFMatrixDual<double>(array, nx_, ny_);

    double global_sum;
    double local_sum;
    DO_REDUCE_SUM(j, 1, ny_,
                  i, 1, nx_,
                  local_sum, {
        local_sum += array_2D_dual_view(i, j);
    }, global_sum);

    // update sum_of_elements memory location
    *sum_of_elements = global_sum;
}
