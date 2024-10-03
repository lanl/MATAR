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
#include <stdio.h>
#include "matar.h"

using namespace mtr; // matar namespace

int main()
{
    // DENSE
    int dim0 = 2;
    int dim1 = 3;
    int dim2 = 2;

    printf("HOST TYPES");

    FArray <double> testing (dim0,dim1,dim2);
    ViewFArray <double> testing2 (&testing(0,0,0),3,2);
    testing.set_values(1.3);
    testing2.set_values(2.6);
    printf("ViewFArray set_values 2.6 writing over FArray set_values 1.3.\n");
    for (int i = 0; i < dim2; i++) {
        for (int j = 0; j < dim1; j++) {
            for (int k = 0; k < dim0; k++) {
                printf("%.1f  ", testing(k,j,i));
            }
        }
    } 
    printf("\n");
    for (int i = 0; i < dim2; i++) {
        for (int j = 0; j < dim1; j++) {
            printf("%.1f  ", testing2(j,i));
        }
    }
    printf("\n");
    CArray <double> testing3 (dim0,dim1,dim2);
    ViewCArray <double> testing4 (&testing3(0,0,0),3,2);
    testing3.set_values(1.3);
    testing4.set_values(2.6);
    printf("ViewCArray set_values 2.6 writing over CArray set_values 1.3.\n");
    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            for (int k = 0; k < dim2; k++) {
                printf("%.1f  ", testing3(i,j,k));
            }
        }
    } 
    printf("\n");
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            printf("%.1f  ", testing4(i,j));
        }
    }
    printf("\n");
    CMatrix <double> testing5 (dim0,dim1,dim2);
    ViewCMatrix <double> testing6 (&testing5(1,1,1),3,2);
    testing5.set_values(1.3);
    testing6.set_values(2.6);
    printf("ViewCMatrix set_values 2.6 writing over CMatrix set_values 1.3.\n");
    for (int i = 1; i < dim0+1; i++) {
        for (int j = 1; j < dim1+1; j++) {
            for (int k = 1; k < dim2+1; k++) {
                printf("%.1f  ", testing5(i,j,k));
            }
        }
    } 
    printf("\n");
    for (int i = 1; i < dim1+1; i++) {
        for (int j = 1; j < dim2+1; j++) {
            printf("%.1f  ", testing6(i,j));
        }
    }
    printf("\n");
    FMatrix <double> testing7 (dim0,dim1,dim2);
    ViewFMatrix <double> testing8 (&testing7(1,1,1),3,2);
    testing7.set_values(1.3);
    testing8.set_values(2.6);
    printf("ViewFMatrix set_values 2.6 writing over FMatrix set_values 1.3.\n");
    for (int i = 1; i < dim2+1; i++) {
        for (int j = 1; j < dim1+1; j++) {
            for (int k = 1; k < dim0+1; k++) {
                printf("%.1f  ", testing7(k,j,i));
            }
        }
    } 
    printf("\n");
    for (int i = 1; i < dim2+1; i++) {
        for (int j = 1; j < dim1+1; j++) {
            printf("%.1f  ", testing8(j,i));
        }
    }
    printf("\n");

    // RAGGEDS
    CArray <size_t> stridesright (3);
    stridesright(0) = 2;
    stridesright(1) = 3;
    stridesright(2) = 2;
    RaggedRightArray <double> righttest (stridesright);
    righttest.set_values(1.35);
    printf("RaggedRightArray set to values of 1.35\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < stridesright(i); j++) {
            printf("%.2f  ", righttest(i,j));
        }
    }
    printf("\n");
    CArray <size_t> stridesdown (4);
    stridesdown(0) = 2;
    stridesdown(1) = 3;
    stridesdown(2) = 2;
    stridesdown(3) = 1;
    RaggedDownArray <double> downtest (stridesdown);
    downtest.set_values(2.55);
    printf("RaggedDownArray set to values of 2.55\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < stridesdown(i); j++) {
            printf("%.2f  ", downtest(j,i));
        }
    }
    printf("\n");
    DynamicRaggedRightArray <double> dynright (3,4);
    dynright.stride(0) = 1;
    dynright.stride(1) = 3;
    dynright.stride(2) = 2;
    dynright.set_values(2.14);
    dynright.set_values_sparse(1.35);
    printf("The values within the populated strides of the DynamicRaggedRight are set to 1.35 and the data in the rest of the array is set to 2.14.\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.2f  ", dynright(i,j));
        }
        printf("\n");
    }
    DynamicRaggedDownArray <double> dyndown (3,4);
    dyndown.stride(0) = 1;
    dyndown.stride(1) = 3;
    dyndown.stride(2) = 2;
    dyndown.stride(3) = 1;
    dyndown.set_values(2.14);
    dyndown.set_values_sparse(1.35);
    printf("The values within the populated strides of the DynamicRaggedDown are set to 1.35 and the data in the rest of the array is set to 2.14.\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.2f  ", dyndown(i,j));
        }
        printf("\n");
    } 
    printf("Ragged Right Array of Vectors, CSCArray, and CSRArray are not currently tested in this file as of 8/6/24.\n");


    printf("DUAL TYPES");
    Kokkos::initialize();
    {
        DFArrayKokkos <double> DFAtest (2, 3, 4);
        DFAtest.set_values(1.25);
        printf("DViewFArrayKokkos set_values 2.34 writing over DFArrayKokkos set_values 1.25.\n");
        FOR_ALL(i, 0, 4,
                j, 0, 3,
                k, 0, 2, {
                printf("%.2f ", DFAtest(k,j,i));
        });
        printf("\n");
        DFMatrixKokkos <double> DFMtest (2, 3, 4);
        DFMtest.set_values(1.33);
        printf("DViewFMatrixKokkos set_values 3.24 writing over DFMatrixKokkos set_values 1.33.\n");
        FOR_ALL(i, 1, 5,
                j, 1, 4,
                k, 1, 3, {
                printf("%.2f ", DFMtest(k,j,i));
        });
        printf("\n");
        DCArrayKokkos <double> DCAtest (2, 3, 4);
        DCAtest.set_values(1.53);
        printf("DViewCArrayKokkos set_values 2.33 writing over DCArrayKokkos set_values 1.53.\n");
        FOR_ALL(i, 0, 4,
                j, 0, 3,
                k, 0, 2, {
                printf("%.2f ", DCAtest(k,j,i));
        });
        printf("\n");
        DCMatrixKokkos <double> DCMtest (2, 3, 4);
        DCMtest.set_values(1.77);
        printf("DViewCMatrixKokkos set_values 2.17 writing over DCMatrixKokkos set_values 1.77.\n");
        FOR_ALL(i, 1, 5,
                j, 1, 4,
                k, 1, 3, {
                printf("%.2f ", DCMtest(k,j,i));
        });
        printf("\n");

        DynamicRaggedRightArrayKokkos <double> dynrightK (3,4);
        //dynrightK.stride(0) = 1;
        //dynrightK.stride(1) = 3;
        //dynrightK.stride(2) = 2;
        RUN({
            dynrightK.stride(0) = 1;
            dynrightK.stride(1) = 3;
            dynrightK.stride(2) = 2;
        });
        dynrightK.set_values(2.14);
        dynrightK.set_values_sparse(1.35);
        printf("The values within the populated strides of the DynamicRaggedRight are set to 1.35 and the data in the rest of the array is set to 2.14.\n");
        FOR_FIRST(i, 3, {
            FOR_SECOND(j, 0, dynrightK.stride(i), {
                //printf("%.2f  ", dynrightK(i,j));
            });
        });
        printf("\n");
        //for (int i = 0; i < 3; i++) {
        //    for (int j = 0; j < 4; j++) {
                //printf("%.2f  ", dynrightK(i,j));
        //    }
            //printf("\n");
        //}
        DynamicRaggedDownArrayKokkos <double> dyndownK (3,4);
        RUN({
            dyndownK.stride(0) = 1;
            dyndownK.stride(1) = 3;
            dyndownK.stride(2) = 2;
            dyndownK.stride(3) = 1;
        });
        dyndownK.set_values(2.14);
        dyndownK.set_values_sparse(1.35);
        printf("The values within the populated strides of the DynamicRaggedDown are set to 1.35 and the data in the rest of the array is set to 2.14.\n");
        FOR_FIRST(i, 4, {
            FOR_SECOND(j, 0, dyndownK.stride(i), {
                //printf("%.2f  ", dyndownK(i,j));
            });
        });
        printf("\n");
        //for (int i = 0; i < 4; i++) {
        //    for (int j = 0; j < 3; j++) {
        //    }
        //} 
    }
    Kokkos::finalize();    
}
