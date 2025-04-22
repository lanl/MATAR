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
#include <iostream>
#include <matar.h>

#include <algorithm>  // std::max, std::min, etc.

using namespace mtr; // matar namespace

// main
int main()
{
    // Serial (Host) Data Types
    // ========================

    // 1. Dense Arrays (C-style - Row Major)
    CArray<int> carr_1D(10);           // 1D C-style array
    CArray<int> carr_2D(10, 10);       // 2D C-style array
    CArray<int> carr_3D(10, 10, 10);   // 3D C-style array

    // 2. Dense Arrays (F-style - Column Major)
    FArray<int> farr_1D(10);           // 1D F-style array
    FArray<int> farr_2D(10, 10);       // 2D F-style array
    FArray<int> farr_3D(10, 10, 10);   // 3D F-style array

    // 3. Views of C-style arrays
    ViewCArray<int> view_carr_1D(carr_1D.pointer(), 10);
    ViewCArray<int> view_carr_2D(carr_2D.pointer(), 10, 10);
    ViewCArray<int> view_carr_3D(carr_3D.pointer(), 10, 10, 10);

    // Views of existing 1D arrays
    int A[10];
    ViewCArray<int> view_A(A, 10);

    // View of a std::vector (WARNING: this only works if B is contiguous in memory)
    std::vector<int> B(10);
    ViewCArray<int> view_B(B.data(), 10);

    // 4. Views of F-style arrays
    ViewFArray<int> view_farr_1D(farr_1D.pointer(), 10);
    ViewFArray<int> view_farr_2D(farr_2D.pointer(), 10, 10);
    ViewFArray<int> view_farr_3D(farr_3D.pointer(), 10, 10, 10);


    // 5. Ragged Arrays (C-style)
    RaggedCArray<int> ragged_carr(10);  // 10 rows
    FOR_ALL(i, 0, 10, {
        ragged_carr(i) = CArray<int>(i+1);  // Each row has i+1 elements
    });

    // 6. Ragged Arrays (F-style)
    RaggedFArray<int> ragged_farr(10);  // 10 columns
    FOR_ALL(i, 0, 10, {
        ragged_farr(i) = FArray<int>(i+1);  // Each column has i+1 elements
    });

    // 7. Dynamic Ragged Arrays (C-style)
    DynamicRaggedCArray<int> dyn_ragged_carr;
    FOR_ALL(i, 0, 10, {
        dyn_ragged_carr.push_back(CArray<int>(i+1));
    });

    // 8. Dynamic Ragged Arrays (F-style)
    DynamicRaggedFArray<int> dyn_ragged_farr;
    FOR_ALL(i, 0, 10, {
        dyn_ragged_farr.push_back(FArray<int>(i+1));
    });

    // 9. Sparse Arrays (CSR format)
    CSRArray<int> csr_arr(10, 20);  // 10 rows, 20 non-zero elements
    // Initialize with some non-zero values
    FOR_ALL(i, 0, 10, {
        csr_arr.insert(i, i, i+1);  // Diagonal elements
    });

    // 10. Sparse Arrays (CSC format)
    CSCArray<int> csc_arr(10, 20);  // 10 columns, 20 non-zero elements
    // Initialize with some non-zero values
    FOR_ALL(i, 0, 10, {
        csc_arr.insert(i, i, i+1);  // Diagonal elements
    });

#ifdef HAVE_KOKKOS
    // Kokkos (Device) Data Types
    // =========================

    // 1. Dense Arrays (C-style)
    CArrayDevice<int> carr_dev_1D(10);
    CArrayDevice<int> carr_dev_2D(10, 10);
    CArrayDevice<int> carr_dev_3D(10, 10, 10);

    // 2. Dense Arrays (F-style)
    FArrayDevice<int> farr_dev_1D(10);
    FArrayDevice<int> farr_dev_2D(10, 10);
    FArrayDevice<int> farr_dev_3D(10, 10, 10);

    // 3. Views of C-style arrays
    ViewCArrayDevice<int> view_carr_dev_1D(carr_dev_1D.pointer(), 10);
    ViewCArrayDevice<int> view_carr_dev_2D(carr_dev_2D.pointer(), 10, 10);
    ViewCArrayDevice<int> view_carr_dev_3D(carr_dev_3D.pointer(), 10, 10, 10);

    // 4. Views of F-style arrays
    ViewFArrayDevice<int> view_farr_dev_1D(farr_dev_1D.pointer(), 10);
    ViewFArrayDevice<int> view_farr_dev_2D(farr_dev_2D.pointer(), 10, 10);
    ViewFArrayDevice<int> view_farr_dev_3D(farr_dev_3D.pointer(), 10, 10, 10);

    // 5. Ragged Arrays (C-style)
    RaggedCArrayDevice<int> ragged_carr_dev(10);
    FOR_ALL(i, 0, 10, {
        ragged_carr_dev(i) = CArrayDevice<int>(i+1);
    });

    // 6. Ragged Arrays (F-style)
    RaggedFArrayDevice<int> ragged_farr_dev(10);
    FOR_ALL(i, 0, 10, {
        ragged_farr_dev(i) = FArrayDevice<int>(i+1);
    });

    // 7. Dynamic Ragged Arrays (C-style)
    DynamicRaggedCArrayDevice<int> dyn_ragged_carr_dev;
    FOR_ALL(i, 0, 10, {
        dyn_ragged_carr_dev.push_back(CArrayDevice<int>(i+1));
    });

    // 8. Dynamic Ragged Arrays (F-style)
    DynamicRaggedFArrayDevice<int> dyn_ragged_farr_dev;
    FOR_ALL(i, 0, 10, {
        dyn_ragged_farr_dev.push_back(FArrayDevice<int>(i+1));
    });

    // 9. Sparse Arrays (CSR format)
    CSRArrayDevice<int> csr_arr_dev(10, 20);
    FOR_ALL(i, 0, 10, {
        csr_arr_dev.insert(i, i, i+1);
    });

    // 10. Sparse Arrays (CSC format)
    CSCArrayDevice<int> csc_arr_dev(10, 20);
    FOR_ALL(i, 0, 10, {
        csc_arr_dev.insert(i, i, i+1);
    });
#endif

    std::cout << "MATAR data types demonstration completed successfully" << std::endl;
    return 0;
}
