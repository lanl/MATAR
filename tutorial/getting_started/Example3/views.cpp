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
    // A view example
    int A[10];
    ViewCArray<int> arr(A, 10);
    FOR_ALL(i, 0, 10, {
        arr(i) = 314;
    });

    // A 2D array example
    CArray<int> arr_2D(10, 10);
    FOR_ALL(i, 0, 10,
             j, 0, 10, {
        arr_2D(i, j) = 314;
    });

    // A 3D array example
    CArray<int> arr_3D(10, 10, 10);
    FOR_ALL(i, 0, 10,
             j, 0, 10,
             k, 0, 10, {
        arr_3D(i, j, k) = 314;
    });

    int loc_sum = 0;
    int result  = 0;
    FOR_REDUCE_SUM(i, 0, 10,
               loc_sum, {
        loc_sum += arr(i) * arr(i);
               }, result);

    // testing
    loc_sum = 0;
    for (int i = 0; i < 10; i++) {
        loc_sum += 314 * 314;
    }
    std::cout << "1D reduce : " << result << " vs. " << loc_sum << " \n";

    loc_sum = 0;
    result  = 0;
    FOR_REDUCE_SUM(i, 0, 10,
                   j, 0, 10,
                   loc_sum, {
        loc_sum += arr_2D(i, j) * arr_2D(i, j);
    }, result);

    // testing
    loc_sum = 0;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            loc_sum += 314 * 314;
        }
    }
    std::cout << "2D reduce : " << result << " vs. " << loc_sum << " \n";

    loc_sum = 0;
    result  = 0;
    FOR_REDUCE_SUM(i, 0, 10,
                   j, 0, 10,
                   k, 0, 10,
                   loc_sum, {
        loc_sum += arr_3D(i, j, k) * arr_3D(i, j, k);
    }, result);

    // testing
    loc_sum = 0;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                loc_sum += 314 * 314;
            }
        }
    }
    std::cout << "3D reduce : " << result << " vs. " << loc_sum << " \n";

    int loc_max;
    FOR_REDUCE_MAX(i, 0, 10,
                   j, 0, 10,
                   k, 0, 10,
                   loc_max, {
        loc_max = std::max<int>(arr_3D(i, j, k), loc_max);
    }, result);

    std::cout << "3D reduce MAX : " << result << " \n";

    int loc_min;
    FOR_REDUCE_MIN(i, 0, 10,
                   j, 0, 10,
                   k, 0, 10,
                   loc_min, {
        loc_min = std::min<int>(arr_3D(i, j, k), loc_min);
    }, result);

    std::cout << "3D reduce MIN : " << result << " \n";

    FOR_REDUCE_MIN_CLASS(i, 0, 10,
                         j, 0, 10,
                         k, 0, 10,
                         loc_min, {
        loc_min = std::min<int>(arr_3D(i, j, k), loc_min);
    }, result);

    std::cout << "3D reduce MIN CLASS : " << result << " \n";

    std::cout << "done" << std::endl;

    return 0;
}
