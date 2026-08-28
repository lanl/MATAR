#ifndef CRAMERSRULE_H
#define CRAMERSRULE_H
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

#include "matar.h"
using namespace mtr;

/////////////////////////////////////////////////////////////////////////////
///
/// \fn det_2x2
///
/// \brief Calculates the determinate of a 2D MATAR device array
///
/// \param array The input array
///
/////////////////////////////////////////////////////////////////////////////
template <typename T>
KOKKOS_INLINE_FUNCTION real_t det_2x2(const T& A) {
    return real_t(A(0, 0)) * real_t(A(1, 1)) - real_t(A(0, 1)) * real_t(A(1, 0));
}  // end of det_2d function

/////////////////////////////////////////////////////////////////////////////
///
/// \fn det_2x2
///
/// \brief Calculates the determinate of a 2D MATAR device array
///
/// \param a00 The 00 component of the array
/// \param a01 The 01 component of the array
/// ....
/// \param a11 The 11 component of the array
///
/////////////////////////////////////////////////////////////////////////////
KOKKOS_INLINE_FUNCTION double det_2x2(const double a00, const double a01, const double a10, const double a11) {
    const double det = (a00 * a11 - a01 * a10);
    return det;
}  // end function

/////////////////////////////////////////////////////////////////////////////
///
/// \fn det_3x3
///
/// \brief Calculates the determinate of a 3D MATAR device array
///
/// \param array The input array
///
/////////////////////////////////////////////////////////////////////////////
template <typename T>
KOKKOS_INLINE_FUNCTION real_t det_3x3(const T& A) {
    const real_t det = A(0, 0) * (A(1, 1) * A(2, 2) - A(2, 1) * A(1, 2)) - A(0, 1) * (A(1, 0) * A(2, 2) - A(2, 0) * A(1, 2)) +
                       A(0, 2) * (A(1, 0) * A(2, 1) - A(2, 0) * A(1, 1));

    return det;
}  // end of det_3d function

/////////////////////////////////////////////////////////////////////////////
///
/// \fn det_3x3
///
/// \brief Calculates the determinate of a 3D MATAR device array
///
/// \param a00 The 00 component of the array
/// \param a01 The 01 component of the array
/// ....
/// \param a22 The 22 component of the array
///
/////////////////////////////////////////////////////////////////////////////
KOKKOS_INLINE_FUNCTION double det_3x3(const double a00, const double a01, const double a02, const double a10, const double a11, const double a12,
                                      const double a20, const double a21, const double a22) {
    const double det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);

    return det;
}  // end function

/////////////////////////////////////////////////////////////////////////////
///
/// \fn det_4x4
///
/// \brief Calculates the determinate of a 4D MATAR device array
///
/// \param array The input routine
///
/////////////////////////////////////////////////////////////////////////////
template <typename T>
KOKKOS_INLINE_FUNCTION real_t det_4x4(const T& A) {
    const real_t det = A(0, 3) * A(1, 2) * A(2, 1) * A(3, 0) - A(0, 2) * A(1, 3) * A(2, 1) * A(3, 0) - A(0, 3) * A(1, 1) * A(2, 2) * A(3, 0) +
                       A(0, 1) * A(1, 3) * A(2, 2) * A(3, 0) + A(0, 2) * A(1, 1) * A(2, 3) * A(3, 0) - A(0, 1) * A(1, 2) * A(2, 3) * A(3, 0) -
                       A(0, 3) * A(1, 2) * A(2, 0) * A(3, 1) + A(0, 2) * A(1, 3) * A(2, 0) * A(3, 1) + A(0, 3) * A(1, 0) * A(2, 2) * A(3, 1) -
                       A(0, 0) * A(1, 3) * A(2, 2) * A(3, 1) - A(0, 2) * A(1, 0) * A(2, 3) * A(3, 1) + A(0, 0) * A(1, 2) * A(2, 3) * A(3, 1) +
                       A(0, 3) * A(1, 1) * A(2, 0) * A(3, 2) - A(0, 1) * A(1, 3) * A(2, 0) * A(3, 2) - A(0, 3) * A(1, 0) * A(2, 1) * A(3, 2) +
                       A(0, 0) * A(1, 3) * A(2, 1) * A(3, 2) + A(0, 1) * A(1, 0) * A(2, 3) * A(3, 2) - A(0, 0) * A(1, 1) * A(2, 3) * A(3, 2) -
                       A(0, 2) * A(1, 1) * A(2, 0) * A(3, 3) + A(0, 1) * A(1, 2) * A(2, 0) * A(3, 3) + A(0, 2) * A(1, 0) * A(2, 1) * A(3, 3) -
                       A(0, 0) * A(1, 2) * A(2, 1) * A(3, 3) - A(0, 1) * A(1, 0) * A(2, 2) * A(3, 3) + A(0, 0) * A(1, 1) * A(2, 2) * A(3, 3);

    return det;
}  // end of det_4x4 function

// ============================================
// Array Inversion routines using Cramers Rule
// ============================================

/////////////////////////////////////////////////////////////////////////////
///
/// \fn invert_2x2
///
/// \brief Inverts a 2x2 MATAR device array using Cramer's rule
///
/// \param A The array to be inverted
/// \param inv The inverse of the passed inarray
/// \param det The determate of the array to be inverted
///
/////////////////////////////////////////////////////////////////////////////
template <typename T>
KOKKOS_INLINE_FUNCTION void invert_2x2(const T& A, const T& inv, const real_t det) {
    inv(0, 0) = A(1, 1) / (det + 1.e-16);
    inv(0, 1) = -A(0, 1) / (det + 1.e-16);
    inv(1, 0) = -A(1, 0) / (det + 1.e-16);
    inv(1, 1) = A(0, 0) / (det + 1.e-16);
}  // end of 2D jacobin inverse

/////////////////////////////////////////////////////////////////////////////
///
/// \fn invert_3x3
///
/// \brief Inverts a 3x3 MATAR device array using Cramer's rule
///
/// \param A The array to be inverted
/// \param inv The inverse of the passed inarray
/// \param det The determate of the array to be inverted
///
/////////////////////////////////////////////////////////////////////////////
template <typename T>
KOKKOS_INLINE_FUNCTION void invert_3x3(const T& A, const T& inv, const real_t det) {
    inv(0, 0) = +(A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)) / (det + 1e-16);
    inv(0, 1) = -(A(0, 1) * A(2, 2) - A(0, 2) * A(2, 1)) / (det + 1e-16);
    inv(0, 2) = +(A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1)) / (det + 1e-16);

    inv(1, 0) = -(A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) / (det + 1e-16);
    inv(1, 1) = +(A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0)) / (det + 1e-16);
    inv(1, 2) = -(A(0, 0) * A(1, 2) - A(0, 2) * A(1, 0)) / (det + 1e-16);

    inv(2, 0) = +(A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0)) / (det + 1e-16);
    inv(2, 1) = -(A(0, 0) * A(2, 1) - A(0, 1) * A(2, 0)) / (det + 1e-16);
    inv(2, 2) = +(A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0)) / (det + 1e-16);

    return;
}  // end of inverse matrix

/////////////////////////////////////////////////////////////////////////////
///
/// \fn invert_3x3
///
/// \brief Inverts a 3x3 MATAR device array using Cramer's rule and returns
///        the determinate of the array
///
/// \param A   The array to be inverted
/// \param inv The inverse of the passed in array
///
/////////////////////////////////////////////////////////////////////////////
template <typename T>
KOKKOS_INLINE_FUNCTION real_t invert_3x3(const T& A, const T& inv) {
    real_t det = A(0, 0) * (A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)) - A(0, 1) * (A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) +
                 A(0, 2) * (A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0));

    inv(0, 0) = +(A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1)) / (det + 1e-16);
    inv(0, 1) = -(A(0, 1) * A(2, 2) - A(0, 2) * A(2, 1)) / (det + 1e-16);
    inv(0, 2) = +(A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1)) / (det + 1e-16);

    inv(1, 0) = -(A(1, 0) * A(2, 2) - A(1, 2) * A(2, 0)) / (det + 1e-16);
    inv(1, 1) = +(A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0)) / (det + 1e-16);
    inv(1, 2) = -(A(0, 0) * A(1, 2) - A(0, 2) * A(1, 0)) / (det + 1e-16);

    inv(2, 0) = +(A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0)) / (det + 1e-16);
    inv(2, 1) = -(A(0, 0) * A(2, 1) - A(0, 1) * A(2, 0)) / (det + 1e-16);
    inv(2, 2) = +(A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0)) / (det + 1e-16);

    return det;

}  // end of inverse matrix

/////////////////////////////////////////////////////////////////////////////
///
/// \fn invert_4x4
///
/// \brief Inverts a 4x4 MATAR device array using Cramer's rule and returns
///        the determinate of the array
///
/// \param A   The array to be inverted
/// \param inv The inverse of the passed in array
///
/////////////////////////////////////////////////////////////////////////////
template <typename T>
KOKKOS_INLINE_FUNCTION real_t invert_4x4(const T& A, const T& inv) {
    // helper array
    real_t cof[4][4];

    // Compute cofactor matrix
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            // Build 3x3 minor matrix excluding row i and column j
            real_t minor[3][3];
            size_t mi = 0;
            for (size_t ii = 0; ii < 4; ++ii) {
                if (ii == i) continue;
                size_t mj = 0;
                for (size_t jj = 0; jj < 4; ++jj) {
                    if (jj == j) continue;
                    minor[mi][mj] = A(ii, jj);
                    ++mj;
                }  // end jj
                ++mi;
            }  // end ii

            cof[i][j] = ((i + j) % 2 == 0 ? 1 : -1) * det_3x3(minor[0][0],
                                                              minor[0][1],
                                                              minor[0][2],
                                                              minor[1][0],
                                                              minor[1][1],
                                                              minor[1][2],
                                                              minor[2][0],
                                                              minor[2][1],
                                                              minor[2][2]);  // function

        }  // end j
    }  // end i

    // Compute determinant from first row and cofactors
    real_t det = real_t(0);
    for (size_t j = 0; j < 4; ++j) {
        det += A(0, j) * cof[0][j];
    }  // end for j

    // Transpose cofactors to get adjugate, then divide by determinant
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            inv(i, j) = cof[j][i] / (det + 1.e-16);
        }  // end j
    }  // end i

    return det;
}  // end function

/////////////////////////////////////////////////////////////////////////////
///
/// \fn invert_4x4
///
/// \brief Inverts a 4x4 MATAR device array using Cramer's rule
///
/// \param A   The array to be inverted
/// \param inv The inverse of the passed in array
/// \param det The determate of the array to be inverted
///
/////////////////////////////////////////////////////////////////////////////
template <typename T>
KOKKOS_INLINE_FUNCTION void invert_4x4(const T& A, const T& inv, const real_t det) {
    // helper array
    real_t cof[4][4];

    // Compute cofactor matrix
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            // Build 3x3 minor matrix excluding row i and column j
            real_t minor[3][3];
            size_t mi = 0;
            for (size_t ii = 0; ii < 4; ++ii) {
                if (ii == i) continue;
                size_t mj = 0;
                for (size_t jj = 0; jj < 4; ++jj) {
                    if (jj == j) continue;
                    minor[mi][mj] = A(ii, jj);
                    ++mj;
                }  // end jj
                ++mi;
            }  // end ii

            cof[i][j] = ((i + j) % 2 == 0 ? 1 : -1) * det_3x3(minor[0][0],
                                                              minor[0][1],
                                                              minor[0][2],
                                                              minor[1][0],
                                                              minor[1][1],
                                                              minor[1][2],
                                                              minor[2][0],
                                                              minor[2][1],
                                                              minor[2][2]);  // function

        }  // end j
    }  // end i

    // Transpose cofactors to get adjugate, then divide by determinant
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            inv(i, j) = cof[j][i] / (det + 1.e-16);
        }  // end j
    }  // end i

    return;
}  // end function

#endif  // CRAMERS