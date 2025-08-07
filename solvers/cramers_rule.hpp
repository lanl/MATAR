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
 

 // Compute inverse of 3x3 matrix using Cramer's rule
KOKKOS_FUNCTION
double invert_3x3(
    const DCArrayKokkos <double> &A,
    const DCArrayKokkos <double> &inv){

    double  det = 
        A(0,0)*(A(1,1)*A(2,2) - A(1,2)*A(2,1)) -
        A(0,1)*(A(1,0)*A(2,2) - A(1,2)*A(2,0)) +
        A(0,2)*(A(1,0)*A(2,1) - A(1,1)*A(2,0));


    inv(0,0) = +(A(1,1)*A(2,2) - A(1,2)*A(2,1)) / (det+1e-16);
    inv(0,1) = -(A(0,1)*A(2,2) - A(0,2)*A(2,1)) / (det+1e-16);
    inv(0,2) = +(A(0,1)*A(1,2) - A(0,2)*A(1,1)) / (det+1e-16);

    inv(1,0) = -(A(1,0)*A(2,2) - A(1,2)*A(2,0)) / (det+1e-16);
    inv(1,1) = +(A(0,0)*A(2,2) - A(0,2)*A(2,0)) / (det+1e-16);
    inv(1,2) = -(A(0,0)*A(1,2) - A(0,2)*A(1,0)) / (det+1e-16);

    inv(2,0) = +(A(1,0)*A(2,1) - A(1,1)*A(2,0)) / (det+1e-16);
    inv(2,1) = -(A(0,0)*A(2,1) - A(0,1)*A(2,0)) / (det+1e-16);
    inv(2,2) = +(A(0,0)*A(1,1) - A(0,1)*A(1,0)) / (det+1e-16);

    return det;

} // end of inverse matrix 



// Helper to compute 3x3 determinant for submatrices
double det3x3(
    double a00, double a01, double a02,
    double a10, double a11, double a12,
    double a20, double a21, double a22) {
    
    return a00 * (a11 * a22 - a12 * a21)
         - a01 * (a10 * a22 - a12 * a20)
         + a02 * (a10 * a21 - a11 * a20);
} // end function

// Compute inverse of 4x4 matrix using Cramer's rule
double invert_4x4(const DCArrayKokkos <double> &A,  
                  const DCArrayKokkos <double> &inv) {
    
    // helper array
    double cof[4][4];

    // Compute cofactor matrix
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            // Build 3x3 minor matrix excluding row i and column j
            double minor[3][3];
            size_t mi = 0;
            for (size_t ii = 0; ii < 4; ++ii) {
                if (ii == i) continue;
                size_t mj = 0;
                for (size_t jj = 0; jj < 4; ++jj) {
                    if (jj == j) continue;
                    minor[mi][mj] = A(ii,jj);
                    ++mj;
                } // end jj
                ++mi;
            } // end ii

            cof[i][j] = ((i + j) % 2 == 0 ? 1 : -1) * det3x3(
                minor[0][0], minor[0][1], minor[0][2],
                minor[1][0], minor[1][1], minor[1][2],
                minor[2][0], minor[2][1], minor[2][2]
            ); // function

        } // end j
    }// end i

    // Compute determinant from first row and cofactors
    double det = 0.0;
    for (size_t j = 0; j < 4; ++j){
        det += A(0,j) * cof[0][j];
    } // end for j

    // if (std::fabs(det) < 1e-12) {
    //     std::cerr << "Matrix is singular or nearly singular.\n";
    //     return false;
    // }

    // Transpose cofactors to get adjugate, then divide by determinant
    for (size_t i = 0; i < 4; ++i){
        for (size_t j = 0; j < 4; ++j){
            inv(i,j) = cof[j][i] / (det+1.e-16);
        } // end j
    } // end i

    return det;
} // end function
