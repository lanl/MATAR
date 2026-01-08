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

// -----------------------------------------------
// level set to stl file
// by: Andrew Morgan
//
// The following rountines in this code came from:
//   https://paulbourke.net/geometry/polygonise/
//   - Polygonise
//   - VertexInterp
// -----------------------------------------------

using namespace mtr;

static const int edgeTable[256]={
        0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
        0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
        0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
        0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
        0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
        0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
        0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
        0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
        0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
        0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
        0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
        0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
        0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
        0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
        0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
        0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
        0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
        0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
        0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
        0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
        0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
        0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
        0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
        0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
        0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
        0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
        0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
        0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
        0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
        0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
        0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
        0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   };
    
static const int triTable[256][16] =
    {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
        {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
        {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
        {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
        {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
        {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
        {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
        {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
        {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
        {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
        {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
        {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
        {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
        {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
        {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
        {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
        {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
        {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
        {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
        {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
        {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
        {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
        {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
        {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
        {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
        {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
        {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
        {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
        {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
        {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
        {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
        {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
        {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
        {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
        {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
        {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
        {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
        {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
        {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
        {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
        {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
        {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
        {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
        {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
        {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
        {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
        {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
        {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
        {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
        {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
        {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
        {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
        {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
        {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
        {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
        {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
        {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
        {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
        {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
        {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
        {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
        {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
        {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
        {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
        {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
        {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
        {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
        {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
        {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
        {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
        {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
        {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
        {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
        {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
        {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
        {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
        {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
        {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
        {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
        {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
        {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
        {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
        {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
        {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
        {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
        {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
        {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
        {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
        {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
        {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
        {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
        {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
        {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
        {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
        {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
        {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
        {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
        {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
        {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
        {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
        {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
        {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
        {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
        {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
        {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
        {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
        {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
        {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
        {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
        {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
        {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
        {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
        {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
        {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
        {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
        {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
        {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
        {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
        {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
        {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
        {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
        {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
        {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
        {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
        {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
        {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
        {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
        {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
        {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
        {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
        {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
        {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
        {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
        {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
        {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
        {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
        {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
        {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
        {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
        {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
        {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
        {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
        {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
        {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
        {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
        {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
        {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
        {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
        {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
        {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
        {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
        {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
        {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
        {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
        {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
        {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
        {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
        {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
        {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
        {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
        {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
        {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
        {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
        {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
        {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
        {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
        {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
        {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
        {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
        {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
        {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
        {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
        {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
        {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
        {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
        {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};

// a vector type with 3 components
struct vec_t{
    double x;
    double y;
    double z;
    
    // default constructor
    vec_t() : x(0.0), y(0.0), z(0.0) {};

    // overloaded constructor for 2D
    vec_t(double x_in, double y_in){
        x = x_in;
        y = y_in;
        z = 0;
    };
    
    // overloaded constructor
    vec_t(double x_in, double y_in, double z_in){
        x = x_in;
        y = y_in;
        z = z_in;
    };

    // Overload the addition operator
    KOKKOS_INLINE_FUNCTION
    vec_t operator+(const vec_t& other) const {
        return vec_t(x + other.x, y + other.y, z + other.z);
    }

    // Overload the subtraction operator
    KOKKOS_INLINE_FUNCTION
    vec_t operator-(const vec_t& other) const {
        return vec_t(x - other.x, y - other.y, z - other.z);
    }

    // Overload the multiplication operator (scalar)
    KOKKOS_INLINE_FUNCTION
    vec_t operator*(double scalar) const {
        return vec_t(x * scalar, y * scalar, z * scalar);
    }

    // Overload the division operator (scalar)
    KOKKOS_INLINE_FUNCTION
    vec_t operator/(double scalar) const {
        return vec_t(x / scalar, y / scalar, z / scalar);
    }
    
}; // end vec_t




// a triangle data type
struct triangle_t {
    
    vec_t normal; // surface normal
    
    vec_t p[3];   // three nodes with x,y,z coords
    
    // default constructor
    triangle_t(){};
    
    // overloaded constructor
    triangle_t (vec_t p_in[3])
    {
        p[0]=p_in[0];
        p[1]=p_in[1];
        p[2]=p_in[2];
    };
    
}; // end triangle_t


// calculate the surface normal of a triangle
KOKKOS_INLINE_FUNCTION
void calc_normal(triangle_t *triangle){
    
    //A = p1 - p0;
    //B = p2 - p0;
    vec_t A;
    A.x = triangle->p[1].x - triangle->p[0].x;
    A.y = triangle->p[1].y - triangle->p[0].y;
    A.z = triangle->p[1].z - triangle->p[0].z;
    
    vec_t B;
    B.x = triangle->p[2].x - triangle->p[0].x;
    B.y = triangle->p[2].y - triangle->p[0].y;
    B.z = triangle->p[2].z - triangle->p[0].z;
    
    vec_t N;
    N.x = A.y * B.z - A.z * B.y;
    N.y = A.z * B.x - A.x * B.z;
    N.z = A.x * B.y - A.y * B.x;
    
    double mag;
    mag = sqrt(N.x*N.x + N.y*N.y + N.z*N.z);
    if (mag > 1.e-12) {
        // save the unit normal
        triangle->normal.x = N.x/(mag);
        triangle->normal.y = N.y/(mag);
        triangle->normal.z = N.z/(mag);
    }
    else{
        triangle->normal.x = 0.0;
        triangle->normal.y = 0.0;
        triangle->normal.z = 0.0;
    }
    
} // end normal


// cross prodcut
vec_t cross(const vec_t &a, const vec_t &b) {
    return {a.y*b.z - a.z*b.y,
            a.z*b.x - a.x*b.z,
            a.x*b.y - a.y*b.x};
}

double dot(const vec_t &a, const vec_t &b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}


// calculate the volume of a tet with this triangular face
double compute_volume(const triangle_t &triangle) {
    // triangle.p[0] is the first vec_t, being node 0
    // ...
    // triangle.p[1] is the third vec_t, being node 2
    double volume = dot(triangle.p[0], cross(triangle.p[1], triangle.p[2])) / 6.0;

    return volume;
}

struct gridcell_t {
    
    vec_t p[8];
    double val[8];
    
    // default constructor
    gridcell_t(){};
    
    // overloaded constructor
    gridcell_t (vec_t p_in[8], double val_in[8])
    {
        for (size_t i = 0; i < 8; i++) {
            p[i] = p_in[i];
            val[i] = val_in[i];
        }
    };
    
}; // end gridcell_t


/*
   Linearly interpolate the position where an isosurface cuts
   an edge between two vertices, each with their own scalar value
*/
KOKKOS_INLINE_FUNCTION
vec_t VertexInterp(double isolevel, vec_t p1, vec_t p2, double valp1, double valp2)
{
   double mu;
   vec_t p;

   if (fabs(isolevel-valp1) < 1.e-12)
      return(p1);
   if (fabs(isolevel-valp2) < 1.e-12)
      return(p2);
   //if (fabs(valp1-valp2) < 1.e-12)
   //   return(p1);
   mu = (isolevel - valp1) / (valp2 - valp1 + 1.e-12);
   p.x = p1.x + mu * (p2.x - p1.x);
   p.y = p1.y + mu * (p2.y - p1.y);
   p.z = p1.z + mu * (p2.z - p1.z);

   return(p);
}

/*
   Given a grid cell and an isolevel, calculate the triangular
   facets required to represent the isosurface through the cell.
   Return the number of triangular facets, the array "triangles"
   will be loaded up with the vertices at most 5 triangular facets.
    0 will be returned if the grid cell is either totally above
   of totally below the isolevel.
*/
KOKKOS_INLINE_FUNCTION
int Polygonise(gridcell_t grid, double isolevel, triangle_t *triangles)
{
    
    int i,ntriang;
    int cubeindex;
    vec_t vertlist[12];
    
    /*
      Determine the index into the edge table which
      tells us which vertices are inside of the surface
     */
    cubeindex = 0;

    if (grid.val[0] < isolevel) cubeindex |= 1;
    if (grid.val[1] < isolevel) cubeindex |= 2;
    if (grid.val[2] < isolevel) cubeindex |= 4;
    if (grid.val[3] < isolevel) cubeindex |= 8;
    if (grid.val[4] < isolevel) cubeindex |= 16;
    if (grid.val[5] < isolevel) cubeindex |= 32;
    if (grid.val[6] < isolevel) cubeindex |= 64;
    if (grid.val[7] < isolevel) cubeindex |= 128;
    
    
    
    /* Cube is entirely in/out of the surface */
    if (edgeTable[cubeindex] == 0)
        return(0);
    
    /* Find the vertices where the surface intersects the cube */
    if (edgeTable[cubeindex] & 1)
        vertlist[0] =
         VertexInterp(isolevel,grid.p[0],grid.p[1],grid.val[0],grid.val[1]);
    if (edgeTable[cubeindex] & 2)
        vertlist[1] =
         VertexInterp(isolevel,grid.p[1],grid.p[2],grid.val[1],grid.val[2]);
    if (edgeTable[cubeindex] & 4)
        vertlist[2] =
         VertexInterp(isolevel,grid.p[2],grid.p[3],grid.val[2],grid.val[3]);
    if (edgeTable[cubeindex] & 8)
        vertlist[3] =
         VertexInterp(isolevel,grid.p[3],grid.p[0],grid.val[3],grid.val[0]);
    if (edgeTable[cubeindex] & 16)
        vertlist[4] =
         VertexInterp(isolevel,grid.p[4],grid.p[5],grid.val[4],grid.val[5]);
    if (edgeTable[cubeindex] & 32)
        vertlist[5] =
         VertexInterp(isolevel,grid.p[5],grid.p[6],grid.val[5],grid.val[6]);
    if (edgeTable[cubeindex] & 64)
        vertlist[6] =
         VertexInterp(isolevel,grid.p[6],grid.p[7],grid.val[6],grid.val[7]);
    if (edgeTable[cubeindex] & 128)
        vertlist[7] =
         VertexInterp(isolevel,grid.p[7],grid.p[4],grid.val[7],grid.val[4]);
    if (edgeTable[cubeindex] & 256)
        vertlist[8] =
         VertexInterp(isolevel,grid.p[0],grid.p[4],grid.val[0],grid.val[4]);
    if (edgeTable[cubeindex] & 512)
        vertlist[9] =
         VertexInterp(isolevel,grid.p[1],grid.p[5],grid.val[1],grid.val[5]);
    if (edgeTable[cubeindex] & 1024)
        vertlist[10] =
         VertexInterp(isolevel,grid.p[2],grid.p[6],grid.val[2],grid.val[6]);
    if (edgeTable[cubeindex] & 2048)
        vertlist[11] =
         VertexInterp(isolevel,grid.p[3],grid.p[7],grid.val[3],grid.val[7]);
    
    /* Create the triangle */
    ntriang = 0;
    for (i=0; triTable[cubeindex][i]!=-1; i+=3) {
        
        triangles[ntriang].p[0] = vertlist[triTable[cubeindex][i  ]];
        triangles[ntriang].p[1] = vertlist[triTable[cubeindex][i+1]];
        triangles[ntriang].p[2] = vertlist[triTable[cubeindex][i+2]];
        
        ntriang++;
    } // end for i
    
    return(ntriang);
}



// export triangles as ASCII STL file
void write_ascii_stl(const DCArrayKokkos <triangle_t> &all_mesh_surf_triangles,
                     const DCArrayKokkos <size_t> &num_triangles_in_elem){
        
        const size_t num_elems = all_mesh_surf_triangles.dims(0);

        uint32_t num_triangles = 0;
        uint32_t count_lcl = 0;
        FOR_REDUCE_SUM(elem_gid, 0, num_elems, count_lcl, {
            count_lcl += num_triangles_in_elem(elem_gid);
        }, num_triangles);

        FILE * myfile;
        myfile=fopen("surface.stl","w");
        fprintf(myfile,"solid points \n");
        // a serial file write
        for(size_t elem_gid=0; elem_gid<num_elems; elem_gid++){
            for (size_t tri = 0; tri < num_triangles_in_elem.host(elem_gid); tri++){
        
                fprintf(myfile,"facet normal %f %f %f\n",
                        all_mesh_surf_triangles.host(elem_gid,tri).normal.x,
                        all_mesh_surf_triangles.host(elem_gid,tri).normal.y,
                        all_mesh_surf_triangles.host(elem_gid,tri).normal.z);
                
                fprintf(myfile,"outer loop \n");
                
                fprintf(myfile,"vertex %f %f %f\n",
                        all_mesh_surf_triangles.host(elem_gid,tri).p[0].x,
                        all_mesh_surf_triangles.host(elem_gid,tri).p[0].y,
                        all_mesh_surf_triangles.host(elem_gid,tri).p[0].z);
                
                fprintf(myfile,"vertex %f %f %f\n",
                        all_mesh_surf_triangles.host(elem_gid,tri).p[1].x,
                        all_mesh_surf_triangles.host(elem_gid,tri).p[1].y,
                        all_mesh_surf_triangles.host(elem_gid,tri).p[1].z);
                
                fprintf(myfile,"vertex %f %f %f\n",
                        all_mesh_surf_triangles.host(elem_gid,tri).p[2].x,
                        all_mesh_surf_triangles.host(elem_gid,tri).p[2].y,
                        all_mesh_surf_triangles.host(elem_gid,tri).p[2].z);
                fprintf(myfile,"endloop \n");
                fprintf(myfile,"endfacet \n");
            }   
        } // end loop over triangles

        fprintf(myfile,"endsolid points \n");
        
        fclose(myfile);

        std::cout << "Wrote " << num_triangles << " triangles to " << "surface.stl" << std::endl;
}


// Function to write a binary STL file
void write_binary_stl(const DCArrayKokkos <triangle_t> &all_mesh_surf_triangles,
                      const DCArrayKokkos <size_t> &num_triangles_in_elem){

    const std::string& filename = "surface.stl";

    const size_t num_elems = all_mesh_surf_triangles.dims(0);

    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file) {
        std::cerr << "Error: cannot open file " << filename << " for writing.\n";
        return;
    }

    // --- 80-byte header ---
    char header[80];
    std::memset(header, 0, 80);
    std::strncpy(header, "Generated by C++ Binary STL Writer", 79);
    file.write(header, 80);

    uint32_t num_triangles = 0;
    uint32_t count_lcl = 0;
    FOR_REDUCE_SUM(elem_gid, 0, num_elems, count_lcl, {
         count_lcl += num_triangles_in_elem(elem_gid);
    }, num_triangles);

    // --- Number of triangles ---
    file.write(reinterpret_cast<const char*>(&num_triangles), sizeof(uint32_t));

    // --- Each triangle record ---
    for(size_t elem_gid=0; elem_gid<num_elems; elem_gid++){
        for (size_t tri = 0; tri < num_triangles_in_elem.host(elem_gid); tri++){

            const auto& tri_ref = all_mesh_surf_triangles.host(elem_gid, tri);

            // STL expects float precision
            auto write_vec_as_float = [&](const vec_t& v) {
                float xf = static_cast<float>(v.x);
                float yf = static_cast<float>(v.y);
                float zf = static_cast<float>(v.z);
                file.write(reinterpret_cast<const char*>(&xf), sizeof(float));
                file.write(reinterpret_cast<const char*>(&yf), sizeof(float));
                file.write(reinterpret_cast<const char*>(&zf), sizeof(float));
            };

            // normal
            write_vec_as_float(tri_ref.normal);
            // 3 vertices
            write_vec_as_float(tri_ref.p[0]);
            write_vec_as_float(tri_ref.p[1]);
            write_vec_as_float(tri_ref.p[2]);

            // Attribute byte count (2 bytes)
            uint16_t attributeByteCount = 0;
            file.write(reinterpret_cast<const char*>(&attributeByteCount), sizeof(uint16_t));


        } // end tri
    } // end elem

    file.close();

    std::cout << "Wrote " << num_triangles << " triangles to " << filename << std::endl;
}


// -----------------------------------------------------------------------------
// Use marching cubes build triangulated surfaces and exports it as a .stl file
// -----------------------------------------------------------------------------
void marching_cubes(const DCArrayKokkos <double> &x,
                    const DCArrayKokkos <double> &y,
                    const DCArrayKokkos <double> &z,
                    const DCArrayKokkos <double> &levelset,
                    const double isoLevel){


        size_t num_pt_x = x.size();
        size_t num_pt_y = y.size();
        size_t num_pt_z = z.size();


        // define the triangles of extracted surface
        const size_t num_elems = (num_pt_x-1)*(num_pt_y-1)*(num_pt_z-1);
        DCArrayKokkos <triangle_t> all_mesh_surf_triangles(num_elems, 5, "mesh_surf_tris"); // max of 5 per elem
        DCArrayKokkos <size_t> num_triangles_in_elem(num_elems, "num_tris_in_elem");
        num_triangles_in_elem.set_values(0);



        printf("Running marching cubes algorithm\n");
        
        FOR_ALL(k, 0, num_pt_z-1,
                j, 0, num_pt_y-1,
                i, 0, num_pt_x-1, {

                    // elem gid
                    size_t elem_gid = i + j*(num_pt_x-1) + k*(num_pt_x-1)*(num_pt_y-1);
        
                    // extract the x,y,z node coords
                    // using the index ordering for the cell
                    vec_t xyzs [8];
                    xyzs[0] = vec_t(x(i  ), y(j  ), z(k  ));
                    xyzs[1] = vec_t(x(i+1), y(j  ), z(k  ));
                    xyzs[2] = vec_t(x(i+1), y(j  ), z(k+1));
                    xyzs[3] = vec_t(x(i  ), y(j  ), z(k+1));
                    xyzs[4] = vec_t(x(i  ), y(j+1), z(k  ));
                    xyzs[5] = vec_t(x(i+1), y(j+1), z(k  ));
                    xyzs[6] = vec_t(x(i+1), y(j+1), z(k+1));
                    xyzs[7] = vec_t(x(i  ), y(j+1), z(k+1));
        
        
                    // extract the values at the nodes
                    // using the index ordering for the cell
                    double vals [8];
                    vals[0] = levelset(i  ,j  ,k  );
                    vals[1] = levelset(i+1,j  ,k  );
                    vals[2] = levelset(i+1,j  ,k+1);
                    vals[3] = levelset(i  ,j  ,k+1);
                    vals[4] = levelset(i  ,j+1,k  );
                    vals[5] = levelset(i+1,j+1,k  );
                    vals[6] = levelset(i+1,j+1,k+1);
                    vals[7] = levelset(i  ,j+1,k+1);
        
        
                    // details of the cell, save coords and the values at the nodes
                    gridcell_t cell(xyzs, vals);
        
        
                    // the most triangles in a cell is 5
                    triangle_t triangles[5];
                    num_triangles_in_elem(elem_gid) = Polygonise(cell, isoLevel, triangles);
        
                    // save the triangles
                    for (size_t tri = 0; tri < num_triangles_in_elem(elem_gid); tri++)
                    {
                        all_mesh_surf_triangles(elem_gid,tri) = triangles[tri];
                    } // end for tri

        });  // end parallel for k,j,i

        
        // calculate the normal vector of triangles
        FOR_ALL(elem_gid, 0, num_elems, {
            for (size_t tri = 0; tri < num_triangles_in_elem(elem_gid); tri++){
                calc_normal(&all_mesh_surf_triangles(elem_gid, tri));
            }
        }); // end loop over triangles

        all_mesh_surf_triangles.update_host();
        num_triangles_in_elem.update_host();

        
        printf("Marching cubes finished \n\n");
        

        // --------------------------------------------------
        // volume calculation
        // --------------------------------------------------

        double volume = 0.0;
        double vol_lcl = 0.0;
        FOR_REDUCE_SUM(elem_gid, 0, num_elems, 
                       vol_lcl, {

            for (size_t tri = 0; tri < num_triangles_in_elem(elem_gid); tri++){
                vol_lcl += compute_volume(all_mesh_surf_triangles(elem_gid,tri)); 
            }

        }, volume);
        volume = fabs(volume);
        printf("volume = %f \n", volume);



        // --------------------------------------------------
        // Export STL file using results from marching cubes
        // --------------------------------------------------

        printf("Exporting STL file for a 3D printer\n");
        
        
        //write_ascii_stl(all_mesh_surf_triangles, num_triangles_in_elem);
        write_binary_stl(all_mesh_surf_triangles, num_triangles_in_elem);
            
    
        printf("Finished \n\n");

} // end of marching cubes code



