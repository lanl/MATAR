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


#include <fstream>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cmath>


#include "matar.h"
#include "levelset_to_stl.hpp"



// -----------------------------------------------
// -----------------------------------------------
// inputs:

const double box_dims = 15; // box dims
const double r = 0.375; // radius of struts

// parameters for NPR lattice option
const double h2 = box_dims;
const double alpha1 = 0.5;
const double alphah = 0.5;

// the number of nodes in the mesh
const int num_pt_x = 300; // resolution
const int num_pt_y = 300; // resolution
const int num_pt_z = 300; // resolution

// build sc=0; bcc=1; fcc=2; octet=3; other=4
const size_t build_lattice_type = 4;

// -----------------------------------------------
// -----------------------------------------------

// spacing
const double dx0 = (box_dims)/((double)num_pt_x-3.0); // delta's for extending mesh, +2 for ghosts
const double dy0 = (box_dims)/((double)num_pt_y-3.0); 
const double dz0 = (box_dims)/((double)num_pt_z-3.0); 

// the mesh dimensions
const double XMax = box_dims + 1.5*dx0; 
const double YMax = box_dims + 1.5*dy0; 
const double ZMax = box_dims + 1.5*dz0; 

const double X0 = 0.0 - 1.5*dx0; 
const double Y0 = 0.0 - 1.5*dy0; 
const double Z0 = 0.0 - 1.5*dz0; 

const double LX = (XMax - X0);   // length in x-dir
const double LY = (YMax - Y0);
const double LZ = (ZMax - Z0);

const double dx = LX/((double)num_pt_x-1.0); 
const double dy = LY/((double)num_pt_y-1.0); 
const double dz = LZ/((double)num_pt_z-1.0); 

const double isoLevel=0.0; // contour to extract

//
// -----------------------------------------------


// --- functions for vec_t ---

KOKKOS_INLINE_FUNCTION
double length(const vec_t &p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

KOKKOS_INLINE_FUNCTION
vec_t vabs(const vec_t &p)
{
    return vec_t (fabs(p.x), fabs(p.y), fabs(p.z));
}

// max vector and scalar
KOKKOS_INLINE_FUNCTION
vec_t vmax(const vec_t &a, const double b)
{
    return vec_t (fmax(a.x, b), fmax(a.y, b), fmax(a.z, b));
}

// max of two vectors
KOKKOS_INLINE_FUNCTION
vec_t vmax(const vec_t &a, const vec_t &b)
{
    return vec_t (fmax(a.x, b.x), fmax(a.y, b.y), fmax(a.z, b.z));
}

KOKKOS_INLINE_FUNCTION
vec_t vmin(const vec_t &a, const vec_t &b)
{
    return vec_t (fmin(a.x, b.x), fmin(a.y, b.y), fmin(a.z, b.z));
}



KOKKOS_INLINE_FUNCTION
double op_union(const double d1,const double d2 )
{
    return fmin(d1,d2);
}

KOKKOS_INLINE_FUNCTION
double op_subtraction(const double d1,const double d2 )
{
    return fmax(-d1,d2);
}

KOKKOS_INLINE_FUNCTION
double op_intersection(const double d1,const double d2 )
{
    return fmax(d1,d2);
}



// --- geometries ---

// r is the radius
KOKKOS_INLINE_FUNCTION
double sphere(const vec_t &p, const vec_t center, const double r)
{
    return length(p - center)-r;
}

// a capsule going from point a to point b (defining a centerline) with radius r
KOKKOS_INLINE_FUNCTION
double capsule(const vec_t &p, const vec_t &a, const vec_t &b, const double r)
{
    vec_t pa = p - a;
    vec_t ba = b - a;
    
    double h = fmin(fmax(dot(pa,ba)/dot(ba,ba), 0.0), 1.0);  // clamp(blah) to be in range [0:1];

    return length( pa - ba*h ) - r;
}

// p is the query point
// b is the half-sizes of the box in x,y,z
// center = center location of box
KOKKOS_INLINE_FUNCTION
double box( vec_t p, vec_t b, vec_t center)
{
    // Shift point into the box’s local coordinates
    vec_t q = vabs(p - center) - b; // b=halfsizes (bx, by, bz))

    // Standard SDF for axis-aligned box
    return length(vmax(q,0.0)) + fmin(fmax(q.x,fmax(q.y,q.z)),0.0);

    // verbose:
    //vec_t d = vabs(p - center) - b;
    //vec_t d_out = vmax(d, 0.0);
    //double outside = length(d_out);
    //double inside = fmin(fmax(d.x, fmax(d.y, d.z)), 0.0);
    //return outside + inside;

}



// b is the half size
// e is the frame thickness
KOKKOS_INLINE_FUNCTION
double boxframe(vec_t p, const vec_t b, vec_t center, const double e)
{
    p = vabs(p - center) - b;
    vec_t q = vabs(p + vec_t(e,e,e)) - vec_t(e,e,e);

    double d1 = length(vmax(vec_t(p.x, q.y, q.z), 0.0)) + fmin(fmax(p.x, fmax(q.y, q.z)), 0.0);
    double d2 = length(vmax(vec_t(q.x, p.y, q.z), 0.0)) + fmin(fmax(q.x, fmax(p.y, q.z)), 0.0);
    double d3 = length(vmax(vec_t(q.x, q.y, p.z), 0.0)) + fmin(fmax(q.x, fmax(q.y, p.z)), 0.0);

    return fmin(fmin(d1, d2), d3);
}


//  ==========================
//  Nodal indexing convention
//  ==========================
//  
//                K
//                ^         J
//                |        /
//                |       /
//                |      /
//        6------------------7
//       /|                 /|
//      / |                / |
//     /  |               /  |
//    /   |              /   |
//   /    |             /    |
//  4------------------5     |
//  |     |            |     | ----> I
//  |     |            |     |
//  |     |            |     |
//  |     |            |     |
//  |     2------------|-----3
//  |    /             |    /
//  |   /              |   /
//  |  /               |  /
//  | /                | /
//  |/                 |/
//  0------------------1
//
//  ==========================



// a0....a7 are the coordinates in the box
// r is the radius
KOKKOS_INLINE_FUNCTION
double simple_cube(const vec_t &p,
                   const vec_t &a0,
                   const vec_t &a1,
                   const vec_t &a2,
                   const vec_t &a3,
                   const vec_t &a4,
                   const vec_t &a5,
                   const vec_t &a6,
                   const vec_t &a7,
                   const vec_t &center,
                   const vec_t &box_dim,
                   const double r){

    // variable name nomenclature
    // e04 means an edge between node 0 and node 4

    // surface 0 [0,4,6,2] xi-minus dir
    const double e04e46 = op_union(capsule(p,a0,a4,r), capsule(p,a4,a6,r));
    const double e62e20 = op_union(capsule(p,a6,a2,r), capsule(p,a2,a0,r));
    const double surf0 = op_union(e04e46, e62e20);

    // surface 1: [1,3,7,5] xi-plus dir
    const double e13e75 = op_union(capsule(p,a1,a3,r), capsule(p,a3,a7,r));
    const double e75e51 = op_union(capsule(p,a7,a5,r), capsule(p,a5,a1,r));
    const double surf1 = op_union(e13e75, e75e51);

    const double xfaces = op_union(surf0, surf1);

    // surface 2: [0,1,5,4]  eta-minus dir
    const double e01e15 = op_union(capsule(p,a0,a1,r), capsule(p,a1,a5,r));
    const double e54e40 = op_union(capsule(p,a5,a4,r), capsule(p,a4,a0,r));
    const double surf2 = op_union(e01e15, e54e40);

    // surface 3: [3,2,6,7]  eta-plus  dir
    const double e32e26 = op_union(capsule(p,a3,a2,r), capsule(p,a2,a6,r));
    const double e67e73 = op_union(capsule(p,a6,a7,r), capsule(p,a7,a3,r));
    const double surf3 = op_union(e32e26, e67e73);

    const double yfaces = op_union(surf2, surf3);

    // surface 4: [0,2,3,1]  zeta-minus dir
    // surface 6: [4,5,7,6]  zeta-plus  dir

    const double allfaces = op_union(xfaces, yfaces);

    // box half height is L/2
    // box center is L/2
    // thickness is r, in this case
    //return op_intersection(allfaces, boxframe(p, vec_t(center.x,center.y,center.z), vec_t(center.x,center.y,center.z), r));
    return op_intersection(box(p, box_dim, center), allfaces);
}


// a0....a7 are the coordinates in the box
KOKKOS_INLINE_FUNCTION
double body_centered_cubic(const vec_t &p,
                           const vec_t &a0,
                           const vec_t &a1,
                           const vec_t &a2,
                           const vec_t &a3,
                           const vec_t &a4,
                           const vec_t &a5,
                           const vec_t &a6,
                           const vec_t &a7,
                           const vec_t &center,
                           const vec_t &box_dim,
                           const double r)
{
    // build the simple cube
    double sc = simple_cube(p,a0,a1,a2,a3,a4,a5,a6,a7,center,box_dim,r);

    // build the diagonal struts using a capsule going from point A to point B (defining a centerline) with radius r

    // go from point 0 to 7
    double strut07 = capsule(p, a0+vec_t(r,r,r), a7+vec_t(-r,-r,-r), r);

    // go from point 1 to 6
    double strut16 = capsule(p, a1+vec_t(-r,r,r), a6+vec_t(r,-r,-r), r);

    // go from point 3 to 4
    double strut34 = capsule(p, a3+vec_t(-r,-r,r), a4+vec_t(r,r,-r), r);

    // go from point 2 to 5
    double strut25 = capsule(p, a2+vec_t(r,-r,r), a5+vec_t(-r,r,-r), r);

    // add all struts together
    double struts = op_union( op_union(strut07, strut16), op_union(strut34, strut25) );

    return op_union(sc, struts);
}



// a0....a7 are the coordinates in the box
KOKKOS_INLINE_FUNCTION
double face_centered_cubic(const vec_t &p,
                           const vec_t &a0,
                           const vec_t &a1,
                           const vec_t &a2,
                           const vec_t &a3,
                           const vec_t &a4,
                           const vec_t &a5,
                           const vec_t &a6,
                           const vec_t &a7,
                           const vec_t &center,
                           const vec_t &box_dim,
                           const double r)
{

    // build the diagonal struts using a capsule going from point A to point B (defining a centerline) with radius r

    // x-face-dir
    // 0 to 5
    double strut05 = capsule(p, a0, a5, r);
    // 1 to 4
    double strut14 = capsule(p, a1, a4, r);
    // 3 to 6
    double strut36 = capsule(p, a3, a6, r);
    // 2 to 7
    double strut27 = capsule(p, a2, a7, r);


    // y-face-dir
    // 1 to 7
    double strut17 = capsule(p, a1, a7, r);
    // 3 to 5
    double strut35 = capsule(p, a3, a5, r);
    // 0 to 6
    double strut06 = capsule(p, a0, a6, r);
    // 2 to 4
    double strut24 = capsule(p, a2, a4, r);

    // z-dir
    // 4 to 7
    double strut47 = capsule(p, a4, a7, r);
    // 5 to 6
    double strut56 = capsule(p, a5, a6, r);
    // 0 to 3
    double strut03 = capsule(p, a0, a3, r);
    // 1 to 2
    double strut12 = capsule(p, a1, a2, r);


    // add all struts together
    double struts_xfaces = op_union( op_union(strut05, strut14), op_union(strut36, strut27) );
    double struts_yfaces = op_union( op_union(strut17, strut35), op_union(strut06, strut24) );
    double struts_zfaces = op_union( op_union(strut47, strut56), op_union(strut03, strut12) );

    double struts = op_union(struts_yfaces, op_union(struts_xfaces, struts_zfaces));

    double struts_with_caps = op_intersection(box(p, box_dim, center),struts);

    return struts_with_caps;
}

double f2ccz(const vec_t &p,
                           const vec_t &a0,
                           const vec_t &a1,
                           const vec_t &a2,
                           const vec_t &a3,
                           const vec_t &a4,
                           const vec_t &a5,
                           const vec_t &a6,
                           const vec_t &a7,
                           const vec_t &center,
                           const vec_t &box_dim,
                           const double r)
{
    // vertical struts
    double strut04 = capsule(p, a0, a4, r);
    double strut15 = capsule(p, a1, a5, r);
    double strut26 = capsule(p, a2, a6, r);
    double strut37 = capsule(p, a3, a7, r);

    // diagonal struts
    double strut05 = capsule(p, a0, a5, r);
    double strut06 = capsule(p, a0, a6, r);

    double strut14 = capsule(p, a1, a4, r);
    double strut17 = capsule(p, a1, a7, r);

    double strut24 = capsule(p, a2, a4, r);
    double strut27 = capsule(p, a2, a7, r);

    double strut35 = capsule(p, a3, a5, r);
    double strut36 = capsule(p, a3, a6, r);

    // add all struts together
    double struts0 = op_union(strut04, op_union(strut05, strut06));
    double struts1 = op_union(strut15, op_union(strut14, strut17));
    double struts2 = op_union(strut26, op_union(strut24, strut27));
    double struts3 = op_union(strut37, op_union(strut35, strut36));
    double struts = op_union(op_union(struts0, struts1), op_union(struts2, struts3));

    return op_intersection(box(p, box_dim, center), struts);
}


// a0....a7 are the coordinates in the box
KOKKOS_INLINE_FUNCTION
double octet(const vec_t &p,
             const vec_t &a0,
             const vec_t &a1,
             const vec_t &a2,
             const vec_t &a3,
             const vec_t &a4,
             const vec_t &a5,
             const vec_t &a6,
             const vec_t &a7,
             const vec_t &center,
             const vec_t &box_dim,
             const double r)
{
    
    // build the interior, diagonal struts
    // using a capsule going from point A to point B (defining a centerline) with radius r

    // go from point 0 to 7
    double strut07 = capsule(p, a0+vec_t(r,r,r), a7+vec_t(-r,-r,-r), r);

    // go from point 1 to 6
    double strut16 = capsule(p, a1+vec_t(-r,r,r), a6+vec_t(r,-r,-r), r);

    // go from point 3 to 4
    double strut34 = capsule(p, a3+vec_t(-r,-r,r), a4+vec_t(r,r,-r), r);

    // go from point 2 to 5
    double strut25 = capsule(p, a2+vec_t(r,-r,r), a5+vec_t(-r,r,-r), r);

    // add all struts together
    double interior_struts = op_union( op_union(strut07, strut16), op_union(strut34, strut25) );

    // build the exterior struts
    double exterior_struts = face_centered_cubic(p,a0,a1,a2,a3,a4,a5,a6,a7,center,box_dim,r);
    
    return op_union(interior_struts, exterior_struts);
}

KOKKOS_INLINE_FUNCTION
double neg_nu(const vec_t &p,
              const double h2, // cell height
              const double a1, // ratio: lower peak height/upper peak height
              const double ah, // base length/upper peak height
              const double r)
{
    // defining the points based on input parameters
    vec_t n0(0.0,0.0,0.0);
    vec_t n1(0.0, ah*h2, 0.0);
    vec_t n2(sqrt(pow(ah*h2,2) - pow(ah*h2/2,2)), ah*h2/2, 0.0);
    vec_t n3(sqrt(pow(ah*h2,2) - pow(ah*h2/2,2))/3, ah*h2/2, h2);
    vec_t n4(sqrt(pow(ah*h2,2) - pow(ah*h2/2,2))/3, ah*h2/2, h2*a1);
    vec_t center(sqrt(pow(ah*h2,2) - pow(ah*h2/2,2))/2, ah*h2/2,h2*a1);

    // unit cell
    double strut04 = capsule(p, n0, n4, r);
    double strut14 = capsule(p, n1, n4, r);
    double strut24 = capsule(p, n2, n4, r);
    double strut03 = capsule(p, n0, n3, r);
    double strut13 = capsule(p, n1, n3, r);
    double strut23 = capsule(p, n2, n3, r);

    double struts0 = op_union(strut04, strut03);
    double struts1 = op_union(strut14, strut13);
    double struts2 = op_union(strut24, strut23);
    double struts3 = op_union(op_union(strut03, strut13), strut23);
    double struts4 = op_union(op_union(strut04, strut14), strut24);

    double struts_unit = op_union(op_union(op_union(struts0,struts1),op_union(struts2,struts3)),struts4);

    // making it periodic
    // top periodicity
    double strut_top0 = capsule(p, n0+vec_t(0.0,0.0,h2-h2*a1), n4+vec_t(0.0,0.0,h2-h2*a1), r);
    double strut_top1 = capsule(p, n1+vec_t(0.0,0.0,h2-h2*a1), n4+vec_t(0.0,0.0,h2-h2*a1), r);
    double strut_top2 = capsule(p, n2+vec_t(0.0,0.0,h2-h2*a1), n4+vec_t(0.0,0.0,h2-h2*a1), r);
    double struts_top = op_union(op_union(strut_top0,strut_top1), strut_top2);

    // bottom periodicity
    double strut_bot0 = capsule(p, n0-vec_t(0.0,0.0,h2-h2*a1), n3-vec_t(0.0,0.0,h2-h2*a1), r);
    double strut_bot1 = capsule(p, n1-vec_t(0.0,0.0,h2-h2*a1), n3-vec_t(0.0,0.0,h2-h2*a1), r);
    double strut_bot2 = capsule(p, n2-vec_t(0.0,0.0,h2-h2*a1), n3-vec_t(0.0,0.0,h2-h2*a1), r);
    double struts_bot = op_union(op_union(strut_bot0,strut_bot1), strut_bot2);
    double struts = op_union(struts_top, op_union(struts_unit, struts_bot));

    return op_intersection(box(p, center, center),struts);
}

KOKKOS_INLINE_FUNCTION
double stack3(const vec_t &p,
              const double box_dims,
              const double r,
              const CArrayKokkos<int> &type)
{
    double factor = (double)type.dims(0);
    vec_t a0(0.0,0.0,0.0);
    vec_t a1(box_dims/factor,0.0,0.0);
    vec_t a2(0.0,box_dims/factor,0.0);
    vec_t a3(box_dims/factor,box_dims/factor,0.0);
    vec_t a4(0.0,0.0,box_dims/factor);
    vec_t a5(box_dims/factor,0.0,box_dims/factor);
    vec_t a6(0.0,box_dims/factor,box_dims/factor);
    vec_t a7(box_dims/factor,box_dims/factor,box_dims/factor);
    vec_t center(box_dims/factor/2,box_dims/factor/2,box_dims/factor/2);
    vec_t translate(0.0, 0.0, 0.0);

    double field = 0;
    double new_section;
    for (int i = 0; i < (int)type.dims(0); i++) {
        translate.z = i*box_dims/factor;
        for (int j = 0; j < (int)type.dims(1); j++) {
            translate.y = j*box_dims/factor;
            for (int k = 0; k < (int)type.dims(2); k++) {
                if (type(i,j,k) > 3) {
                    continue;
                }
                translate.x = k*box_dims/factor;
                switch(type(i,j,k)) {
                    case 0:
                        new_section = simple_cube(p,a0+translate,a1+translate,a2+translate,a3+translate,a4+translate,a5+translate,a6+translate,a7+translate,center+translate,center,r);
                        break;
                    case 1:
                        new_section = body_centered_cubic(p,a0+translate,a1+translate,a2+translate,a3+translate,a4+translate,a5+translate,a6+translate,a7+translate,center+translate,center,r);
                        break;
                    case 2:
                        new_section = face_centered_cubic(p,a0+translate,a1+translate,a2+translate,a3+translate,a4+translate,a5+translate,a6+translate,a7+translate,center+translate,center,r);
                        break;
                    case 3:
                        new_section = octet(p,a0+translate,a1+translate,a2+translate,a3+translate,a4+translate,a5+translate,a6+translate,a7+translate,center+translate,center,r);
                        break;
                }
                field = op_union(field, new_section);
            }
        }
    }

    return field; //op_union(first,op_union(second,third));
}

KOKKOS_INLINE_FUNCTION
double neg_nu_square(const vec_t &p,
              const double h2, // cell height
              const double a1, // ratio: lower peak height/upper peak height
              const double ah, // base length/upper peak height
              const double r)
{
    // defining the points based on input parameters
    vec_t n0(0.0,0.0,0.0);
    vec_t n1(0.0, ah*h2, 0.0);
    vec_t n2(ah*h2, 0.0, 0.0);
    vec_t n3(ah*h2, ah*h2, 0.0);
    vec_t n4(ah*h2/2, ah*h2/2, h2);
    vec_t n5(ah*h2/2, ah*h2/2, h2*a1);
    vec_t center(ah*h2/2, ah*h2/2,h2/2);

    // unit cell
    double strut05 = capsule(p, n0, n5, r);
    double strut15 = capsule(p, n1, n5, r);
    double strut25 = capsule(p, n2, n5, r);
    double strut35 = capsule(p, n3, n5, r);
    double strut04 = capsule(p, n0, n4, r);
    double strut14 = capsule(p, n1, n4, r);
    double strut24 = capsule(p, n2, n4, r);
    double strut34 = capsule(p, n3, n4, r);

    double struts0 = op_union(strut05, strut04);
    double struts1 = op_union(strut15, strut14);
    double struts2 = op_union(strut25, strut24);
    double struts3 = op_union(strut35, strut34);
    double struts4 = op_union(op_union(strut04, strut14), op_union(strut24,strut34));
    double struts5 = op_union(op_union(strut05, strut15), op_union(strut25,strut35));

    double struts_unit = op_union(op_union(op_union(struts0,struts1),op_union(struts2,struts3)),op_union(struts4,struts5));

    // making it periodic
    // top periodicity
    double strut_top0 = capsule(p, n0+vec_t(0.0,0.0,h2-h2*a1), n5+vec_t(0.0,0.0,h2-h2*a1), r);
    double strut_top1 = capsule(p, n1+vec_t(0.0,0.0,h2-h2*a1), n5+vec_t(0.0,0.0,h2-h2*a1), r);
    double strut_top2 = capsule(p, n2+vec_t(0.0,0.0,h2-h2*a1), n5+vec_t(0.0,0.0,h2-h2*a1), r);
    double strut_top3 = capsule(p, n3+vec_t(0.0,0.0,h2-h2*a1), n5+vec_t(0.0,0.0,h2-h2*a1), r);
    double struts_top = op_union(op_union(strut_top0,strut_top1), op_union(strut_top2, strut_top3));

    // bottom periodicity
    double strut_bot0 = capsule(p, n0-vec_t(0.0,0.0,h2-h2*a1), n4-vec_t(0.0,0.0,h2-h2*a1), r);
    double strut_bot1 = capsule(p, n1-vec_t(0.0,0.0,h2-h2*a1), n4-vec_t(0.0,0.0,h2-h2*a1), r);
    double strut_bot2 = capsule(p, n2-vec_t(0.0,0.0,h2-h2*a1), n4-vec_t(0.0,0.0,h2-h2*a1), r);
    double strut_bot3 = capsule(p, n3-vec_t(0.0,0.0,h2-h2*a1), n4-vec_t(0.0,0.0,h2-h2*a1), r);
    double struts_bot = op_union(op_union(strut_bot0,strut_bot1), op_union(strut_bot2, strut_bot3));
    
    double struts = op_union(struts_top, op_union(struts_unit, struts_bot));

    return op_intersection(box(p, center, center),struts);
}



// main code
int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);
    {  
        // define mesh spacing, it is used to create a mesh
        
        // sc 0; bcc 1; fcc 2; octet 3
        CArrayKokkos <int> type(3,3,3);
        RUN({
            type(0,0,0) = 4;
            type(0,0,1) = 4;
            type(0,0,2) = 4;

            type(0,1,0) = 4;
            type(0,1,1) = 2;
            type(0,1,2) = 4;
            
            type(0,2,0) = 4;
            type(0,2,1) = 4;
            type(0,2,2) = 4;
            
            type(1,0,0) = 4;
            type(1,0,1) = 2;
            type(1,0,2) = 4;
            
            type(1,1,0) = 2;
            type(1,1,1) = 2;
            type(1,1,2) = 2;
            
            type(1,2,0) = 4;
            type(1,2,1) = 2;
            type(1,2,2) = 4;
            
            type(2,0,0) = 4;
            type(2,0,1) = 4;
            type(2,0,2) = 4;
            
            type(2,1,0) = 4;
            type(2,1,1) = 2;
            type(2,1,2) = 4;
            
            type(2,2,0) = 4;
            type(2,2,1) = 4;
            type(2,2,2) = 4;
        });

        // mesh coordinates
        DCArrayKokkos <double> x(num_pt_x, "pt_x");
        DCArrayKokkos <double> y(num_pt_y, "pt_y");
        DCArrayKokkos <double> z(num_pt_z, "pt_z");

        
        // function with isosurface that we want extracted
        DCArrayKokkos <double> levelset(num_pt_x,num_pt_y,num_pt_z, "grid_values");


        // ------------------------------------
        // Create a mesh
        // ------------------------------------

        printf("Creating structured mesh \n");

        FOR_ALL(i, 0, num_pt_x, {
            x(i) = dx*(double)i + X0;
        });
        FOR_ALL(j, 0, num_pt_y, {
            y(j) = dy*(double)j + Y0;
        });
        FOR_ALL(k, 0, num_pt_z, {
            z(k) = dz*(double)k + Z0;
        });
        Kokkos::fence();
        x.update_host();
        y.update_host();
        z.update_host();


        printf("Evaluating level set field on mesh \n");

        FOR_ALL(k, 0, num_pt_z, 
                j, 0, num_pt_y,
                i, 0, num_pt_x, {

                    //const double dx = x(i) - 0.5;
                    //const double dy = y(j) - 0.5;
                    //const double dz = z(k) - 0.5;
                    //const vec_t p(dx,dy,dz);
                    //levelset(i,j,k) =  sphere(p,0.25); 

                    const vec_t p(x(i),y(j),z(k));
                    

                    // simple cube

                    // coords of box corners
                    vec_t a0(0.0,0.0,0.0);
                    vec_t a1(box_dims,0.0,0.0);
                    vec_t a2(0.0,box_dims,0.0);
                    vec_t a3(box_dims,box_dims,0.0);
                    vec_t a4(0.0,0.0,box_dims);
                    vec_t a5(box_dims,0.0,box_dims);
                    vec_t a6(0.0,box_dims,box_dims);
                    vec_t a7(box_dims,box_dims,box_dims);

                    if(build_lattice_type==0){
                        levelset(i,j,k) = simple_cube(p,a0,a1,a2,a3,a4,a5,a6,a7,vec_t(box_dims/2.,box_dims/2.,box_dims/2.),vec_t(box_dims/2.,box_dims/2.,box_dims/2.),r);
                    }
                    else if (build_lattice_type==1){ 
                        levelset(i,j,k) = body_centered_cubic(p,a0,a1,a2,a3,a4,a5,a6,a7,vec_t(box_dims/2.,box_dims/2.,box_dims/2.),vec_t(box_dims/2.,box_dims/2.,box_dims/2.),r);
                    }
                    else if (build_lattice_type==2){
                        levelset(i,j,k) = face_centered_cubic(p,a0,a1,a2,a3,a4,a5,a6,a7,vec_t(box_dims/2.,box_dims/2.,box_dims/2.),vec_t(box_dims/2.,box_dims/2.,box_dims/2.),r);
                    }
                    else if (build_lattice_type==3){
                        levelset(i,j,k) = octet(p,a0,a1,a2,a3,a4,a5,a6,a7,vec_t(box_dims/2.,box_dims/2.,box_dims/2.),vec_t(box_dims/2.,box_dims/2.,box_dims/2.),r);
                    }   
                    else {
                        // testing objects
                        //levelset(i,j,k) = box(p, vec_t(box_dims/2.,box_dims/2.,box_dims/2.), vec_t(box_dims/2.,box_dims/2.,box_dims/2.));
                        //levelset(i,j,k) = boxframe(p, vec_t(0.5,0.5,0.5), vec_t(0.5,0.5,0.5), r);

                        //levelset(i,j,k) = neg_nu(p, h2, alpha1, alphah, r);
                        levelset(i,j,k) = stack3(p, box_dims, r, type);
                        //levelset(i,j,k) = neg_nu_square(p, h2, alpha1, alphah, r);
                        //levelset(i,j,k) = hollow_capsule(p, a0, a7, r, r/2);
                        //levelset(i,j,k) = f2ccz(p,a0,a1,a2,a3,a4,a5,a6,a7,vec_t(box_dims/2.,box_dims/2.,box_dims/2.),vec_t(box_dims/2.,box_dims/2.,box_dims/2.),r);
                        
                        //const vec_t a2(0.0,0.0,0.0);
                        //const vec_t b2(1.0,0.0,0.0);
                        //double strut2 = op_intersection(capsule(p,a2,b2,r), box(p,vec_t(1.0, r, r),vec_t(0.5,r,r)) );
                        //double strut2 = boxframe(p, vec_t(0.5,0.5,0.5), vec_t(0.5,0.5,0.5), r);
                        //double strut2 = 
                        //op_intersection(capsule(p,a2,b2,r), boxframe(p, vec_t(0.5,0.5,0.5), vec_t(0.5,0.5,0.5), r));
                        //levelset(i,j,k) = op_union(strut1, strut2);
                    }



        
        }); // end parallel over k,j,i
        Kokkos::fence();
        levelset.update_host();


        // call marching cubes, which builds surface and exports it as an STL file
        marching_cubes(x, y, z, levelset, isoLevel);



        printf("Writing VTK Graphics File \n\n");

        std::ofstream out("levelset.vtk");

        size_t num_points = num_pt_x*num_pt_y*num_pt_z;

        out << "# vtk DataFile Version 3.0\n";
        out << "3D levelset\n";
        out << "ASCII\n";
        out << "DATASET STRUCTURED_GRID\n";
        out << "DIMENSIONS " << num_pt_x << " " << num_pt_y << " " << num_pt_z << "\n";
        out << "POINTS " << num_points << " float\n";
        for (size_t k = 0; k < num_pt_z; ++k) {
            for (size_t j = 0; j < num_pt_y; ++j) {
                for (size_t i = 0; i < num_pt_x; ++i) {
                    out << x.host(i) << " " 
                        << y.host(j) << " " 
                        << z.host(k) << "\n";
                } // end i
            } // end j
        } // end k

        out << "\nPOINT_DATA " << num_points << "\n";
        out << "SCALARS levelset float 1\n";
        out << "LOOKUP_TABLE default\n";
        for (size_t k = 0; k < num_pt_z; ++k) {
            for (size_t j = 0; j < num_pt_y; ++j) {
                for (size_t i = 0; i < num_pt_x; ++i) {
                    out << levelset.host(i,j,k) << "\n";
                } // end i
            } // end j
        } // end k

        out.close();
        std::cout << "Wrote levelset.vtk successfully.\n";


    } // end of kokkos scope


    Kokkos::finalize();

    return 0;
    
} // end main