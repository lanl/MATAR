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
// pointcloud reproducing kernels in C++
//  Nathaniel Morgan
// -----------------------------------------------

#include <fstream>
#include <iostream>
#include <string>
#include <stdio.h>
#include <cmath>

#include <cstdlib> // For rand() and srand()


#include "matar.h"

#include "lu_solver.hpp"


using namespace mtr;

const double PI = 3.14159265358979323846;

// -----------------------------------------------
// inputs:

const size_t num_points = 101;

// the bin sizes for finding neighboring points
const double bin_dx = 0.5; // 2 bins in x
const double bin_dy = 0.5; // 2 bins in y
const double bin_dz = 0.5; // 2 bins in z

const double X0 = 0.0;   // origin
const double Y0 = 0.0;
const double Z0 = 0.0;

// length of the domain 
const double LX = 1.0;   // length in x-dir
const double LY = 1.0;
const double LZ = 1.0;

//
// -----------------------------------------------




struct bin_ijk_t{
    size_t i, j, k;
};


bin_ijk_t get_bin_ijk(const double x_pt, const double y_pt, const double z_pt){
            
    bin_ijk_t bin_ijk;

    double i_dbl = fmax(1.0e-15, round((x_pt - X0 - bin_dx*0.5)/bin_dx - 1.0e-10)); // x = ih + X0 + dx_bin*0.5
    double j_dbl = fmax(1.0e-15, round((y_pt - Y0 - bin_dy*0.5)/bin_dy - 1.0e-10));
    double k_dbl = fmax(1.0e-15, round((z_pt - Z0 - bin_dz*0.5)/bin_dz - 1.0e-10));

    // get the integers for the bins
    bin_ijk.i = (size_t)i_dbl;
    bin_ijk.j = (size_t)j_dbl;
    bin_ijk.k = (size_t)k_dbl;
    
    return bin_ijk;
} // end function

// Gaussian function part of the RBF
// rbf = exp(-(x - xj)*(x - xj)/h)
KOKKOS_FUNCTION
double kernel(const double r[3], const double h){

    double diff_sqrd = 0.0;

    for(size_t dim=0; dim<3; dim++){
        diff_sqrd += r[dim]*r[dim];
    } // dim

    return exp(-diff_sqrd/(h*h));
} // end of function


// Polynomial basis up to quadratic in 3D (10 terms)
const size_t num_poly_basis = 10;
KOKKOS_INLINE_FUNCTION
void poly_basis(const double r[3], double *p) {

    p[0] = 1.0;
    p[1] = r[0];
    p[2] = r[1];
    p[3] = r[2];
    p[4] = r[0] * r[0];
    p[5] = r[0] * r[1];
    p[6] = r[0] * r[2];
    p[7] = r[1] * r[1];
    p[8] = r[1] * r[2];
    p[9] = r[2] * r[2];

    // for high-order will use (x^a y^b z^c)

    return;
} // end function


void compute_shape_functions(
    size_t i,
    const DCArrayKokkos <double>& x,
    const CArrayKokkos <double>& vol,
    const CArrayKokkos <double>& rk_coeffs,
    const CArrayKokkos <double>& rk_basis,
    const double h)
{

    // global num_points at this time, make it num_points in neighborhood
    size_t num_points_neighborhood = x.dims(0); // will come from hash bins

    // loop over all neighbors around point i
    FOR_ALL(j, 0, num_points_neighborhood, {

        double p[num_poly_basis];    // array holding polynomial basis [x, y, z, x^2, y^2, ... , yz]
        double r[3];    // vecx_j - vecx_i
        r[0] = x(j,0) - x(i,0); // x_j-x_i
        r[1] = x(j,1) - x(i,1); // y_j-y_i
        r[2] = x(j,2) - x(i,2); // z_j-z_i

        double W = kernel(r, h);
        poly_basis(r,p);

        double correction = 0.0;
        for (size_t a = 0; a < num_poly_basis; ++a){
            correction += rk_coeffs(i,a) * p[a];
        } // end for a

        rk_basis(j) = W * correction;
    });


    return;
} // end function



// Build reproducing kernel coefficients for all particles in the domain
void build_rk_coefficients(
    const DCArrayKokkos <double>& x,
    const CArrayKokkos <double>& vol,
    const CArrayKokkos <double>& rk_coeffs,
    double h)
{

    // global num_points at this time, make it num_points in neighborhood
    size_t num_points_neighborhood = x.dims(0); // will come from hash bins

    // actual number of points
    size_t num_points = x.dims(0);

    
    // loop over all nodes in the problem
    FOR_ALL(i, 0, num_points, {

        double M_1D[num_poly_basis*num_poly_basis]; 
        ViewCArrayKokkos <double> M(&M_1D[0], num_poly_basis, num_poly_basis);
        M.set_values(0.0);

        // values in rhs after this function will be accessed as rk_coeffs(i,0:N)
        ViewCArrayKokkos <double> rhs (&rk_coeffs(i,0), num_poly_basis);
        rhs.set_values(0.0);
        rhs(0) = 1.0;   // enforce reproduction of constant 1, everything else is = 0

        double p[num_poly_basis];    // array holding polynomial basis [x, y, z, x^2, y^2, ... , yz]
        double r[3];    // vecx_j - vecx_i

        // loop over all nodes around point i
        for (size_t j = 0; j < num_points_neighborhood; ++j) {
           
            r[0] = x(j,0) - x(i,0); // x_j-x_i
            r[1] = x(j,1) - x(i,1); // y_j-y_i
            r[2] = x(j,2) - x(i,2); // z_j-z_i

            double W = kernel(r, h);
            poly_basis(r,p);

            // assemble matrix

            for (size_t a = 0; a < num_poly_basis; ++a) {
                for (size_t b = 0; b < num_poly_basis; ++b) {
                    M(a,b) += vol(j) * W * p[a] * p[b]; 
                } // end for b
            } // for a

        } // end for point neighbors j
    
        // -------------
        // solve Ax=B
        // -------------

        size_t perm_1D[num_poly_basis];
        ViewCArrayKokkos <size_t> perm (&perm_1D[0], num_poly_basis);
        for (size_t a = 0; a < num_poly_basis; ++a) {
            perm(a)= 0;
        } // end a

        double vv_1D[num_poly_basis];
        ViewCArrayKokkos <double> vv(&vv_1D[0], num_poly_basis);
        
        // used for LU problem
        int singular = 0;
        int parity = 0;
        singular = LU_decompose(M, perm, vv, parity);  // M is returned as the LU matrix  
        if(singular==0){
            printf("ERROR: matrix is singluar \n");
        }

        LU_backsub(M, perm, rhs);  // note: answer is sent back in rhs

    }); // end parallel loop


    return; 
} // end function





int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);
    {  

        printf("Pointcloud Reproducing Kernels \n\n");


        // define a point cloud
        DCArrayKokkos <double> point_positions(num_points, 3, "point_positions");
        DCArrayKokkos <double> point_values(num_points, "point_values"); 

        // point locations
        srand(static_cast<unsigned int>(time(0))); // Seed the random number generator
        for(size_t i=0; i<num_points; i++){
            point_positions.host(i, 0) = X0 + LX*static_cast<double>(rand())/static_cast<double>(RAND_MAX);
            point_positions.host(i, 1) = Y0 + LY*static_cast<double>(rand())/static_cast<double>(RAND_MAX);
            point_positions.host(i, 2) = Z0 + LZ*static_cast<double>(rand())/static_cast<double>(RAND_MAX);
        }
        point_positions.update_device();
        Kokkos::fence();

        // point values
        FOR_ALL(i, 0, num_points, {

            printf("point location at i=%d is (%f, %f, %f) \n", i, point_positions(i, 0), point_positions(i, 1), point_positions(i, 2));
            point_values(i) = sqrt(point_positions(i, 0)*point_positions(i, 0) + 
                                   point_positions(i, 1)*point_positions(i, 1) +
                                   point_positions(i, 2)*point_positions(i, 2));

        }); // end parallel for tri's in the file
        printf("\n");


        // ----------------------------
        // Make bins here
        // ----------------------------
        
        // the number of nodes in the mesh
        size_t num_bins_x = (size_t)( round(LX/bin_dx) );  
        size_t num_bins_y = (size_t)( round(LY/bin_dy) );  
        size_t num_bins_z = (size_t)( round(LZ/bin_dz) );  

        

        size_t num_bins = num_bins_x*num_bins_y*num_bins_z;

        //printf("num bins = %zu \n", num_bins);

        DCArrayKokkos <size_t> num_points_in_bin(num_bins);
        num_points_in_bin.set_values(0);
        DCArrayKokkos <size_t> points_bin_id(num_points);
        DCArrayKokkos <size_t> points_bin_id_storage(num_points);
        
        FOR_ALL(pt_id, 0, num_points, {

            // get i,j,k indices of the bins
            bin_ijk_t bin_ijk = get_bin_ijk(point_positions(pt_id,0), 
                                            point_positions(pt_id,1), 
                                            point_positions(pt_id,2));

            // get the 1D index
            size_t bin_id = bin_ijk.i + (bin_ijk.j + bin_ijk.k*num_bins_y)*num_bins_x;
          
            size_t storage_place = Kokkos::atomic_fetch_add(&num_points_in_bin(bin_id), 1);
            points_bin_id(pt_id) = bin_id; // the id of the bin
            points_bin_id_storage(pt_id) = storage_place; // the storage place in the bin

        }); // end for all

        DRaggedRightArrayKokkos <size_t> points_in_bin(num_points_in_bin);

        FOR_ALL(pt_id, 0, num_points, {

            size_t bin_id = points_bin_id(pt_id);
            size_t storage_place = points_bin_id_storage(pt_id);
            points_in_bin(bin_id, storage_place) = pt_id;

        }); // end for all
        

        // ----------------------------
        // Reconstruct surface here
        // ----------------------------

        printf("Reconstructing surface using point cloud data \n\n");

        // assuming all point neighbors contribute, will change to a hash bins
        const size_t num_points_neighborhood = num_points;

        CArrayKokkos <double> rk_coeffs(num_points, num_poly_basis);  // reproducing kernel coefficients at each point
        CArrayKokkos <double> rk_basis(num_points);       // reproducing kernel basis, should have size num_points_neighborhood
        CArrayKokkos <double> vol(num_points);
        vol.set_values(1.0);

        double h = 1.0;


        printf("building rk coefficients \n");

        // build coefficients on basis functions
        build_rk_coefficients(point_positions, vol, rk_coeffs, h);

        
        
        // performing checks on rk_coeffs
        double partion_unity;
        double partion_unity_lcl;

        double linear_preserving;
        double linear_preserving_lcl;

        for(size_t i=0; i<num_points; i++){
            
            // build basis functions at point i
            compute_shape_functions(i, 
                                    point_positions, 
                                    vol, 
                                    rk_coeffs, 
                                    rk_basis, 
                                    h);

            FOR_REDUCE_SUM(j, 0, num_points_neighborhood, partion_unity_lcl, {
                partion_unity_lcl += rk_basis(j)*vol(j);
            }, partion_unity);
            

            FOR_REDUCE_SUM(j, 0, num_points_neighborhood, linear_preserving_lcl, {
                linear_preserving_lcl += rk_basis(j)*vol(j)*point_positions(j,0);
            }, linear_preserving);

            printf("partition unity = %f,  ", partion_unity);
            printf("linear preserving error = %f at i=%zu \n", fabs(linear_preserving-point_positions(i,0)), i);

        } // end for i


    
        printf("Finished \n\n");


    } // end of kokkos scope



    Kokkos::finalize();

    return 0;
    
} // end main
