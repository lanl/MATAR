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

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

using namespace mtr;

const double PI = 3.14159265358979323846;

// -----------------------------------------------
// inputs:

const size_t num_points = 101;

// the bin sizes for finding neighboring points
const double bin_dx = 0.05; // bins in x
const double bin_dy = 0.05; // bins in y
const double bin_dz = 0.05; // bins in z

const double X0 = 0.0;   // origin
const double Y0 = 0.0;
const double Z0 = 0.0;

// length of the domain 
const double LX = 1.0;   // length in x-dir
const double LY = 1.0;
const double LZ = 1.0;

//
// -----------------------------------------------



struct bin_keys_t{
    size_t i,j,k;
};

KOKKOS_INLINE_FUNCTION
size_t get_gid(size_t i, size_t j, size_t k, size_t num_x, size_t num_y){
    return i + (j + k*num_y)*num_x;
}

KOKKOS_INLINE_FUNCTION
bin_keys_t get_bin_keys(const double x_pt, 
                        const double y_pt, 
                        const double z_pt){
            

    double i_dbl = fmax(1.0e-15, round((x_pt - X0 - bin_dx*0.5)/bin_dx - 1.0e-10)); // x = ih + X0 + dx_bin*0.5
    double j_dbl = fmax(1.0e-15, round((y_pt - Y0 - bin_dy*0.5)/bin_dy - 1.0e-10));
    double k_dbl = fmax(1.0e-15, round((z_pt - Z0 - bin_dz*0.5)/bin_dz - 1.0e-10));

    bin_keys_t bin_keys; // save i,j,k to the bin keys

    // get the integer for the bins
    bin_keys.i = (size_t)i_dbl;
    bin_keys.j = (size_t)j_dbl;
    bin_keys.k = (size_t)k_dbl;

    return bin_keys;

} // end function

KOKKOS_INLINE_FUNCTION
size_t get_bin_gid(const double x_pt, 
                   const double y_pt, 
                   const double z_pt, 
                   const size_t num_bins_x,
                   const size_t num_bins_y){
            

    double i_dbl = fmax(1.0e-15, round((x_pt - X0 - bin_dx*0.5)/bin_dx - 1.0e-10)); // x = ih + X0 + dx_bin*0.5
    double j_dbl = fmax(1.0e-15, round((y_pt - Y0 - bin_dy*0.5)/bin_dy - 1.0e-10));
    double k_dbl = fmax(1.0e-15, round((z_pt - Z0 - bin_dz*0.5)/bin_dz - 1.0e-10));

    // get the integers for the bins
    size_t i = (size_t)i_dbl;
    size_t j = (size_t)j_dbl;
    size_t k = (size_t)k_dbl;
    
    // get the 1D index for this bin                               
    return get_gid(i, j, k, num_bins_x, num_bins_y);

} // end function


// Gaussian function part of the RBF
// rbf = exp(-(xj - x)*(xj - x)/h)
KOKKOS_FUNCTION
double kernel(const double r[3], const double h){

    double diff_sqrd = 0.0;

    for(size_t dim=0; dim<3; dim++){
        diff_sqrd += r[dim]*r[dim];
    } // dim

    return exp(-diff_sqrd/(h*h));
} // end of function


// Gradient Gaussian function
// rbf = exp(-(xj - x)*(xj - x)/h)
KOKKOS_FUNCTION
void grad_kernel(const double r[3], const double h, double *grad_W){

    double diff_sqrd = 0.0;

    for(size_t dim=0; dim<3; dim++){
        diff_sqrd += r[dim]*r[dim];
    } // dim

    const double rbf = exp(-diff_sqrd/(h*h));

    // gradient
    grad_W[0] = 2.0/h*r[0]*rbf; 
    grad_W[1] = 2.0/h*r[1]*rbf; 
    grad_W[2] = 2.0/h*r[2]*rbf;

    return;
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


KOKKOS_INLINE_FUNCTION
void grad_poly_basis(const double r[3], double (*grad_p)[num_poly_basis]) {
    
    const double drdx = -1.0;

    grad_p[0][0] = 0.0;
    grad_p[0][1] = drdx;
    grad_p[0][2] = 0.0;
    grad_p[0][3] = 0.0;
    grad_p[0][4] = 2.0*r[0]*drdx;
    grad_p[0][5] = r[1]*drdx;
    grad_p[0][6] = r[2]*drdx;
    grad_p[0][7] = 0.0;
    grad_p[0][8] = 0.0;
    grad_p[0][9] = 0.0;

    // for high-order will use (x^a y^b z^c)

    const double drdy = -1.0;

    grad_p[1][0] = 0.0;
    grad_p[1][1] = 0.0;
    grad_p[1][2] = drdy;
    grad_p[1][3] = 0.0;
    grad_p[1][4] = 0.0;
    grad_p[1][5] = r[0]*drdy;
    grad_p[1][6] = 0.0;
    grad_p[1][7] = 2.0*r[1]*drdy;
    grad_p[1][8] = r[2]*drdy;
    grad_p[1][9] = 0.0;

    // for high-order will use (x^a y^b z^c)

    const double drdz = -1.0;

    grad_p[2][0] = 0.0;
    grad_p[2][1] = 0.0;
    grad_p[2][2] = 0.0;
    grad_p[2][3] = drdz;
    grad_p[2][4] = 0.0;
    grad_p[2][5] = 0.0;
    grad_p[2][6] = r[0]*drdz;
    grad_p[2][7] = 0.0;
    grad_p[2][8] = r[1]*drdz;
    grad_p[2][9] = 2.0*r[2]*drdz;

    // for high-order will use (x^a y^b z^c)

    return;
} // end function


void calc_shape_functions(
    size_t point_gid,
    const DCArrayKokkos <double>& x,
    const DCArrayKokkos <size_t> points_num_neighbors, 
    const DRaggedRightArrayKokkos <size_t> points_in_point,
    const CArrayKokkos <double>& vol,
    const CArrayKokkos <double>& p_coeffs,
    const DRaggedRightArrayKokkos <double>& basis,
    const double h)
{

    //---------------------------------------------
    // walk over the neighboring points 
    //---------------------------------------------

    FOR_ALL(neighbor_point_lid, 0, points_num_neighbors(point_gid), {

        size_t neighbor_point_gid = points_in_point(point_gid, neighbor_point_lid);

        double p[num_poly_basis];    // array holding polynomial basis [x, y, z, x^2, y^2, ... , yz]
        double r[3];    // vecx_j - vecx_i
        r[0] = x(neighbor_point_gid,0) - x(point_gid,0); // x_j-x_i
        r[1] = x(neighbor_point_gid,1) - x(point_gid,1); // y_j-y_i
        r[2] = x(neighbor_point_gid,2) - x(point_gid,2); // z_j-z_i

        double W = kernel(r, h);
        poly_basis(r,p);

        double correction = 0.0;
        for (size_t a = 0; a < num_poly_basis; ++a){
            correction += p_coeffs(point_gid,a) * p[a];
        } // end for a

        basis(point_gid, neighbor_point_lid) = W * correction;

    }); // neighbor_point_lid

    return;
    
} // end function



void calc_grad_shape_functions(
    size_t point_gid,
    const DCArrayKokkos <double>& x,
    const DCArrayKokkos <size_t> points_num_neighbors, 
    const DRaggedRightArrayKokkos <size_t> points_in_point,
    const CArrayKokkos <double>& vol,
    const CArrayKokkos <double>& p_coeffs,
    const DRaggedRightArrayKokkos <double>& basis,
    const DRaggedRightArrayKokkos <double>& grad_basis,
    const double h)
{

    //---------------------------------------------
    // walk over the neighboring points 
    //---------------------------------------------

    FOR_ALL(neighbor_point_lid, 0, points_num_neighbors(point_gid), {

        size_t neighbor_point_gid = points_in_point(point_gid, neighbor_point_lid);

        double p[num_poly_basis];    // array holding polynomial basis [x, y, z, x^2, y^2, ... , yz]
        double grad_p[3][num_poly_basis]; // matrix holding grad polynomial basis

        double r[3];    // vecx_j - vecx_i
        r[0] = x(neighbor_point_gid,0) - x(point_gid,0); // x_j-x_i
        r[1] = x(neighbor_point_gid,1) - x(point_gid,1); // y_j-y_i
        r[2] = x(neighbor_point_gid,2) - x(point_gid,2); // z_j-z_i

        double W = kernel(r, h);
        poly_basis(r,p);

        double correction = 0.0;
        for (size_t a = 0; a < num_poly_basis; ++a){
            correction += p_coeffs(point_gid,a) * p[a];
        } // end for a

        basis(point_gid, neighbor_point_lid) = W * correction;

        // --- gradient ---
        double grad_W[3];
        grad_kernel(r, h, grad_W);
        grad_poly_basis(r, grad_p);

        double term1_x = 0.0;
        double term1_y = 0.0;
        double term1_z = 0.0;

        double term2_x = 0.0;
        double term2_y = 0.0;
        double term2_z = 0.0;

        for (size_t a = 0; a < num_poly_basis; ++a){
            term1_x += grad_p[0][a] * p_coeffs(point_gid,a);
            term1_y += grad_p[1][a] * p_coeffs(point_gid,a);
            term1_z += grad_p[2][a] * p_coeffs(point_gid,a);
        } // end for a
        term1_x *= W;
        term1_y *= W;
        term1_z *= W;

        term2_x = correction*grad_W[0];
        term2_y = correction*grad_W[1];
        term2_z = correction*grad_W[2];

    }); // neighbor_point_lid

    return;
    
} // end function



// Build reproducing kernel poly coefficients for all particles in the domain
void calc_p_coefficients(
    const DCArrayKokkos <double>& x,
    const DCArrayKokkos <size_t> points_num_neighbors, 
    const DRaggedRightArrayKokkos <size_t> points_in_point,
    const CArrayKokkos <double>& vol,
    const CArrayKokkos <double>& p_coeffs,
    const CArrayKokkos <double>& M_inv,
    double h)
{

    // actual number of points
    size_t num_points = x.dims(0);

    
    // loop over all nodes in the problem
    FOR_ALL(point_gid, 0, num_points, {

        double M_1D[num_poly_basis*num_poly_basis]; 
        ViewCArrayKokkos <double> M(&M_1D[0], num_poly_basis, num_poly_basis);
        M.set_values(0.0);

        // values in rhs after this function will be accessed as p_coeffs(i,0:N)
        ViewCArrayKokkos <double> rhs (&p_coeffs(point_gid,0), num_poly_basis);
        rhs.set_values(0.0);
        rhs(0) = 1.0;   // enforce reproduction of constant 1, everything else is = 0

        double p[num_poly_basis];    // array holding polynomial basis [x, y, z, x^2, y^2, ... , yz]
        double r[3];    // vecx_j - vecx_i

        //---------------------------------------------
        // walk over the neighboring points
        //---------------------------------------------

        for (size_t neighbor_point_lid=0; neighbor_point_lid<points_num_neighbors(point_gid); neighbor_point_lid++){

            size_t neighbor_point_gid = points_in_point(point_gid, neighbor_point_lid);

            r[0] = x(neighbor_point_gid,0) - x(point_gid,0); // x_j-x_i
            r[1] = x(neighbor_point_gid,1) - x(point_gid,1); // y_j-y_i
            r[2] = x(neighbor_point_gid,2) - x(point_gid,2); // z_j-z_i

            double W = kernel(r, h);
            poly_basis(r,p);

            // assemble matrix

            for (size_t a = 0; a < num_poly_basis; ++a) {
                for (size_t b = 0; b < num_poly_basis; ++b) {
                    M(a,b) += vol(neighbor_point_gid) * W * p[a] * p[b]; 
                } // end for b
            } // for a

        } // neighbor_point_lid

    
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


        // --------------------------------------------------
        // things needed for gradient of the basis function
        double col_1D[num_poly_basis];
        ViewCArrayKokkos <double> col(&col_1D[0], num_poly_basis);

        // making a view, inverting only the matrix at point i
        ViewCArrayKokkos <double> M_inv_pt(&M_inv(point_gid,0,0), num_poly_basis,num_poly_basis);

        LU_invert(M,        // input matrix
                  perm,     // permutations
                  M_inv_pt, // inverse matrix at point gid
                  col);     // tmp array
        // -------------------------------------------------
        
        // solve for p_coefs
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
        point_values.update_host();
        Kokkos::fence();
        printf("\n");


        // ----------------------------
        // Make bins here
        // ----------------------------
        
        // the number of nodes in the mesh
        size_t num_bins_x = (size_t)( round(LX/bin_dx) );  
        size_t num_bins_y = (size_t)( round(LY/bin_dy) );  
        size_t num_bins_z = (size_t)( round(LZ/bin_dz) );  
        size_t num_bins = num_bins_x*num_bins_y*num_bins_z;

        // bins and their connectivity to each other and points
        DCArrayKokkos <bin_keys_t> keys_in_bin(num_bins, "keys_in_bin"); // mapping from gid to (i,j,k)
        DCArrayKokkos <size_t> num_points_in_bin(num_bins, "num_bins");
        num_points_in_bin.set_values(0);
        DRaggedRightArrayKokkos <size_t> points_in_bin; // allocated later
        

        // connectivity from points to bins
        DCArrayKokkos <size_t> points_bin_gid(num_points, "points_in_gid");
        CArrayKokkos <size_t>  points_bin_lid_storage(num_points, "bin_lid_storage");  // only used to create storage
        DCArrayKokkos <size_t> points_bin_stencil(num_points, "bin_stencil"); // how many bins needed for a particle
        DCArrayKokkos <size_t> points_num_neighbors(num_points, "num_neighbors");
        
        // build reverse mapping between gid and i,j,k
        FOR_ALL(i, 0, num_bins_x,
                j, 0, num_bins_y,
                k, 0, num_bins_z, {
            

            // get bin gid for this i,j,k
            size_t bin_gid = get_gid(i, j, k, num_bins_x, num_bins_y);

            // the i,j,k for this bin
            bin_keys_t bin_keys;
            bin_keys.i = i;
            bin_keys.j = j;
            bin_keys.k = k;

            // save mapping from bin_gid to bin_keys i,j,k
            keys_in_bin(bin_gid) = bin_keys;

        });
        Kokkos::fence();
        keys_in_bin.update_host();


        // -------------------------------------------------------------------
        // below here, these routine must be called every time particles move
        // -------------------------------------------------------------------

        // save bin id to points
        FOR_ALL(point_gid, 0, num_points, {

            // get the 1D index for this bin
            size_t bin_gid = get_bin_gid(point_positions(point_gid,0), 
                                         point_positions(point_gid,1), 
                                         point_positions(point_gid,2),
                                         num_bins_x, 
                                         num_bins_y);

            size_t storage_lid = Kokkos::atomic_fetch_add(&num_points_in_bin(bin_gid), 1);
            points_bin_gid(point_gid) = bin_gid; // the id of the bin
            points_bin_lid_storage(point_gid) = storage_lid; // the storage place in the bin

        }); // end for all
        Kokkos::fence();
        points_bin_gid.update_host();
        num_points_in_bin.update_host();


        // allocate points in bin connectivity
        points_in_bin = DRaggedRightArrayKokkos <size_t> (num_points_in_bin, "num_points_in_bin");

        // save points in bin
        FOR_ALL(point_gid, 0, num_points, {

            // get bin gid
            size_t bin_gid = points_bin_gid(point_gid);

            // get it's storage location in the ragged right compressed storage
            size_t storage_lid = points_bin_lid_storage(point_gid);

            // save the point to this bin
            points_in_bin(bin_gid, storage_lid) = point_gid;

        }); // end for all



        // ------------------------------------------------
        // Find the neighbors around each point using bins
        // ------------------------------------------------
        
        FOR_ALL(point_gid, 0, num_points, {

            // get bin gid
            size_t bin_gid = points_bin_gid(point_gid);
            
            // get i,j,k for this bin
            bin_keys_t bin_keys = keys_in_bin(bin_gid);
            // printf(" keys = %zu, %zu, %zu, bin size = %zu, %zu, %zu \n", 
            //     bin_keys.i, bin_keys.j, bin_keys.k,
            //     num_bins_x, num_bins_y, num_bins_z);

            // loop over neighboring bins
            size_t num_points_found;

            // establish the stencil size to get enough particles
            for(int stencil=1; stencil<1000; stencil++){

                num_points_found = 0;

                const int i = bin_keys.i;
                const int j = bin_keys.j;
                const int k = bin_keys.k;

                const int imin = MAX(0, i-stencil);
                const int imax = MIN(num_bins_x-1, i+stencil);

                const int jmin = MAX(0, j-stencil);
                const int jmax = MIN(num_bins_y-1, j+stencil);

                const int kmin = MAX(0, k-stencil);
                const int kmax = MIN(num_bins_z-1, k+stencil);

                for (int icount=imin; icount<=imax; icount++){
                    for (int jcount=jmin; jcount<=jmax; jcount++) {
                        for (int kcount=kmin; kcount<=kmax; kcount++){

                            // get bin neighbor gid 
                            size_t neighbor_bin_gid = get_gid(icount, jcount, kcount, num_bins_x, num_bins_y);
                            num_points_found += num_points_in_bin(neighbor_bin_gid);

                        } // end for kcount
                    } // end for jcount
                } // end for icount

                // the min number of points required to solve the system is num_poly_basis+1
                if (num_points_found > num_poly_basis+5){

                    points_bin_stencil(point_gid) = stencil;
                    points_num_neighbors(point_gid) = num_points_found; // key for allocations
                    break;
                }
                
            } // end for stencil


        }); // end for all
        Kokkos::fence();
        points_bin_stencil.update_host();
        points_num_neighbors.update_host();
        
        // allocate memory for points in point
        DRaggedRightArrayKokkos <size_t> points_in_point(points_num_neighbors, "points_in_point");

        // ---------------------
        // Save the neighbors
        // ---------------------

        // find my neighbors using bins
        FOR_ALL(point_gid, 0, num_points, {

            // get bin gid for this point
            size_t bin_gid = points_bin_gid(point_gid);
                    
            // get i,j,k for this bin
            bin_keys_t bin_keys = keys_in_bin(bin_gid);

            const int i = bin_keys.i;
            const int j = bin_keys.j;
            const int k = bin_keys.k;

            // walk over the stencil to get neighbors
            const int stencil = points_bin_stencil(point_gid);

            const int imin = MAX(0, i-stencil);
            const int imax = MIN(num_bins_x-1, i+stencil);

            const int jmin = MAX(0, j-stencil);
            const int jmax = MIN(num_bins_y-1, j+stencil);

            const int kmin = MAX(0, k-stencil);
            const int kmax = MIN(num_bins_z-1, k+stencil);

            size_t num_saved = 0;
            size_t num_points_found = 0;

            for (int icount=imin; icount<=imax; icount++){
                for (int jcount=jmin; jcount<=jmax; jcount++) {
                    for (int kcount=kmin; kcount<=kmax; kcount++){

                        // get bin neighbor gid 
                        size_t neighbor_bin_gid = get_gid(icount, jcount, kcount, num_bins_x, num_bins_y);
                        num_points_found += num_points_in_bin(neighbor_bin_gid);

                        // save the points in this bin
                        for(size_t neighbor_pt_lid=0; neighbor_pt_lid<num_points_in_bin(neighbor_bin_gid); neighbor_pt_lid++){

                            size_t neighbor_point_gid = points_in_bin(neighbor_bin_gid, neighbor_pt_lid);

                            points_in_point(point_gid, num_saved) = neighbor_point_gid;
                            
                            num_saved++;

                        } // neighbor_point_lid

                    } // end for kcount
                } // end for jcount
            } // end for icount        

        }); // end for all
        Kokkos::fence();
        points_in_point.update_host();



        // ----------------------------------------
        // Find basis that reconstructs polynomial 
        // ----------------------------------------

        printf("Reconstructing basis using point cloud data \n\n");


        CArrayKokkos <double> p_coeffs(num_points, num_poly_basis); // reproducing kernel coefficients at each point
        CArrayKokkos <double> vol(num_points);
        vol.set_values(1.0);

        CArrayKokkos <double> M_inv(num_points, num_poly_basis, num_poly_basis);
        CArrayKokkos <double> grad_M(num_points, num_poly_basis, num_poly_basis);
        
        DRaggedRightArrayKokkos <double> basis(points_num_neighbors);        // reproducing kernel basis (num_points, num_neighbors)
        DRaggedRightArrayKokkos <double> grad_basis(points_num_neighbors,3); // reproducing kernel basis (num_points, num_neighbors)


        double h = 1.0;

        printf("building reproducing kernel coefficients \n");

        // build coefficients on basis functions
        calc_p_coefficients(point_positions, 
                              points_num_neighbors, 
                              points_in_point, 
                              vol, 
                              p_coeffs, 
                              M_inv,
                              h);
        
        // performing checks on p_coeffs
        double partion_unity;
        double partion_unity_lcl;

        double linear_preserving;
        double linear_preserving_lcl;

        double quadratic_preserving;
        double quadratic_preserving_lcl;

        // loop over the particles in the domain
        for(size_t point_gid=0; point_gid<num_points; point_gid++){
            
            // build basis functions at point i
            calc_shape_functions(point_gid, 
                                    point_positions, 
                                    points_num_neighbors, 
                                    points_in_point, 
                                    vol, 
                                    p_coeffs, 
                                    basis, 
                                    h);

            // partition of unity
            FOR_REDUCE_SUM(neighbor_point_lid, 0, points_num_neighbors.host(point_gid), partion_unity_lcl, {
                partion_unity_lcl += basis(point_gid,neighbor_point_lid)*vol(neighbor_point_lid);
            }, partion_unity);
            

            // linear reproducing
            FOR_REDUCE_SUM(neighbor_point_lid, 0, points_num_neighbors.host(point_gid), linear_preserving_lcl, {
                // get the point gid for this neighboring
                size_t neighbor_point_gid = points_in_point(point_gid, neighbor_point_lid);
                linear_preserving_lcl += basis(point_gid,neighbor_point_lid)*vol(neighbor_point_gid)*point_positions(neighbor_point_gid,0);
            }, linear_preserving);


            // quadratic reproducing
            FOR_REDUCE_SUM(neighbor_point_lid, 0, points_num_neighbors.host(point_gid), quadratic_preserving_lcl, {
                // get the point gid for this neighboring
                size_t neighbor_point_gid = points_in_point(point_gid, neighbor_point_lid);
                quadratic_preserving_lcl += basis(point_gid,neighbor_point_lid)*vol(neighbor_point_gid)*point_positions(neighbor_point_gid,0)*point_positions(neighbor_point_gid,0);
            }, quadratic_preserving);

            printf("partition unity = %f, ", partion_unity);
            printf("linear fcn error = %f, ", fabs(linear_preserving-point_positions(point_gid,0)));
            printf("quadratic fcn error = %f at i=%zu \n", fabs(quadratic_preserving-point_positions(point_gid,0)*point_positions(point_gid,0)), point_gid);

        } // end for point gid


        printf("Writing VTK Graphics File \n\n");

        std::ofstream out("cloud.vtk");

        out << "# vtk DataFile Version 3.0\n";
        out << "3D point cloud\n";
        out << "ASCII\n";
        out << "DATASET POLYDATA\n";
        out << "POINTS " << num_points << " float\n";
        for (size_t point_gid = 0; point_gid < num_points; ++point_gid) {
            out << point_positions.host(point_gid,0) << " " 
                << point_positions.host(point_gid,1) << " " 
                << point_positions.host(point_gid,2) << "\n";
        }

        out << "\nPOINT_DATA " << num_points << "\n";
        out << "SCALARS field float 1\n";
        out << "LOOKUP_TABLE default\n";
        for (size_t point_gid = 0; point_gid < num_points; ++point_gid) {
            out << point_values.host(point_gid) << "\n";
        }

    
        printf("Finished \n\n");



    } // end of kokkos scope


    Kokkos::finalize();



    return 0;
    
} // end main
