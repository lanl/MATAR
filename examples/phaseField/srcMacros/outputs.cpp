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
#include <stdio.h>

#include "outputs.h"
#include "local_free_energy.h"

void track_progress(int iter, int* nn, DCArrayKokkos<double>& comp)
{
    // unpack simimulation parameters needed
    // for calculations in this function
    int nx = nn[0];
    int ny = nn[1];
    int nz = nn[2];

    // sum of comp field
    double sum_comp = 0.0;
    double loc_sum  = 0.0;
    REDUCE_SUM(i, 0, nx,
               j, 0, ny,
               k, 0, nz,
               loc_sum, {
        loc_sum += comp(i, j, k);
               }, sum_comp);

    // max of comp field
    double max_comp;
    double loc_max;
    REDUCE_MAX(i, 0, nx,
               j, 0, ny,
               k, 0, nz,
               loc_max, {
        if (loc_max < comp(i, j, k)) {
            loc_max = comp(i, j, k);
        }
               },
               max_comp);

    // min of comp field
    double min_comp;
    double loc_min;
    REDUCE_MIN(i, 0, nx,
               j, 0, ny,
               k, 0, nz,
               loc_min, {
        if (loc_min > comp(i, j, k)) {
            loc_min = comp(i, j, k);
        }
               },
               min_comp);

    printf("\n----------------------------------------------------\n");
    printf("Iteration : %d\n", iter);
    printf("Conservation of comp : %f\n", sum_comp);
    printf("Max comp : %f\n", max_comp);
    printf("Min comp : %f\n", min_comp);
}

void write_vtk(int iter, int* nn, double* delta, DCArrayKokkos<double>& comp)
{
    // unpack simimulation parameters needed
    // for calculations in this function
    int    nx = nn[0];
    int    ny = nn[1];
    int    nz = nn[2];
    double dx = delta[0];
    double dy = delta[1];
    double dz = delta[2];

    // update host copy of comp
    comp.update_host();

    // output file management
    FILE* output_file;
    char  filename[50];

    // create name of output vtk file
    sprintf(filename, "outputComp_%d.vtk", iter);

    // open output vtk file
    output_file = fopen(filename, "w");

    // write vtk file heading
    fprintf(output_file, "%s\n", "# vtk DataFile Version 2.0");
    fprintf(output_file, "%s\n", filename);
    fprintf(output_file, "%s\n", "ASCII");
    fprintf(output_file, "%s\n", "DATASET STRUCTURED_GRID");
    fprintf(output_file, "%s %i  %i  %i\n", "DIMENSIONS", nx, ny, nz);
    fprintf(output_file, "%s   %i   %s\n", "POINTS", nx * ny * nz, "double");

    // write grid point values
    // Note: order of for loop is important (k,j,i)
    double x, y, z;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                x = double(i) * dx;
                y = double(j) * dy;
                z = double(k) * dz;
                fprintf(output_file, "  %12.6E  %12.6E  %12.6E\n", x, y, z);
            }
        }
    }

    // write data values
    // Note: order of for loop is important (k,j,i)
    fprintf(output_file, "%s %i\n", "POINT_DATA", nx * ny * nz);
    fprintf(output_file, "%s\n", "SCALARS data double");
    fprintf(output_file, "%s\n", "LOOKUP_TABLE default");
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                fprintf(output_file, " %12.6E\n", comp.host(i, j, k));
            }
        }
    }

    // close file
    fclose(output_file);
}

void output_total_free_energy(int iter, int print_rate, int num_steps, int* nn,
    double* delta, double kappa, DCArrayKokkos<double>& comp)
{
    // get total_free_energy
    double total_free_energy = calculate_total_free_energy(nn, delta, kappa, comp);

    // output file management
    static FILE* output_file;
    static char  filename[50];

    // open output vtk file
    if (iter == print_rate) {
        // create name of output vtk file
        sprintf(filename, "total_free_energy.csv");
        output_file = fopen(filename, "w");
    }

    // write total_free_energy to file
    fprintf(output_file, "%i,%12.6E\n", iter, total_free_energy);

    // close file
    if (iter == num_steps) {
        fclose(output_file);
    }
}
