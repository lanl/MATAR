#pragma once

#include "matar.h"

// function to output simulation progress
void track_progress(int iter, int* nn, CArrayKokkos<double> &comp);

// function to write vtk files for visualization
void write_vtk(int iter, int* nn, double* delta, CArrayKokkos<double> &comp);

// function to write total_free_energy to file
void output_total_free_energy(int iter, int print_rate, int num_steps, int* nn, 
                              double* delta, double kappa, CArrayKokkos<double> &comp);
