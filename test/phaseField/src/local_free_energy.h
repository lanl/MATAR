#pragma once
#include "matar.h"

double calculate_total_free_energy(int* nn, double* delta, double kappa, CArrayKokkos<double> &comp);

void calculate_dfdc(int* nn, CArrayKokkos<double> &comp, CArrayKokkos<double> &dfdc);
