#pragma once

#include "sim_parameters.h"
#include "matar.h"

void initialize_comp(const SimParameters &sp, DCArrayKokkos<double> &comp, CArray<double> &comp_all);
