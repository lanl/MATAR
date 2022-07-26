#pragma once
#include "matar.h"

struct GlobalArrays
{
    DCArrayKokkos<double> comp;
    CArrayKokkos<double> dfdc;

    GlobalArrays(int* nn);
};
