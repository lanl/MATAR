#pragma once
#include "matar.h"

struct GlobalArrays
{
    CArrayKokkos<double> comp;
    CArrayKokkos<double> dfdc;

    GlobalArrays(int* nn);
};