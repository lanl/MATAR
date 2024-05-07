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
#ifndef KOKKOS_ALIAS_H
#define KOKKOS_ALIAS_H

#include <stdlib.h>
#include "parents.h"
#include <Kokkos_Core.hpp>
#include "matar.h"

// MACROS to make the code less scary
// #define kmalloc(size) ( Kokkos::kokkos_malloc<DefaultMemSpace>(size) )
// #define kfree(pnt)        (  Kokkos::kokkos_free(pnt) )
// #define ProfileRegionStart  ( Kokkos::Profiling::pushRegion )
// #define ProfileRegionEnd  ( Kokkos::Profiling::popRegion )

using real_t = double;
using u_int  = unsigned int;

/*
#ifdef HAVE_CUDA
//using UVMMemSpace     = Kokkos::CudaUVMSpace;
using DefaultMemSpace  = Kokkos::CudaSpace;
using DefaultExecSpace = Kokkos::Cuda;
using DefaultLayout    = Kokkos::LayoutLeft;
#elif HAVE_OPENMP
using DefaultMemSpace  = Kokkos::HostSpace;
using DefaultExecSpace = Kokkos::OpenMP;
using DefaultLayout    = Kokkos::LayoutRight;
#elif TRILINOS_INTERFACE
using DefaultMemSpace  = void;
using DefaultExecSpace = void;
using DefaultLayout    = void;
#elif HAVE_HIP
using DefaultMemSpace  = Kokkos::HipSpace;
using DefaultExecSpace = Kokkos::Hip;
using DefaultLayout    = Kokkos::LayoutLeft;
#endif
*/

using TeamPolicy = Kokkos::TeamPolicy<DefaultExecSpace>;
using mdrange_policy2 = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
using mdrange_policy3 = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

using RMatrix1D = Kokkos::View<real_t*, DefaultLayout, DefaultExecSpace>;
using RMatrix2D = Kokkos::View<real_t**, DefaultLayout, DefaultExecSpace>;
using RMatrix3D = Kokkos::View<real_t***, DefaultLayout, DefaultExecSpace>;
using RMatrix4D = Kokkos::View<real_t****, DefaultLayout, DefaultExecSpace>;
using RMatrix5D = Kokkos::View<real_t*****, DefaultLayout, DefaultExecSpace>;
using IMatrix1D = Kokkos::View<int*, DefaultLayout, DefaultExecSpace>;
using IMatrix2D = Kokkos::View<int**, DefaultLayout, DefaultExecSpace>;
using IMatrix3D = Kokkos::View<int***, DefaultLayout, DefaultExecSpace>;
using IMatrix4D = Kokkos::View<int****, DefaultLayout, DefaultExecSpace>;
using IMatrix5D = Kokkos::View<int*****, DefaultLayout, DefaultExecSpace>;
using SVar     = Kokkos::View<size_t, DefaultLayout, DefaultExecSpace>;
using SArray1D = Kokkos::View<size_t*, DefaultLayout, DefaultExecSpace>;
using SArray2D = Kokkos::View<size_t**, DefaultLayout, DefaultExecSpace>;
using SArray3D = Kokkos::View<size_t***, DefaultLayout, DefaultExecSpace>;
using SArray4D = Kokkos::View<size_t****, DefaultLayout, DefaultExecSpace>;
using SArray5D = Kokkos::View<size_t*****, DefaultLayout, DefaultExecSpace>;

using SHArray1D = Kokkos::View<size_t*, DefaultLayout, Kokkos::HostSpace>;

using Parent1D     = Kokkos::View<parent_models*, DefaultLayout, DefaultExecSpace>;
using ParentHost1D = Kokkos::View<parent_models*, DefaultLayout, Kokkos::HostSpace>;

#endif
