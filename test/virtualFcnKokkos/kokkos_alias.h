#ifndef KOKKOS_ALIAS_H
#define KOKKOS_ALIAS_H

#include <stdlib.h> 
#include "parents.h"
#include <Kokkos_Core.hpp>

//MACROS to make the code less scary
#define kmalloc(size) ( Kokkos::kokkos_malloc<MemSpace>(size) )
#define kfree(pnt)        (  Kokkos::kokkos_free(pnt) ) 
#define ProfileRegionStart  ( Kokkos::Profiling::pushRegion )
#define ProfileRegionEnd  ( Kokkos::Profiling::popRegion )

using real_t = double;
using u_int  = unsigned int;

#ifdef HAVE_CUDA
//using UVMMemSpace     = Kokkos::CudaUVMSpace;
using MemSpace        = Kokkos::CudaSpace;
using ExecSpace       = Kokkos::Cuda;
using Layout          = Kokkos::LayoutLeft;
#endif

// Won't have both
#if HAVE_OPENMP
using MemSpace        = Kokkos::HostSpace;
using ExecSpace       = Kokkos::OpenMP;
using Layout          = Kokkos::LayoutRight;
#endif

using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
using mdrange_policy2 = Kokkos::MDRangePolicy<Kokkos::Rank<2>>;
using mdrange_policy3 = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

using RMatrix1D    = Kokkos::View<real_t *,Layout,ExecSpace>;
using RMatrix2D    = Kokkos::View<real_t **,Layout,ExecSpace>;
using RMatrix3D    = Kokkos::View<real_t ***,Layout,ExecSpace>;
using RMatrix4D    = Kokkos::View<real_t ****,Layout,ExecSpace>;
using RMatrix5D    = Kokkos::View<real_t *****,Layout,ExecSpace>;
using IMatrix1D    = Kokkos::View<int *,Layout,ExecSpace>;
using IMatrix2D    = Kokkos::View<int **,Layout,ExecSpace>;
using IMatrix3D    = Kokkos::View<int ***,Layout,ExecSpace>;
using IMatrix4D    = Kokkos::View<int ****,Layout,ExecSpace>;
using IMatrix5D    = Kokkos::View<int *****,Layout,ExecSpace>;
using SVar         = Kokkos::View<size_t,Layout,ExecSpace>;
using SArray1D     = Kokkos::View<size_t *,Layout,ExecSpace>;
using SArray2D     = Kokkos::View<size_t **,Layout,ExecSpace>;
using SArray3D     = Kokkos::View<size_t ***,Layout,ExecSpace>;
using SArray4D     = Kokkos::View<size_t ****,Layout,ExecSpace>;
using SArray5D     = Kokkos::View<size_t *****,Layout,ExecSpace>;

using SHArray1D     = Kokkos::View<size_t *,Layout,Kokkos::HostSpace>;

using Parent1D     = Kokkos::View<parent_models*,Layout,ExecSpace>;
using ParentHost1D = Kokkos::View<parent_models*,Layout,Kokkos::HostSpace>;

#endif
