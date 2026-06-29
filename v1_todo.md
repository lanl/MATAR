# MATAR v1 Release — To-Do List

> Prepared: June 2026  
> Branch: `MPI_Updates`  
> Scope: Low-hanging-fruit improvements targeting a fall 2026 v1 release.  
> Categories: Build System · OS Robustness · CPU-GPU Portability · Documentation · Performance

> **Note on ordering:** Section 5 (Build System Refactor) is the primary v1 goal and is listed last only because many items in Sections 1–4 reference it. Read Section 5 first to understand the overall direction, then read Sections 1–4 for the code-level work that runs in parallel.

---

## 1. OS Robustness

### 1.1 `matar.h` unconditionally includes `mpi_types.h` and `tpetra_wrapper_types.h`
- **File:** `src/include/matar.h:84-86`
- **Problem:** All three optional headers (`mpi_types.h`, `mapped_mpi_types.h`, `tpetra_wrapper_types.h`) are included without preprocessor guards. Even though each file guards its content with `#ifdef HAVE_MPI` / `#ifdef TRILINOS_INTERFACE`, the files are still found and opened by the preprocessor on every build. If a downstream project installs only the serial subset of MATAR, compilation fails because the files are absent. Additionally, `mapped_mpi_types.h` uses an angle-bracket include `<mpi_types.h>` (line 46) instead of `"mpi_types.h"` — this only resolves if the MATAR include directory is on the system path, which is fragile.
- **Fix:** Wrap with preprocessor guards:
  ```cpp
  #ifdef HAVE_MPI
  #include "mpi_types.h"
  #include "mapped_mpi_types.h"
  #endif
  #ifdef TRILINOS_INTERFACE
  #include "tpetra_wrapper_types.h"
  #endif
  ```
  Also fix the angle-bracket include in `mapped_mpi_types.h:46` to `#include "mpi_types.h"`. The `HAVE_MPI` and `TRILINOS_INTERFACE` macros will be set correctly via `CMakePresets.json` (see §5.1) and propagated through the installed `MatarTargets.cmake` (see §5.1 / §1.3).
- **Priority:** High — breaks clean non-MPI installs today.

### 1.2 Backend detection macros (`HAVE_CUDA`, `HAVE_OPENMP`, `HAVE_HIP`) not propagated by CMake install
- **File:** `src/include/kokkos_types.h:49-70`, `CMakeLists.txt:76`
- **Problem:** The root `CMakeLists.txt` only calls `add_definitions(-DHAVE_KOKKOS=1)`. It never sets `HAVE_CUDA`, `HAVE_OPENMP`, `HAVE_HIP`, or `HAVE_THREADS`. Downstream consumers who link MATAR via `find_package(Matar)` receive no backend macros; `kokkos_types.h` silently falls through to the `#else` branch, using `LayoutLeft` even for OpenMP builds (which should use `LayoutRight` for C-order cache locality).
- **Fix:** In `CMakeLists.txt`, after `find_package(Kokkos REQUIRED)`, query Kokkos's own config variables and propagate them as interface definitions:
  ```cmake
  foreach(_backend CUDA HIP OPENMP SYCL THREADS)
    if(Kokkos_ENABLE_${_backend})
      target_compile_definitions(matar INTERFACE HAVE_${_backend}=1)
    endif()
  endforeach()
  ```
  The `CMakePresets.json` (§5.1) will pass the right Kokkos enable flags, so the backend macros flow end-to-end without manual `-DHAVE_CUDA=1` arguments.
- **Priority:** High — wrong layout silently chosen for OpenMP builds.

### 1.3 HIP backend uses deprecated `Kokkos::Experimental` namespace
- **File:** `src/include/kokkos_types.h:63-64`
- **Problem:**
  ```cpp
  using DefaultMemSpace  = Kokkos::Experimental::HIPSpace;
  using DefaultExecSpace = Kokkos::Experimental::HIP;
  ```
  These were promoted out of `Experimental` in Kokkos 3.7/4.x. Builds against modern Kokkos on Frontier and Crusher produce compilation errors or deprecation warnings today.
- **Fix:** Drop the `Experimental::` prefix unconditionally — all Kokkos versions targeted for MATAR v1 are ≥3.7:
  ```cpp
  using DefaultMemSpace  = Kokkos::HIPSpace;
  using DefaultExecSpace = Kokkos::HIP;
  ```
  If compatibility with Kokkos < 3.7 must be preserved, guard with `#if KOKKOS_VERSION >= 30700`.
- **Priority:** High — breaks on current Frontier/Crusher toolchains.

### 1.4 `_old` files installed alongside live headers
- **File:** `CMakeLists.txt:110`
- **Problem:** `install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/include/ ...)` installs `communication_plan_old.h` and `mpi_types_old.h`. Users and IDEs auto-complete from these, and they add ~50 KB of dead code to the installed package.
- **Fix:** Confirm no active file includes them (`grep -r "mpi_types_old\|communication_plan_old" src/`), then either delete them or exclude them from the install:
  ```cmake
  install(DIRECTORY src/include/ DESTINATION include
          PATTERN "*_old.h" EXCLUDE)
  ```
- **Priority:** Low.

### 1.5 No `MATAR_VERSION` macro exposed to C++ consumers
- **File:** `CMakeLists.txt:9, 86`
- **Problem:** `project(MATAR)` has no `VERSION` field. `MatarConfigVersion.cmake` hardcodes `VERSION 1.0`. Downstream projects cannot do compile-time version checks (`#if MATAR_VERSION_MAJOR >= 1`).
- **Fix:** Change to `project(MATAR VERSION 1.0.0)` and derive the config-version file from `${PROJECT_VERSION}`. Add a `configure_file` step to generate `matar_version.h`:
  ```cmake
  configure_file(cmake/matar_version.h.in include/matar_version.h)
  install(FILES ${CMAKE_BINARY_DIR}/include/matar_version.h DESTINATION include)
  ```
  where `matar_version.h.in` exposes `MATAR_VERSION_MAJOR`, `MATAR_VERSION_MINOR`, `MATAR_VERSION_PATCH`.
- **Priority:** Medium.

### 1.6 CI: macOS runners use `--machine=linux` and CI doesn't run `ctest`
- **Files:** `.github/workflows/test.yml:63`, `.github/workflows/cmake.yml:82-86`
- **Problem:** Two separate CI issues that are both resolved by the build system refactor (§5.3):
  1. All `TEST_MAC_*` jobs call `build-matar.sh --machine=linux`, bypassing macOS-specific compiler selection and the core-count guard.
  2. `cmake.yml` has `ctest` commented out, so it only validates compilation, not correctness.
- **Fix:** Both are addressed by §5.3 (replace CI with `cmake --preset` invocations). Once the preset-based CI is in place: the macOS jobs use a `macos` preset (no machine flag needed) and `ctest --preset` runs tests correctly. The dead `cmake.yml` can be either deleted or merged into `test.yml` as a build-only job with no script dependency.
- **Priority:** High (resolved by §5.3; flag here for tracking).

### 1.7 `test/CMakeLists.txt` uses non-standard `-DCUDA=ON` backend variables
- **File:** `test/CMakeLists.txt:29-51`
- **Problem:** The test CMakeLists checks `if(CUDA)`, `if(HIP)`, `if(OPENMP)` — ad-hoc cache variables that must be passed by the caller. Any user who runs `cmake` directly (without the wrapper script) gets no backend definitions emitted, so the test binaries compile for the wrong target.
- **Fix:** This is resolved by §5.4 (fold tests into root CMakeLists). With §1.2 (backend macros propagated via `MatarTargets.cmake`), the test CMakeLists can simply link `matar` and receive all backend macros transitively. Remove the `if(CUDA)/if(HIP)` block entirely.
- **Priority:** Medium (resolved by §5.4 + §1.2; flag here for tracking).

### 1.8 No macOS MPI CI coverage
- **File:** `.github/workflows/test.yml`
- **Problem:** The macOS matrix only tests serial Kokkos. MPI availability on macOS runners (`brew install open-mpi`) is not verified.
- **Fix:** After §5.3 lands, add a `serial-mpi` preset entry in the macOS CI matrix. Gate it with a `brew install open-mpi` step matching the Ubuntu `apt-get` step.
- **Priority:** Low — good-to-have, not a v1 blocker.

---

## 2. CPU-GPU Portability

### 2.1 `FOR_FIRST` / `DO_FIRST` hardcode GPU warp size to 32
- **File:** `src/include/macros.h:847-881`
- **Problem:** `Kokkos::TeamPolicy<>((x1)-(x0), Kokkos::AUTO, 32)` hardcodes vector length to 32. CUDA warps are 32-wide, but AMD HIP wavefronts are 64-wide and CPU SIMD varies. The hardcoded `32` is silently suboptimal or wrong on non-NVIDIA targets.
- **Fix:** Replace `32` with `Kokkos::AUTO`:
  ```cpp
  Kokkos::TeamPolicy<>((x1)-(x0), Kokkos::AUTO, Kokkos::AUTO)
  ```
- **Priority:** High — direct performance/correctness impact on HIP/CPU builds.

### 2.2 `F_LOOP_ORDER` is `Kokkos::Iterate::Right` — wrong for Fortran column-major layout on GPU
- **File:** `src/include/macros.h:127-129`
- **Problem:**
  ```cpp
  #define LOOP_ORDER   Kokkos::Iterate::Right   // C arrays: last index fastest — correct
  #define F_LOOP_ORDER Kokkos::Iterate::Right   // F arrays: WRONG — should be Left
  ```
  `DO_ALL` macros for FArray/FMatrix types use `F_LOOP_ORDER`. `Iterate::Right` makes the *last* dimension contiguous in the GPU thread mapping, which is correct for C-order arrays but wrong for Fortran arrays where the *first* index varies fastest. This causes non-coalesced memory access on GPU for all F-type arrays.
- **Fix:** `#define F_LOOP_ORDER Kokkos::Iterate::Left`
- **Priority:** High — GPU memory access pattern is wrong for all F-type arrays.

### 2.3 `policy2D`, `policy3D`, `policy4D` don't bind to `DefaultExecSpace`
- **File:** `src/include/kokkos_types.h:82-84`
- **Problem:**
  ```cpp
  using policy2D = Kokkos::MDRangePolicy< Kokkos::Rank<2> >;
  ```
  Without an explicit execution space, these resolve to the Kokkos global default, which may differ from MATAR's configured `DefaultExecSpace`. Mixing `policy2D` with a `CArrayKokkos<T, DefaultLayout, DefaultExecSpace>` can cause Kokkos to complain or silently dispatch to the wrong device.
- **Fix:**
  ```cpp
  using policy2D = Kokkos::MDRangePolicy<Kokkos::Rank<2>, DefaultExecSpace>;
  using policy3D = Kokkos::MDRangePolicy<Kokkos::Rank<3>, DefaultExecSpace>;
  using policy4D = Kokkos::MDRangePolicy<Kokkos::Rank<4>, DefaultExecSpace>;
  ```
- **Priority:** Medium.

### 2.4 `real_t` and `u_int` in the global namespace — POSIX collision on Linux
- **File:** `src/include/aliases.h:44-45`
- **Problem:**
  ```cpp
  using real_t = double;
  using u_int  = unsigned int;   // conflicts with POSIX u_int from <sys/types.h>
  ```
  Both declarations are at global scope, outside `namespace mtr`. `u_int` is a POSIX typedef on Linux; any translation unit that includes both `<sys/types.h>` and `matar.h` gets a redefinition error.
- **Fix:** Move both into `namespace mtr`:
  ```cpp
  namespace mtr {
      using real_t = double;
      using u_int  = unsigned int;
  }
  ```
  Audit downstream code (Fierro, ELEMENTS) for unqualified uses of `real_t` and `u_int` — they will need `mtr::real_t` or a `using namespace mtr` after this change. Document as a v1 breaking change in `MIGRATION.md` (see §3.7).
- **Priority:** High — actual compile failure with common system headers on Linux.

### 2.5 `FArrayKokkos` 6D and 7D constructors have a parameter name typo (`sone_dim2`)
- **File:** `src/include/kokkos_types.h:178-183`
- **Problem:**
  ```cpp
  FArrayKokkos(size_t dim0, size_t sone_dim2, size_t dim2, ...);
  ```
  The second parameter is named `sone_dim2` (should be `dim1`). The name leak into generated Doxygen and IDE tooltips for every 6D and 7D constructor across all Kokkos array/matrix types.
- **Fix:** Rename `sone_dim2` → `dim1` in all 6D and 7D declarations and definitions throughout `kokkos_types.h`. Verify with `grep -n sone_dim2 src/include/kokkos_types.h`.
- **Priority:** Medium — cosmetic but pollutes all generated documentation.

### 2.6 FOR_ALL macro loop indices use `const int` — limits to ~2B elements
- **File:** `src/include/macros.h:167-168`
- **Problem:** `KOKKOS_LAMBDA(const int (i))` caps the loop index at `INT_MAX` (~2.1B). On GPU with HBM (32–80 GB), arrays with >2^31 elements are common (e.g., 8 GB of floats = 2^31 elements). The Kokkos `RangePolicy<>` itself supports 64-bit ranges, but the lambda signature overrides the index type. The serial fallback functions (`for_all`, lines 931–968) also use `int`.
- **Fix:** Change all Kokkos macro lambda signatures from `const int` to `const int64_t` (or `Kokkos::RangePolicy<>::index_type`). Change the serial `for_all` function signatures from `int` to `ptrdiff_t` or `int64_t`.
- **Priority:** Medium — silent data corruption for large GPU problems.

### 2.7 `DefaultLayout` for CArrayKokkos is `LayoutLeft` on CUDA, contradicting C-order semantics
- **File:** `src/include/kokkos_types.h:51-53`
- **Problem:** The CUDA branch sets `DefaultLayout = Kokkos::LayoutLeft` (column-major). `CArrayKokkos` is documented as C-order (last index fastest = row-major = `LayoutRight`). Using `LayoutLeft` for a C-named type confuses users porting CPU `CArray` code to GPU, because `CArray` uses `LayoutRight`. The mismatch also means C-order 2D access (`A(i,j)` with j varying fastest in the loop) is non-coalesced on the GPU with the current default.
- **Fix:** Decouple `DefaultLayout` from the execution backend. Instead, define layout on a per-type basis: `CArrayKokkos` should default to `LayoutRight` and `FArrayKokkos` to `LayoutLeft` regardless of backend. The template parameter already allows overriding; change only the defaults. Document the tradeoff in the type-selection guide (§3.10).
- **Priority:** Medium — affects all 2D+ CArrayKokkos uses on CUDA.

### 2.8 No SYCL backend path in `kokkos_types.h`
- **File:** `src/include/kokkos_types.h:49-70`
- **Problem:** The backend dispatch chain handles CUDA, OpenMP, Threads, and HIP, then falls back to `Kokkos::DefaultExecutionSpace`. There is no `#elif HAVE_SYCL` branch. Intel GPU clusters (Aurora at Argonne) use the Kokkos SYCL backend; MATAR builds targeting those machines fall through to the generic branch with no explicit memory space selection.
- **Fix:** Add a SYCL branch (after §1.2 propagates `HAVE_SYCL` via CMake):
  ```cpp
  #elif HAVE_SYCL
  using DefaultMemSpace  = Kokkos::Experimental::SYCLDeviceUSMSpace;
  using DefaultExecSpace = Kokkos::Experimental::SYCL;
  using DefaultLayout    = Kokkos::LayoutLeft;
  ```
  Add a `sycl` preset to `CMakePresets.json` (§5.1) that enables `Kokkos_ENABLE_SYCL` and passes `-DHAVE_SYCL=1` via `target_compile_definitions`.
- **Priority:** Low — Intel Aurora is a near-term target for DOE codes but not a v1 blocker.

### 2.9 No CMake assertion that Kokkos was built with the requested backend
- **File:** `CMakeLists.txt:63-64`
- **Problem:** `find_package(Kokkos REQUIRED)` succeeds even when the installed Kokkos was built without CUDA. The mismatch is discovered at runtime (or not at all — kernels silently run on the host). The `CMakePresets.json` approach (§5.1) will use `FetchContent` to build the right Kokkos from scratch, making this less likely for new users. But users who point to a pre-installed Kokkos via `Kokkos_DIR` still hit this.
- **Fix:** After `find_package(Kokkos)`, assert the expected backend is enabled:
  ```cmake
  if(Matar_CUDA_BUILD)
    kokkos_check(DEVICES CUDA)          # fatal error if CUDA not in this Kokkos install
  elseif(Matar_HIP_BUILD)
    kokkos_check(DEVICES HIP)
  endif()
  ```
- **Priority:** Medium.

---

## 3. Documentation

### 3.1 `host_types.h` has zero Doxygen comments across 5400+ lines
- **File:** `src/include/host_types.h`
- **Problem:** There are no `/*!`, `///`, or `/** */` Doxygen comments anywhere in `host_types.h`. The full serial type hierarchy — FArray, CArray, FMatrix, CMatrix, ViewFArray, ViewCArray, RaggedRightArray, RaggedDownArray, CSRArray, CSCArray (~14 types) — has zero per-method documentation. `kokkos_types.h` has Doxygen for the first type only (FArrayKokkos 1D–3D constructors); the remaining ~11,000 lines across the other Kokkos types are also bare. The Doxygen config (`docs_doxygen/Doxyfile`) exists but generates minimal output.
- **Fix:** Add `@brief`, `@param`, `@return` blocks to the constructors, `operator()`, `size()`, `dims()`, `pointer()`, and `set_values()` for all types. Start with `host_types.h` (CPU-only users' entry point), then `DCArrayKokkos`/`DCMatrixKokkos` (most common GPU types). The comment pattern is uniform across all types — a script can generate stubs in a few hours.
- **Priority:** High — most-used types in the library have zero generated docs.

### 3.2 MPI types (`mpi_types.h`, `mapped_mpi_types.h`, `communication_plan.h`) have no Doxygen
- **Files:** `src/include/mpi_types.h`, `src/include/mapped_mpi_types.h`, `src/include/communication_plan.h`
- **Problem:** `MPICArrayKokkos`, `MPICMatrixKokkos`, `CommunicationPlan`, `PartitionMap`, and all related classes have no class-level `\brief` comments or method-level documentation. These are the most complex types in the library and the least documented.
- **Fix:** Add at minimum a class-level `\brief` and method documentation for the key public API: constructors, `communicate()`, `get_comm_plan()`, `update_host()`, `update_device()`. Add code examples (see §3.5).
- **Priority:** High — new API, no documentation = unusable for newcomers.

### 3.3 Doxygen comments absent for 4D–7D constructors in all Kokkos types
- **File:** `src/include/kokkos_types.h`
- **Problem:** Only 1D–3D constructors have `\brief` / `\param` Doxygen. All 4D, 5D, 6D, 7D constructors are undocumented. The pattern repeats across all 16 Kokkos array/matrix types (~100 constructor overloads).
- **Fix:** Add `\brief` and `\param` entries for 4D–7D constructors. The pattern is uniform; a templated sed/awk script can generate stubs from the existing 1D–3D docs in an hour.
- **Priority:** Medium — affects all generated API docs.

### 3.4 README typos and broken code examples
- **File:** `README.md:23, 31, 45`
- **Problems:**
  - Lines 23, 31: "convection" → "convention" (appears twice in array access descriptions)
  - Line 45: "idetical" → "identical"
  - Lines 25, 34: for-loop syntax uses commas (`for (i=0,i<N,i++)`) instead of semicolons — broken pseudocode that will confuse new users
- **Fix:** Fix all spelling errors. Change the pseudocode to valid C++ or add a comment marking it as pseudocode.
- **Priority:** Medium — first impression for new users.

### 3.5 README shows only serial host types — no GPU or MPI quick-start
- **File:** `README.md:50-92`
- **Problem:** The Usage section only shows `CArray`, `ViewCArray`, `CMatrix`, and `RaggedRightArray`. There are no examples of `CArrayKokkos`, `DCArrayKokkos`, `FOR_ALL`, `FOR_REDUCE_SUM`, or `MPICArrayKokkos`. A user with GPU intent sees nothing of MATAR's primary value proposition.
- **Fix:** Add a GPU quick-start section to README showing: a `DCArrayKokkos` allocation, a `FOR_ALL` kernel, a `FOR_REDUCE_SUM`, and `MATAR_KOKKOS_INIT` / `MATAR_KOKKOS_FINALIZE`. Reference the CMake preset workflow (§5.1) so the build instructions match the new system.
- **Priority:** Medium.

### 3.6 No usage example for `PartitionMap` / `CommunicationPlan`
- **Files:** `src/include/partition_map.h`, `src/include/communication_plan.h`, `examples/`
- **Problem:** These are the entry points for distributed-memory programming in MATAR, but there is no quick-start example showing:
  1. How to create a `PartitionMap`
  2. How to derive a `CommunicationPlan` from it
  3. How to perform a halo exchange with `MPICArrayKokkos`
  The existing `examples/laplaceMPI/` and `examples/phaseFieldMPI/` are more complex than a minimal demo.
- **Fix:** Add a minimal `examples/halo_exchange/` showing the three steps above in ~100 lines. Wire it into `CMakePresets.json` via the `Matar_BUILD_EXAMPLES` option (§5.4).
- **Priority:** High — the distributed API is unusable without a tutorial.

### 3.7 No CHANGELOG
- **Problem:** There is no `CHANGELOG.md`. With MATAR used by Fierro and ELEMENTS, downstream users have no machine-readable history of API changes.
- **Fix:** Create `CHANGELOG.md` covering at minimum:
  - What `_old` files replaced (the previous MPI API)
  - The new `MPICArrayKokkos` API vs. the old `MappedMPIArrayKokkos` pattern
  - `CommunicationPlan` API additions
  - The move from bash scripts to `CMakePresets.json` (a user-facing workflow change)
  - Any macro renames or breaking type changes
- **Priority:** High — required for a v1 release.

### 3.8 No v1 migration guide
- **Problem:** Users of Fierro, ELEMENTS, and other downstream projects need to know what changed between pre-v1 and v1. No such document exists.
- **Fix:** Create `MIGRATION.md` covering:
  - New required CMake workflow: `cmake --preset <name>` replaces `source scripts/build-matar.sh`
  - `mtr::real_t` and `mtr::u_int` (namespaced — breaking if §2.4 is fixed)
  - Minimum Kokkos version (≥3.7 for non-`Experimental` HIP)
  - Removed `_old` headers
- **Priority:** Medium — important for downstream projects, especially given the build system change.

### 3.9 No type-selection guide (when to use which MATAR type)
- **Problem:** The distinction between `CArrayKokkos` (device-only), `DCArrayKokkos` (dual host+device), `DViewCArrayKokkos` (wraps existing pointer), and `MPICArrayKokkos` (distributed) is non-obvious. New users regularly pick the wrong type, discover the issue at runtime, and must resort to reading source code.
- **Fix:** Add a decision-tree or table to the README or a `docs/choosing_a_type.md`:
  - CPU-only data → `CArray` / `FArray`
  - GPU-only data (no host access after init) → `CArrayKokkos`
  - Data that moves between CPU and GPU → `DCArrayKokkos`
  - Wrapping an existing host pointer for GPU use → `DViewCArrayKokkos`
  - Distributed data across MPI ranks → `MPICArrayKokkos`
  - Data that grows or shrinks at runtime on device → `DynamicArrayKokkos` (pre-allocate a capacity at construction; use `push_back`/`pop_back` within that capacity; use `resize()` — once §4.4 is implemented — to grow the backing buffer)
- **Priority:** Medium.

### 3.10 `macros.h` header comment shows wrong macro names
- **File:** `src/include/macros.h:73-96`
- **Problem:** The header comment says "The syntax to use the FOR_REDUCE is as follows:" and shows `REDUCE_SUM(...)`. The actual macro is `FOR_REDUCE_SUM(...)`. Users who copy the example get a compile error.
- **Fix:** Update the comment block to use the actual macro names: `FOR_REDUCE_SUM`, `FOR_REDUCE_MAX`, `FOR_REDUCE_MIN`.
- **Priority:** Low.

### 3.11 Sphinx / Doxygen docs are not built in CI
- **Files:** `docs_doxygen/`, `docs_sphinx/`, `.github/workflows/`
- **Problem:** Both a Doxygen config and a Sphinx `conf.py` exist, but neither is run in CI. Broken `\param` entries and broken rST go undetected. There is no Breathe/Exhale integration to pull Doxygen XML into Sphinx.
- **Fix:** Add a `docs.yml` GitHub Actions job: `doxygen docs_doxygen/Doxyfile && make -C docs_sphinx html`. Once `host_types.h` and `kokkos_types.h` have Doxygen coverage (§3.1), add Breathe to `docs_sphinx/conf.py` to pull the API docs into Sphinx.
- **Priority:** Low — implement after §3.1 adds Doxygen coverage.

---

## 4. Performance

### 4.1 `set_values()` launches a parallel kernel instead of `Kokkos::deep_copy`
- **File:** `src/include/kokkos_types.h` (all Kokkos types, e.g., line ~545 for FArrayKokkos)
- **Problem:** All `set_values` implementations do:
  ```cpp
  Kokkos::parallel_for("SetValues", length_,
      KOKKOS_CLASS_LAMBDA(const int i){ this_array_(i) = val; });
  ```
  `Kokkos::deep_copy(view, scalar)` is the correct API: it uses `cudaMemset` for trivially-copyable types on CUDA, avoids kernel-scheduling overhead for small arrays, and is recognized by Kokkos profiling tools as a memory operation rather than a user kernel. The current approach also inherits the `const int` index overflow bug (§2.6).
- **Fix:** Replace all `set_values` bodies with:
  ```cpp
  Kokkos::deep_copy(this_array_, val);
  ```
  For dual-view types (`DCArrayKokkos` etc.), call `Kokkos::deep_copy` on the device view, then `update_host()`.
- **Priority:** Medium.

### 4.2 `CommunicationPlan` displacement setup is O(n²) — should use a prefix sum
- **File:** `src/include/communication_plan.h` (~line 359)
- **Problem:** `send_displs_` and `recv_displs_` are computed with a nested loop:
  ```cpp
  for(int i=0; i<num_send_ranks; i++)
      for(int j=0; j<i; j++) displs[i] += counts[j];
  ```
  This is O(n²) in the number of MPI neighbors. On a fat-tree topology with O(1000) neighbors, this does ~500K additions at every communication setup phase.
- **Fix:** Replace with an O(n) running accumulator:
  ```cpp
  displs[0] = 0;
  for(int i = 1; i < num_ranks; i++)
      displs[i] = displs[i-1] + counts[i-1];
  ```
- **Priority:** Medium.

### 4.3 `CommunicationPlan` issues one GPU fence per buffer during initialization
- **File:** `src/include/communication_plan.h` (~line 211)
- **Problem:** The initialization code calls `update_device()` + `MATAR_FENCE()` separately for each small buffer (`send_rank_ids`, `recv_rank_ids`, `send_counts`, `recv_counts`, etc.). Each `MATAR_FENCE()` is a full `Kokkos::fence()` — a complete GPU synchronization barrier. With 6+ such pairs, init issues 6+ unnecessary barriers on the critical startup path.
- **Fix:** Batch all `update_device()` calls, then issue a single `MATAR_FENCE()` at the end of the initialization function.
- **Priority:** Low.

### 4.4 `DynamicArrayKokkos` is 1D-only and lacks a `resize()` for the backing buffer
- **File:** `src/include/kokkos_types.h:7566-8013`
- **Background:** Fixed-size Kokkos types (`CArrayKokkos`, `DCArrayKokkos`, etc.) are intentionally non-resizable; users who need a dynamically sized array should use `DynamicArrayKokkos`, which pre-allocates a capacity (`dims_[0]`) and tracks a separate logical size (`dims_actual_size_[0]`) via `push_back` / `pop_back`.
- **Problems found in `DynamicArrayKokkos`:**
  1. **No `resize()`:** There is no method to grow the backing buffer beyond the capacity set at construction. Users who need to exceed the initial allocation must destroy and reconstruct the object, triggering a device deallocation and a fresh device allocation. `Kokkos::resize(view, new_size)` handles this efficiently: it no-ops if the new size fits within the current extent and deep-copies into a new allocation only when growth is needed.
  2. **`push_back` is a kernel launch per element:** `push_back(T value)` dispatches `Kokkos::parallel_for(..., 1, ...)` to write a single scalar. This is the most expensive way to do a scalar store on a GPU. On CUDA, a single-thread kernel launch has overhead of ~5–15 µs; for an adaptive algorithm that pushes thousands of elements one at a time, this dominates runtime.
  3. **2D–7D constructors and `operator()` overloads are commented out:** The class declaration includes 2D–7D signatures but they are all wrapped in `/* ... */`. `DynamicArrayKokkos` is therefore 1D-only. Any multi-dimensional dynamic use case falls back to manual flat indexing or reconstructing a `CArrayKokkos`.
- **Fix:**
  1. Add a `void resize(size_t new_capacity)` host method:
     ```cpp
     void resize(size_t new_capacity) {
         Kokkos::resize(this_array_, new_capacity);
         dims_[0] = new_capacity;
         length_ = new_capacity;
         if (dims_actual_size_[0] > new_capacity)
             dims_actual_size_[0] = new_capacity;
     }
     ```
  2. Add a `void push_back_host(T value)` overload that writes directly from host memory (via a `Kokkos::View<T*, Kokkos::HostSpace>` mirror copy or by operating only when the execution space is a host space), and document that the `push_back(T value)` GPU path is only appropriate for device-side lambdas.
  3. Uncomment the 2D–7D constructors and `operator()` bodies, then extend `resize()` to the multi-dimensional case — `Kokkos::resize` accepts up to 8 extents for multi-rank views and the flat 1D backing `View<T*>` makes this straightforward.
- **Priority:** Medium — item 1 (resize) is a clear gap; items 2–3 improve usability before v1.

### 4.5 `DViewCArrayKokkos` and related types do not warn on missing sync in debug builds
- **File:** `src/include/kokkos_types.h` (DView types)
- **Problem:** MATAR's DView wrappers expose `update_host()` and `update_device()` but there is no assertion or warning if a user accesses host data after modifying the device copy without calling `update_host()`. Silent stale-data reads are the most common MATAR user bug.
- **Fix:** In debug builds (`NDEBUG` not defined), add assertions in `operator()` checking the DualView's modification flags. Kokkos `DualView` already tracks these via `modified_host()` and `modified_device()`; MATAR should expose them in `assert()` calls on the host-side accessors.
- **Priority:** Medium.

### 4.6 Benchmark suite is not in CI
- **File:** `benchmark/`, `.github/workflows/`
- **Problem:** A benchmark suite exists (`benchmark/src/CArray_benchmark.cpp`, `CArrayDevice_benchmark.cpp`) but is never run in CI. Performance regressions will not be caught before v1.
- **Fix:** After §5.4 adds `Matar_BUILD_BENCHMARKS` as a root CMake option, add a `benchmark.yml` CI job that uses the `serial` preset and runs with `--benchmark_min_time=0` to validate compilation and execution without timing comparisons.
- **Priority:** Medium (depends on §5.4).

### 4.7 Unqualified `deep_copy` calls in DView types rely on ADL
- **File:** `src/include/kokkos_types.h` (~lines 2532, 3426, 5809, 6699)
- **Problem:** `DViewFArrayKokkos::update_host()` and `update_device()` call `deep_copy(...)` without `Kokkos::` qualification. ADL finds `Kokkos::deep_copy` through the Kokkos namespace, but if any header upstream defines a `deep_copy` in an associated namespace, the wrong one is silently called.
- **Fix:** Qualify all calls as `Kokkos::deep_copy(...)`.
- **Priority:** Low.

### 4.8 Serial `reduce_sum` / `reduce_min` / `reduce_max` pass the accumulator by value
- **File:** `src/include/macros.h:1104-1242`
- **Problem:** The serial fallback functions take `T var` by value and reset it internally (`var = 0`). The original variable at the call site is never written. This pattern is surprising for readers and wastes a copy for non-trivial `T`.
- **Fix:** Remove the `var` parameter and declare the accumulator locally:
  ```cpp
  template <typename T, typename F>
  void reduce_sum(int i_start, int i_end, const F &lambda_fcn, T &result) {
      T var = T{0};
      for(int i=i_start; i<i_end; i++) lambda_fcn(i, var);
      result = var;
  }
  ```
  The non-Kokkos macros must be updated to match the new signature. Document as a minor ABI change in `MIGRATION.md` (§3.8).
- **Priority:** Low.

### 4.9 `reduce_prod` has a `// MIN` comment above it
- **File:** `src/include/macros.h:1247`
- **Problem:** The comment directly above `reduce_prod` says `// MIN` — a copy-paste error from the `reduce_min` section. Also verify `FOR_REDUCE_PRODUCT` is tested (grep test files for the macro).
- **Fix:** Fix the comment to `// PRODUCT`. Add a test if one doesn't exist.
- **Priority:** Low.

### 4.10 Host-type index arithmetic — verify no `int` overflow path for large arrays
- **File:** `src/include/host_types.h` (throughout)
- **Problem:** The `FArray`, `CArray`, etc. constructors take `size_t` dims and `operator()` arguments are `size_t`, so the index products should be `size_t` arithmetic and overflow-safe. However, this should be explicitly audited — intermediate casts or int-typed temporaries in the flat-index computation could silently truncate for arrays larger than 2 GB.
- **Fix:** Audit all `operator()` implementations. Add `static_assert(sizeof(size_t) >= 8, "MATAR requires 64-bit size_t")` at the top of `host_types.h` to catch 32-bit platform builds early.
- **Priority:** Medium.

### 4.11 `MATAR_KOKKOS_INIT` / `MATAR_KOKKOS_FINALIZE` are exception-unsafe
- **File:** `src/include/macros.h:908-915`
- **Problem:** The manual `Kokkos::initialize` / `Kokkos::finalize` pair leaves Kokkos un-finalized if application code throws between them, potentially leaking GPU contexts.
- **Fix:** Add a recommended `MATAR_KOKKOS_SCOPE_GUARD` macro alongside the existing ones:
  ```cpp
  #define MATAR_KOKKOS_SCOPE_GUARD Kokkos::ScopeGuard _kokkos_sg(argc, argv);
  ```
  Keep the old macros for backward compatibility. Reference `MATAR_KOKKOS_SCOPE_GUARD` in the GPU quick-start added to README (§3.5).
- **Priority:** Low.

---

## 5. Build System Refactor (Primary v1 Goal)

> **Goal:** Replace the `scripts/build-matar.sh` bash wrapper system entirely with a pure CMake build. The target workflow is:
> ```
> cmake --preset cuda-mpi   # configure
> cmake --build --preset cuda-mpi   # build
> ctest --preset cuda-mpi   # test
> ```
> This is a prerequisite for: Windows portability, IDE integration (VS Code CMake Tools, CLion), removing the machine/bash dependency, and fixing the CI issues listed in §1.6. Items 5.1–5.4 are ordered as a dependency chain; complete them in sequence.

### 5.1 Add `CMakePresets.json` — the new top-level build interface
- **Files (new):** `CMakePresets.json` at repo root
- **Replaces:** `scripts/build-matar.sh`, `scripts/cmake_build_test.sh`, `scripts/cmake_build_examples.sh`, `scripts/setup-env.sh`
- **Design:** One base preset (`base`) with shared defaults; all other presets `inherit` from it. Each backend gets a configure preset; build and test presets inherit the configure preset:
  ```json
  {
    "version": 6,
    "configurePresets": [
      {
        "name": "base",
        "hidden": true,
        "generator": "Ninja",
        "binaryDir": "${sourceDir}/build-${presetName}",
        "cacheVariables": {
          "CMAKE_CXX_STANDARD": "17",
          "Matar_ENABLE_KOKKOS": "ON"
        }
      },
      { "name": "serial",       "inherits": "base", "displayName": "Serial CPU" },
      { "name": "serial-debug", "inherits": "serial",
        "cacheVariables": { "CMAKE_BUILD_TYPE": "Debug", "Matar_BUILD_TESTS": "ON" }},
      { "name": "openmp",       "inherits": "base",
        "cacheVariables": { "Kokkos_ENABLE_OPENMP": "ON" }},
      { "name": "cuda",         "inherits": "base",
        "cacheVariables": { "Kokkos_ENABLE_CUDA": "ON", "CMAKE_CUDA_ARCHITECTURES": "80" }},
      { "name": "hip",          "inherits": "base",
        "cacheVariables": { "Kokkos_ENABLE_HIP": "ON" }},
      { "name": "serial-mpi",   "inherits": "serial",
        "cacheVariables": { "Matar_ENABLE_MPI": "ON" }},
      { "name": "openmp-mpi",   "inherits": "openmp",
        "cacheVariables": { "Matar_ENABLE_MPI": "ON" }},
      { "name": "cuda-mpi",     "inherits": "cuda",
        "cacheVariables": { "Matar_ENABLE_MPI": "ON" }}
    ],
    "buildPresets": [
      { "name": "serial",       "configurePreset": "serial" },
      { "name": "serial-debug", "configurePreset": "serial-debug" },
      { "name": "cuda",         "configurePreset": "cuda" }
    ],
    "testPresets": [
      { "name": "serial",       "configurePreset": "serial-debug", "output": { "outputOnFailure": true } },
      { "name": "serial-mpi",   "configurePreset": "serial-mpi",  "output": { "outputOnFailure": true } }
    ]
  }
  ```
  Note: machine-specific paths (GPU architecture, MPI install prefix) belong in a `CMakeUserPresets.json` (gitignored) that inherits from the shared presets.
- **Priority:** High — all other build system items depend on this.

### 5.2 Fold Kokkos dependency into root CMakeLists via `FetchContent`
- **Files:** `CMakeLists.txt`, `src/Kokkos/kokkos/` (existing submodule)
- **Replaces:** `scripts/kokkos-install.sh`, `scripts/trilinos-install.sh` (Trilinos path remains manual for now — Trilinos is too large for FetchContent)
- **Design:** Use a find-then-fetch pattern so power users with an existing Kokkos install are not forced to rebuild it:
  ```cmake
  find_package(Kokkos QUIET)
  if(NOT Kokkos_FOUND)
    message(STATUS "Kokkos not found — building from submodule src/Kokkos/kokkos")
    set(FETCHCONTENT_SOURCE_DIR_KOKKOS ${CMAKE_SOURCE_DIR}/src/Kokkos/kokkos)
    include(FetchContent)
    FetchContent_Declare(kokkos SOURCE_DIR ${FETCHCONTENT_SOURCE_DIR_KOKKOS})
    FetchContent_MakeAvailable(kokkos)
  endif()
  ```
  The existing `src/Kokkos/kokkos` git submodule remains the version-pinned source. Delete `scripts/kokkos-install.sh` once this is verified.
- **Priority:** High (depends on §5.1).

### 5.3 Update CI workflows to use `cmake --preset`
- **Files:** `.github/workflows/cmake.yml`, `.github/workflows/test.yml`
- **Replaces:** All `source build-matar.sh ...` steps in CI
- **Design:** Replace the multi-step bash script invocation with standard CMake preset calls:
  ```yaml
  - name: Configure
    run: cmake --preset ${{ matrix.preset }}
  - name: Build
    run: cmake --build --preset ${{ matrix.preset }}
  - name: Test
    run: ctest --preset ${{ matrix.preset }} --output-on-failure
  ```
  Matrix entries become `preset: [serial, serial-debug, openmp, serial-mpi]`. The macOS matrix uses the same presets (no `--machine` flag). Delete `cmake.yml` and consolidate into a single `test.yml`. This resolves §1.6 (macOS CI misconfiguration) and §1.6 (commented-out ctest) automatically.
- **Priority:** High (depends on §5.1 and §5.2).

### 5.4 Fold test, example, and benchmark builds into root CMakeLists as optional subdirs
- **Files:** `CMakeLists.txt`, `test/CMakeLists.txt`, `examples/CMakeLists.txt`, `benchmark/CMakeLists.txt`
- **Replaces:** The standalone project structure that requires installing MATAR before building tests
- **Design:** Add to root `CMakeLists.txt`:
  ```cmake
  option(Matar_BUILD_TESTS      "Build unit tests"    OFF)
  option(Matar_BUILD_EXAMPLES   "Build examples"      OFF)
  option(Matar_BUILD_BENCHMARKS "Build benchmarks"    OFF)
  if(Matar_BUILD_TESTS)      add_subdirectory(test)      endif()
  if(Matar_BUILD_EXAMPLES)   add_subdirectory(examples)  endif()
  if(Matar_BUILD_BENCHMARKS) add_subdirectory(benchmark) endif()
  ```
  Update each subdirectory's `CMakeLists.txt` to support both standalone and in-tree use:
  ```cmake
  if(NOT TARGET matar)
      find_package(Matar REQUIRED)
  endif()
  ```
  Remove the non-standard `-DCUDA=ON` / `-DKOKKOS=ON` variables from `test/CMakeLists.txt` (resolved by §1.2 — backend macros flow from the `matar` target transitively). Enable `Matar_BUILD_TESTS=ON` in the `serial-debug` preset.
- **Priority:** High (depends on §5.1; also resolves §1.7 and §4.6).

### 5.5 Replace `scripts/machines/` compiler paths with CMake toolchain files
- **Files:** `scripts/machines/mac-env.sh`, `scripts/machines/linux-env.sh`, `scripts/machines/darwin-env.sh` (new: `cmake/toolchains/`)
- **Replaces:** Hardcoded `/opt/homebrew/opt/llvm/bin/clang` and `/usr/bin/gcc` paths in shell env scripts
- **Design:** Create `cmake/toolchains/` with one file per target environment (e.g., `darwin-cluster.cmake`, `mac-homebrew-llvm.cmake`). Users pass `-DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/darwin-cluster.cmake` or set `toolchainFile` in a `CMakeUserPresets.json`. This is the standard CMake pattern for cross-compilation and environment-specific compilers.
- **Priority:** Medium (depends on §5.1).

### 5.6 Archive `scripts/` as legacy — do not delete immediately
- **Files:** `scripts/`
- **Plan:** After §5.1–§5.4 are complete and the CI is green on the new system, rename `scripts/` to `scripts/legacy/`. Add `scripts/legacy/README.md` explaining that these scripts are archived for reference and pointing to `CMakePresets.json`. Leave the files intact — they are useful for reproducing HPC environment-specific edge cases. Delete in v1.1 once no downstream users report dependency on them.
- **Priority:** Low (final step after §5.1–§5.4).

---

## Summary Table

| # | Category | Item | Priority |
|---|---|---|---|
| **5.1** | **Build** | **Add `CMakePresets.json` — new top-level build interface** | **High** |
| **5.2** | **Build** | **Fold Kokkos into root CMake via `FetchContent`** | **High** |
| **5.3** | **Build** | **Update CI to `cmake --preset` (resolves §1.6)** | **High** |
| **5.4** | **Build** | **Fold test/example/benchmark into root CMakeLists (resolves §1.7)** | **High** |
| 5.5 | Build | Replace machine env scripts with CMake toolchain files | Medium |
| 5.6 | Build | Archive `scripts/` as legacy after §5.1–§5.4 | Low |
| 1.1 | OS | `matar.h` unconditional MPI/Tpetra includes + angle-bracket include | High |
| 1.2 | OS | Backend macros not propagated by CMake install | High |
| 1.3 | OS | HIP uses deprecated `Kokkos::Experimental` namespace | High |
| 1.4 | OS | `_old` files installed alongside live headers | Low |
| 1.5 | OS | No `MATAR_VERSION` macro | Medium |
| 1.6 | OS | CI macOS `--machine=linux` + ctest not run (resolved by §5.3) | High |
| 1.7 | OS | `test/CMakeLists.txt` non-standard backend variables (resolved by §5.4 + §1.2) | Medium |
| 1.8 | OS | No macOS MPI CI | Low |
| 2.1 | GPU | `FOR_FIRST`/`DO_FIRST` hardcode warp size 32 | High |
| 2.2 | GPU | `F_LOOP_ORDER` wrong direction for GPU F-array traversal | High |
| 2.3 | GPU | `policy2D/3D/4D` don't bind `DefaultExecSpace` | Medium |
| 2.4 | GPU | `real_t`/`u_int` in global namespace — POSIX collision | High |
| 2.5 | GPU | `sone_dim2` typo in FArrayKokkos 6D/7D constructors | Medium |
| 2.6 | GPU | FOR_ALL loop indices use `const int` — limits to 2^31 elements | Medium |
| 2.7 | GPU | CArrayKokkos default `LayoutLeft` on CUDA contradicts C-order semantics | Medium |
| 2.8 | GPU | No SYCL backend path | Low |
| 2.9 | GPU | No CMake assertion that Kokkos has the requested backend | Medium |
| 3.1 | Docs | `host_types.h` zero Doxygen across 5400+ lines | High |
| 3.2 | Docs | MPI types have no class/method documentation | High |
| 3.3 | Docs | No Doxygen for 4D–7D constructors in Kokkos types | Medium |
| 3.4 | Docs | README typos and broken pseudocode | Medium |
| 3.5 | Docs | README has no GPU or MPI quick-start | Medium |
| 3.6 | Docs | No `PartitionMap`/`CommunicationPlan` usage example | High |
| 3.7 | Docs | No CHANGELOG | High |
| 3.8 | Docs | No v1 migration guide (especially for build system change) | Medium |
| 3.9 | Docs | No type-selection guide | Medium |
| 3.10 | Docs | `macros.h` header shows wrong macro names | Low |
| 3.11 | Docs | Sphinx/Doxygen not built in CI | Low |
| 4.1 | Perf | `set_values()` uses kernel launch instead of `Kokkos::deep_copy` | Medium |
| 4.2 | Perf | `CommunicationPlan` displacement setup is O(n²) | Medium |
| 4.3 | Perf | Redundant GPU fences in `CommunicationPlan` init | Low |
| 4.4 | Perf | `DynamicArrayKokkos`: no `resize()`, costly `push_back`, 2D–7D commented out | Medium |
| 4.5 | Perf | DView types have no sync-state assertions in debug builds | Medium |
| 4.6 | Perf | Benchmark suite not in CI (depends on §5.4) | Medium |
| 4.7 | Perf | Unqualified `deep_copy` calls in DView types | Low |
| 4.8 | Perf | Serial reduce accumulator passed by value | Low |
| 4.9 | Perf | `reduce_prod` has wrong `// MIN` comment | Low |
| 4.10 | Perf | Audit host-type index arithmetic for `int` overflow | Medium |
| 4.11 | Perf | `MATAR_KOKKOS_INIT` is exception-unsafe | Low |

---

## v1 Blocking Items

Items that must be complete before tagging 1.0, grouped by theme.

### Build System (do first — everything else depends on it)
1. **5.1** Add `CMakePresets.json`
2. **5.2** Fold Kokkos into CMake via `FetchContent`
3. **5.3** Update CI to `cmake --preset` (also fixes §1.6)
4. **5.4** Fold test/example/benchmark into root CMakeLists (also fixes §1.7)

### Correctness / Portability (parallel with build work)
5. **1.3** HIP deprecated `Kokkos::Experimental` — breaks Frontier builds today
6. **2.1** `FOR_FIRST` hardcoded warp size 32 — wrong on AMD HIP
7. **2.2** `F_LOOP_ORDER` wrong direction — non-coalesced GPU access for all FArray types
8. **2.4** `real_t`/`u_int` global namespace — compile failure with POSIX headers on Linux
9. **1.1** `matar.h` unconditional MPI/Tpetra includes — breaks clean installs
10. **1.2** Backend macros not propagated — wrong layout for OpenMP

### Documentation (required for a usable v1 release)
11. **3.1** `host_types.h` zero Doxygen — most-used types in the library
12. **3.2** MPI types have no documentation — new API, cannot be used without docs
13. **3.6** No `PartitionMap`/`CommunicationPlan` usage example
14. **3.7** No CHANGELOG
