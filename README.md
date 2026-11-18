# MATAR
<p align="center"><img src="https://github.com/lanl/MATAR/blob/main/MATAR-logo.png" width="350">

MATAR (LANL open-source code number C23032) is a C++ library that addresses the need for simple, fast, and memory-efficient multi-dimensional data representations for dense and sparse storage that arise with numerical methods and in software applications. The data representations are designed to perform well across multiple computer architectures, including CPUs and GPUs. MATAR allows users to easily create and use intricate data representations that are also portable across disparate architectures using Kokkos. The performance aspect is achieved by forcing contiguous memory layout (or as close to contiguous as possible) for multi-dimensional and multi-size dense or sparse MATrix and ARray (hence, MATAR) types. Results show that MATAR has the capability to improve memory utilization, performance, and programmer productivity in scientific computing. This is achieved by fitting more work into the available memory, minimizing memory loads required, and by loading memory in the most efficient order. 


## Examples
* [ELEMENTS](https://github.com/lanl/ELEMENTS/):   MATAR is a part of the ELEMENTS Library (LANL C# C20058) and it underpins the routines implemented in ELEMENTS.  MATAR is available in a stand-alone directory outside of the ELEMENTS directory because it can aid many code applications.  The dense and sparse storage types in MATAR are the foundation for the ELEMENTS library, which contains mathematical functions to support a very broad range of element types including: linear, quadratic, and cubic serendipity elements in 2D and 3D; high-order spectral elements; and a linear 4D element. An unstructured high-order mesh class is available in ELEMENTS and it takes advantage of MATAR for efficient access of various mesh entities. 

* [Fierro](https://github.com/lanl/Fierro): The MATAR library underpins the Fierro code that is designed to simulate quasi-static solid mechanics problems and material dynamics problems.  
    
* Simple examples are in the [`examples/`](examples) folder, and step-by-step tutorials are in [`tutorial/getting_started/`](tutorial/getting_started)

## Descriptions

* All Array MATAR types (e.g., CArray, ViewCArray, FArray, RaggedRightArray, etc.) start with an index of 0 and stop at an index of N-1, where N is the number of entries.  

* All Matrix MATAR types  (e.g., CMatrix, ViewCMatrix, FMatrix, etc.)  start with an index of 1 and stop at an index of N, where N is the number of entries. 

* The MATAR View types (e.g., ViewCArray, ViewCMatrix, ViewFArray, etc. ) are designed to accept a pointer to an existing 1D array and then access that 1D data as a multi-dimensional array.  The MATAR View types can also be used to slice an existing View.  

* The C dense storage and View types (e.g., CArray, ViewCArray, CMatrix, etc.) access the data following the C/C++ language convection of having the last index in a multi-dimensional array vary the quickest.  In a 2D CArray A, the index j in A(i,j) varies first followed by the index i, so the optimal performance is achieved using the following loop ordering.
```
// Optimal use of CArray
for (i=0,i<N,i++){
    for (j=0,j<N,j++){
        A(i,j) = 0.0;
    }
}
```

* The F dense storage and View types (e.g., FArray, ViewFArray, FMatrix, etc.) access the data following the Fortran language convection of having the first index in a multi-dimensional array vary the quickest.  In a 2D FMatrix M, the index i in M(i,j) varies first followed by the index j, so the optimal performance is achieved using the following loop ordering.

```
// Optimal use of FMatrix
for (j=1,j<=N,j++){
    for (i=1,i<=N,i++){
        M(i,j) = 0.0;
    }
}
```

* The ragged data types (e.g., RaggedRightArray, RaggedDownArray, etc) in MATAR are special sparse storage types.  The Right access types are for R(i,j) where the number of column entries varies in width across the array.  The Down access types are for D(i,j) where the number of row entries vary in length across the array.

* The SparseRowArray MATAR type is the idetical to the Compressed Sparse Row (CSR) or Compressed Row Storage (CSR) respresentation.

* The SparseColumnArray MATAR type is identical to the Compressed Sparse Column (CSC) or Compressed Column Storage (CCS) respresentation.


## Usage
```
// create a 1D array of integers and then access as a 2D array
int A[9];
auto A_array = ViewCArray <int> (A, 3, 3); // access as A(i,j) 

// create a 3D array of doubles
auto B = CArray <double> (3,3,3); // access as B(i,j,k)

// create a slice of the 3D array at index 1
auto C = ViewCArray <double> (&B(1,0,0),3,3); // access as C(j,k)


// create a 4D matrix of doubles, indices start at 1 
auto D = CMatrix <double> (10,9,8,7); // access as D(i,j,k,l)


// create a 2D view of a standard array
std::array<int, 9> E1d;
auto E = ViewCArray<int> (&E1d[0], 3, 3);
E(0,0) = 1;  // and so on


// create a ragged-right array of integers
//
// [1, 2, 3]
// [4, 5]
// [6]
// [7, 8, 9, 10]
//
size_t my_strides[4] = {3, 2, 1, 4};
RaggedRightArray <int> ragged(my_strides, 4);
    
int value = 1;
for (int i=0; i<4; i++){
    for (int j=0; j<my_ragged.stride(i); j++){
        ragged(i,j) = value;
        value++;
    }
}


```
More information about the capabilities and usage of MATAR can be found in this presentation [here](https://www.researchgate.net/publication/360744549_General_purpose_GPU_programming_made_easy).
    
## Building MATAR

MATAR is a header-only library, so "building MATAR" means two things: configuring the bundled Kokkos for the backend you want (serial, OpenMP, CUDA, ...), and compiling the examples, unit tests, and benchmarks in this repository. If you only want to *use* MATAR in your own code, skip to [Using MATAR as a third-party library](#using-matar-as-a-third-party-library); your project's CMake build will pull MATAR in.

The whole build is CMake. Kokkos is bundled as a git submodule (`src/Kokkos/kokkos`, pinned to Kokkos **5.2.1**) and is compiled automatically with the backend you choose, so you do not need to install Kokkos separately.

### Prerequisites

| Requirement | Version / notes |
|---|---|
| CMake | **3.22 or newer**, or **3.25.2 or newer** for CUDA builds (needed for C++20 in the CUDA language). Check with `cmake --version`. |
| C++ compiler | Must support **C++20**, because Kokkos 5 requires it (recent GCC, Clang, or a vendor compiler). |
| git | Used to clone the repository and its Kokkos submodule. |
| Internet access at configure time | Needed only when tests or benchmarks are enabled, since CMake downloads GoogleTest / Google Benchmark with FetchContent. All presets enable tests. |
| MPI (optional) | Needed for the `*-mpi` presets, for example OpenMPI or MPICH. `mpicxx`/`mpirun` must be on your `PATH`. |
| CUDA toolkit / ROCm (optional) | Needed for the `cuda*` and `hip*` presets. `nvcc` (CUDA) or `hipcc` (HIP) must be on your `PATH`. |

On HPC systems, load your modules **before** configuring, for example `module load cmake gcc cuda openmpi`. CMake records the compilers it finds on the first configure, so loading a module afterwards has no effect until you reconfigure from scratch (see [Rebuilding and starting over](#rebuilding-and-starting-over)).

### Step 1: Get the source

Run this from whichever directory you want the `MATAR/` folder created in:

```bash
git clone --recursive https://github.com/lanl/MATAR.git    # or: ssh://git@github.com/lanl/MATAR.git
cd MATAR
```

`--recursive` is required because it also downloads the Kokkos submodule. If you already cloned without it, or `src/Kokkos/kokkos` is empty, run this from inside `MATAR/`:

```bash
git submodule update --init --recursive
```

Without the submodule, configuring stops with an error that tells you to run that command.

### Step 2: Configure and build (quick start)

**Run every command below from the MATAR repository root**, the directory that contains `CMakePresets.json`. `cmake --preset` and `cmake --build --preset` look for `CMakePresets.json` in the current directory and fail with `Could not read presets` anywhere else.

```bash
cd MATAR                          # the repository root

cmake --preset serial             # 1. configure  -> creates build/serial/
cmake --build --preset serial -j  # 2. compile Kokkos, the examples, and the tests
```

After this:

* All build output is under `build/serial/`, and the compiled executables are in `build/serial/bin/`. Nothing is written into the source tree.
* The first build compiles all of Kokkos, so it takes a few minutes. Later builds only recompile what changed.

Replace `serial` with any preset name from the table below. Every preset uses its own `build/<preset-name>/` directory, so several backends can be built side by side.

> **Not at the repository root?** Presets need the root, but the non-preset form works from anywhere if you give the build directory explicitly:
> ```bash
> cmake --build /path/to/MATAR/build/serial -j
> ```

### Step 3: Test with CTest

Once a build has finished, run its unit tests from the repository root with the test preset of the same name:

```bash
ctest --preset serial
```

See [Testing with CTest](#testing-with-ctest) for running tests for every backend, filtering tests, and running them in parallel.

### Choosing a preset

Run `cmake --list-presets` from the repository root to print every preset. All presets build the examples and unit tests, use `CMAKE_BUILD_TYPE=Release` unless the name ends in `-debug`, and set `Kokkos_ARCH_NATIVE=ON` to optimize for the CPU you are compiling on.

| Preset | Backend(s) | Use it for |
|---|---|---|
| `serial` | Kokkos Serial | First build, debugging, machines with no GPU |
| `openmp` | Kokkos OpenMP | Multi-core CPU parallelism |
| `pthreads` | Kokkos `std::thread` | Multi-core CPU parallelism without OpenMP |
| `cuda` | Kokkos CUDA | NVIDIA GPUs (needs `nvcc`) |
| `hip` | Kokkos HIP | AMD GPUs (needs `hipcc`, which the preset sets as the C++ compiler) |
| `serial-mpi`, `openmp-mpi`, `cuda-mpi`, `hip-mpi` | Same as above, plus MPI | Distributed-memory runs (`MPICArrayKokkos`, `CommunicationPlan`) |
| `serial-debug`, `openmp-debug`, `cuda-debug`, `hip-debug`, `serial-mpi-debug` | Debug builds | Array dimension and index-bounds checks, debug symbols |
| `serial-fp32`, `serial-fp16`, `serial-bf16`, `serial-quad`, `serial-mixed` | Serial with a different `real_t` | Testing reduced or extended precision (see [Precision tiers](#precision-tiers)) |
| `serial-mpi-fp16`, `serial-mpi-quad` | Serial + MPI with a different `real_t` | Testing MPI at reduced or extended precision |
| `hostomp` | OpenMP for both host and device macros | CPU-only use of the host/device backend options |
| `cuda-hostomp` | CUDA device + OpenMP host | Overlapping CPU work with GPU kernels (see [Host-side parallelism](#host-side-parallelism)) |

The configure, build, and test presets share names, so the same word goes after `cmake --preset`, `cmake --build --preset`, and `ctest --preset`.

### Build commands for every backend

Each block below configures and builds one backend. **Run every block from the repository root.** Each backend builds into its own `build/<preset>/` directory, so you can build any combination of them side by side. `-j` builds in parallel on all cores. To test a build, see [Testing with CTest](#testing-with-ctest).

**Serial** (no parallelism; runs anywhere):
```bash
cmake --preset serial
cmake --build --preset serial -j
```

**OpenMP** (multi-core CPU):
```bash
cmake --preset openmp
cmake --build --preset openmp -j
```

**pthreads** (multi-core CPU through `std::thread`):
```bash
cmake --preset pthreads
cmake --build --preset pthreads -j
```

**CUDA** (NVIDIA GPU; `nvcc` on `PATH`, CMake >= 3.25.2):
```bash
module load cuda                  # if your system uses modules
cmake --preset cuda
cmake --build --preset cuda -j
```

**HIP** (AMD GPU; `hipcc` on `PATH`):
```bash
module load rocm                  # if your system uses modules
cmake --preset hip
cmake --build --preset hip -j
```

For CUDA and HIP, if no GPU is visible while configuring (for example on a login node), you also need to name the GPU architecture. See [GPU architecture](#gpu-architecture).

**CUDA device + OpenMP host** (GPU kernels overlapping CPU work through the `_HOST` macros):
```bash
cmake --preset cuda-hostomp
cmake --build --preset cuda-hostomp -j
```

**OpenMP for both host and device macros** (CPU only):
```bash
cmake --preset hostomp
cmake --build --preset hostomp -j
```

**With MPI.** Any of the serial, OpenMP, CUDA, or HIP backends gets MPI by adding `-mpi` to the preset name. CMake finds MPI with `find_package(MPI)`, so `mpicxx` must be on `PATH` (or pass `-DMPI_HOME=<prefix>` when configuring):
```bash
module load mpi/openmpi-x86_64    # module name varies by system
cmake --preset serial-mpi && cmake --build --preset serial-mpi -j
cmake --preset openmp-mpi && cmake --build --preset openmp-mpi -j
cmake --preset cuda-mpi   && cmake --build --preset cuda-mpi   -j
cmake --preset hip-mpi    && cmake --build --preset hip-mpi    -j
```
If your MPI can send and receive GPU buffers directly (CUDA-aware or ROCm-aware MPI), add `-DMATAR_ENABLE_GPU_AWARE_MPI=ON` when configuring a `cuda-mpi` or `hip-mpi` build.

**Debug builds** (bounds and dimension checks). Add `-debug` to the name: `serial-debug`, `openmp-debug`, `cuda-debug`, `hip-debug`, `serial-mpi-debug`:
```bash
cmake --preset serial-debug
cmake --build --preset serial-debug -j
```

**Precision variants.** `serial-fp32`, `serial-fp16`, `serial-bf16`, `serial-quad`, `serial-mixed`, `serial-mpi-fp16`, and `serial-mpi-quad` follow the same pattern:
```bash
cmake --preset serial-fp32
cmake --build --preset serial-fp32 -j
```

**No Kokkos** (dependency-free serial header library). No preset covers this, so configure by hand:
```bash
cmake -S . -B build/nokokkos -DMATAR_ENABLE_KOKKOS=OFF -DMATAR_BUILD_TESTS=ON -DMATAR_BUILD_EXAMPLES=ON
cmake --build build/nokokkos -j
```

**Several backends in one go.** Loop over the presets you want. For example, to build all the CPU backends:
```bash
for p in serial openmp pthreads; do
    cmake --preset $p && cmake --build --preset $p -j || break
done
```
Add `cuda`, `hip`, or any `*-mpi` preset to the list on machines that have those toolchains.

**Add or override an option on top of a preset.** Put `-D` flags after the preset name when you configure:
```bash
cmake --preset openmp -DMATAR_BUILD_BENCHMARKS=ON
cmake --preset serial -DCMAKE_CXX_COMPILER=clang++
```
These values are saved in `build/<preset>/CMakeCache.txt` and kept on later configures. The full option list is in [CMake options reference](#cmake-options-reference).

**Build a single target instead of everything:**
```bash
cmake --build --preset serial --target matar_tests -j
```

### Testing with CTest

All testing goes through CTest. Build first, then run `ctest --preset <name>` from the repository root, using the same preset name as the build. `ctest` runs the GoogleTest unit tests and prints the output of any test that fails. Tests are discovered when `ctest` runs, not during the build, so building never runs the test binaries.

**Test each backend:**
```bash
ctest --preset serial
ctest --preset openmp
ctest --preset pthreads
ctest --preset cuda
ctest --preset hip
ctest --preset cuda-hostomp
ctest --preset hostomp
ctest --preset serial-mpi         # likewise openmp-mpi, cuda-mpi, hip-mpi
ctest --preset serial-debug       # likewise the other -debug presets
ctest --preset serial-fp32        # likewise the other precision presets
```

**Builds configured without a preset** (SYCL, no Kokkos, or your own `-B` directory) have no test preset. Point `ctest` at the build directory instead. This form also works from any directory if you give the full path:
```bash
ctest --test-dir build/sycl --output-on-failure
ctest --test-dir build/nokokkos --output-on-failure
```

**MPI.** In `*-mpi` builds, `ctest` also runs the MPI test suite (`MPICArrayKokkos_mpi_suite`) on 4 ranks as `mpirun --oversubscribe -n 4`, so it works even on a machine with fewer than 4 cores. Load the same MPI module you built with before running `ctest`.

**Thread count.** For the `openmp`, `pthreads`, and `hostomp` builds, set the number of threads the tests use with an environment variable:
```bash
OMP_NUM_THREADS=8 ctest --preset openmp
KOKKOS_NUM_THREADS=8 ctest --preset pthreads
```

**Select, list, and parallelize tests:**
```bash
ctest --preset serial -R CArrayKokkos      # only tests whose name matches the regex
ctest --preset serial -E MPI               # exclude tests whose name matches the regex
ctest --preset serial -N                   # list the tests without running them
ctest --preset serial -j 8                 # run up to 8 tests at a time
ctest --preset serial --rerun-failed       # rerun only the tests that failed last time
ctest --preset serial -V                   # verbose: show the output of every test
```

**Build and test several backends in one go:**
```bash
for p in serial openmp pthreads; do
    cmake --preset $p && cmake --build --preset $p -j && ctest --preset $p || break
done
```

### Rebuilding and starting over

* **After editing source files:** rerun only `cmake --build --preset <name> -j`. CMake reconfigures itself if a `CMakeLists.txt` changed. The one exception: if a `CMakeLists.txt` adds a new target and you build it by name with `--target`, run `cmake --preset <name>` first. Otherwise the build tool reports `No rule to make target`.
* **After changing compilers, loading different modules, or switching backends in an existing build directory:** start from a clean configuration. CMake caches compiler paths and Kokkos backend settings, and a stale cache can cause confusing errors, including MATAR's `Kokkos's default execution space will be ...` check.
  ```bash
  cmake --fresh --preset serial      # CMake >= 3.24: discard the cache and reconfigure
  # or, with any CMake version:
  rm -rf build/serial && cmake --preset serial
  ```
* **Remove every build:** `rm -rf build/`

### GPU architecture

Kokkos tries to detect the GPU on the machine you configure on. If you build on a login node with no GPU, or for a different GPU than the one installed, give the architecture explicitly when configuring:

```bash
cmake --preset cuda -DKokkos_ARCH_AMPERE80=ON     # A100.  Others: VOLTA70, HOPPER90, ADA89, ...
cmake --preset hip  -DKokkos_ARCH_AMD_GFX90A=ON   # MI250X. Others: AMD_GFX942 (MI300), ...
```

Likewise, the presets turn on `Kokkos_ARCH_NATIVE`, which optimizes for the **CPU you compile on**. If your compute nodes have a different CPU than the login node, turn it off and name the target CPU instead, for example `-DKokkos_ARCH_NATIVE=OFF -DKokkos_ARCH_ZEN3=ON`. The [Kokkos CMake keywords documentation](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html) lists every `Kokkos_ARCH_*` value.

### Site-specific presets (`CMakeUserPresets.json`)

To reuse your own settings (a particular compiler, GPU architecture, and so on), create `CMakeUserPresets.json` in the repository root next to `CMakePresets.json`. It is meant to stay local and should not be committed. It can inherit from any preset in `CMakePresets.json`:

```json
{
    "version": 3,
    "configurePresets": [
        {
            "name": "my-cuda",
            "inherits": "cuda",
            "cacheVariables": {
                "Kokkos_ARCH_HOPPER90": "ON",
                "CMAKE_CXX_COMPILER": "g++-13"
            }
        }
    ],
    "buildPresets": [ { "name": "my-cuda", "configurePreset": "my-cuda" } ],
    "testPresets":  [ { "name": "my-cuda", "configurePreset": "my-cuda", "output": { "outputOnFailure": true } } ]
}
```

Then use `cmake --preset my-cuda`, `cmake --build --preset my-cuda -j`, and `ctest --preset my-cuda`. The output goes to `build/my-cuda/`.

### Configuring without presets

Presets are a shortcut for ordinary `-D` flags, so you can call CMake directly. `-S` is the source directory (the repository root), and `-B` is the build directory, which CMake creates if needed:

```bash
# from the repository root
cmake -S . -B build/my-openmp \
      -DCMAKE_BUILD_TYPE=Release \
      -DMATAR_BUILD_EXAMPLES=ON \
      -DMATAR_BUILD_TESTS=ON \
      -DKokkos_ENABLE_OPENMP=ON
cmake --build build/my-openmp -j
```

Test it with `ctest --test-dir build/my-openmp --output-on-failure` (see [Testing with CTest](#testing-with-ctest)). Because `-S` and `-B` are explicit, these commands also work from any other directory if you replace `.` and `build/my-openmp` with full paths. Without presets, the examples and tests are **off by default**, so turn them on as shown above.

Select the Kokkos backend with the standard Kokkos variables: `Kokkos_ENABLE_SERIAL`, `Kokkos_ENABLE_OPENMP`, `Kokkos_ENABLE_THREADS`, `Kokkos_ENABLE_CUDA`, `Kokkos_ENABLE_HIP`, `Kokkos_ENABLE_SYCL`, and `Kokkos_ARCH_*`. They are passed straight through to the bundled Kokkos. You can also use `MATAR_DEVICE_BACKEND`/`MATAR_HOST_BACKEND` (see [Host-side parallelism](#host-side-parallelism)).

### Building without Kokkos

With `MATAR_ENABLE_KOKKOS=OFF`, MATAR is a dependency-free serial header library: only the host types are available, and the parallel macros become plain loops. The build and test commands are in [Build commands for every backend](#build-commands-for-every-backend) and [Testing with CTest](#testing-with-ctest).

### Using an already-installed Kokkos

By default, MATAR always builds its bundled Kokkos, so the backend stays under MATAR's control. To use a Kokkos you installed yourself instead, opt in explicitly:

```bash
cmake --preset serial -DMATAR_USE_EXTERNAL_KOKKOS=ON -DKokkos_ROOT=/path/to/kokkos/install
```

In this case the backend comes from how that Kokkos was built, and `Kokkos_ENABLE_*` flags passed to MATAR have no effect.

### Installing MATAR

Installing copies the MATAR headers and CMake package files (and, with the bundled Kokkos, Kokkos itself) into a prefix, so other projects can use `find_package(Matar)`. Run this from the repository root after building:

```bash
cmake --install build/serial --prefix $HOME/opt/matar-serial
```

Then build your own project with `-DCMAKE_PREFIX_PATH=$HOME/opt/matar-serial`. An installed MATAR is tied to the Kokkos backend it was configured with, so install one prefix per backend.

### Building the getting-started tutorials

Each `tutorial/getting_started/Example*/` directory is a standalone CMake project that pulls MATAR in from the repository with `add_subdirectory`. Build them from **inside the example's directory** with its `build.sh` script:

```bash
cd tutorial/getting_started/Example0
./build.sh -t serial        # or openmp, pthreads, cuda, hip, all;  -d = debug build
```

The output goes to `build_<backend>/` inside the example's directory.

### CMake options reference

| Option | Default | Description |
|---|---|---|
| `MATAR_ENABLE_KOKKOS` | ON | Build the Kokkos-backed device/dual types (builds the bundled Kokkos submodule) |
| `MATAR_ENABLE_MPI` | OFF | Enable the MPI-aware types (`MPICArrayKokkos`, `CommunicationPlan`) |
| `MATAR_ENABLE_GPU_AWARE_MPI` | OFF | Assume the MPI implementation is GPU-aware |
| `MATAR_USE_EXTERNAL_KOKKOS` | OFF | Use an installed Kokkos (`-DKokkos_ROOT=<prefix>`) instead of the submodule |
| `MATAR_BUILD_EXAMPLES` | OFF (ON in presets) | Build the example programs |
| `MATAR_BUILD_TESTS` | OFF (ON in presets) | Build the unit tests (`ctest` to run) |
| `MATAR_BUILD_BENCHMARKS` | OFF | Build the Google Benchmark benchmarks in `benchmark/` |
| `MATAR_INSTALL` | ON at top level | Generate install rules; off when MATAR is embedded in another project |
| `MATAR_DEVICE_BACKEND` | unset | `serial`, `openmp`, `pthreads`, `cuda`, `hip`, `sycl`; backend for `FOR_ALL` and friends |
| `MATAR_HOST_BACKEND` | unset | `serial`, `openmp`, `pthreads`; backend for the `_HOST` macros |
| `MATAR_REAL` | double | Precision of the `real_t` tier: `double`, `float`, `half`, `bfloat16`, `quad` |
| `MATAR_HIGH_REAL` | double | Precision of the `high_real_t` tier: `double`, `float`, `quad` |
| `MATAR_LOW_REAL` | double | Precision of the `low_real_t` tier: `double`, `float`, `half`, `bfloat16` |
| `CMAKE_BUILD_TYPE` | Release | `Debug` turns on bounds and dimension checks |
| `CMAKE_VECTOR_NOVEC` | OFF | Use the same compiler flags but with auto-vectorization disabled (for comparisons) |
| `CMAKE_VECTOR_VERBOSE` | OFF | Print the compiler's vectorization reports |

Plus every standard Kokkos option (`Kokkos_ENABLE_*`, `Kokkos_ARCH_*`), which is forwarded to the bundled Kokkos.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not read presets from ...: File not found: .../CMakePresets.json` | You are not in the repository root. `cd` to the directory containing `CMakePresets.json`, or use the `-S`/`-B` and `--test-dir` forms. |
| `The bundled Kokkos submodule is missing` | Run `git submodule update --init --recursive` from the repository root. |
| `CMake 3.xx or higher is required` | Your CMake is too old. Install a newer one (for example `pip install cmake`) or `module load cmake`. |
| Configure fails while downloading `googletest` / `benchmark` | The machine has no internet access. Configure with `-DMATAR_BUILD_TESTS=OFF` (and leave benchmarks off), or point FetchContent at a local copy with `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=<path>`. |
| Wrong compiler or backend after loading modules | The old configuration is still cached. Run `cmake --fresh --preset <name>` or delete `build/<name>/`. |
| `MATAR_DEVICE_BACKEND=..., but Kokkos's default execution space will be ...` | A higher-priority backend is still enabled from an earlier configure. Use a fresh build directory. |
| CUDA: `CMAKE_CUDA_STANDARD 20` / C++20 errors | CUDA builds need CMake 3.25.2 or newer and a CUDA toolkit with C++20 support. |
| Runtime: `attempt to access inaccessible memory space` | A `*Device` array was used inside a `_HOST` macro or on the host. See [Host-side parallelism](#host-side-parallelism). |

## Host-side parallelism

Every MATAR macro (`FOR_ALL`, `DO_ALL`, `FOR_REDUCE_*`, ...) has a `_HOST` twin that runs in the **host** execution space instead of the device space. Because device kernels are asynchronous and host kernels only block the calling thread, CPU work can proceed while the GPU is still busy — useful for file I/O and for algorithms that are simply faster on a CPU:

```c++
FOR_ALL(i, 0, n, { device_field(i) = compute(i); });   // GPU, returns immediately

FOR_ALL_HOST(i, 0, n, {                                 // CPU, runs concurrently
    out_lines[i] = format_record(host_field(i));
});
MATAR_FENCE_HOST();                                     // wait for the host work
MATAR_FENCE_DEVICE();                                   // wait for the GPU work
```

The `_HOST` macros differ from their device counterparts in two ways: the loop body captures **by reference**, so `std::` containers, file streams, and other non-device-copyable objects can be used directly; and no `_CLASS` variants are needed (`_CLASS` spellings exist as aliases).

**MATAR is device-centric: the `*Device` types always live on the device.** To run host-parallel work over data that also lives on the device, declare it as a **Dual** type and go through the `.host()` accessor inside the host macro:

```c++
CArrayDual<real_t> field(n, "field");
FOR_ALL_HOST(i, 0, n, {                 // CPU-side pass over the host mirror
    field.host(i) = read_from_file(i);
});
field.update_device();                  // publish to the device
```

A host kernel may otherwise only touch host-accessible data — the `*Host` types or plain `std::` data. Passing a `*Device` array to a host macro compiles but aborts at run time with `attempt to access inaccessible memory space`.

Select the two backends independently:

| Option | Values | Description |
|---|---|---|
| `MATAR_DEVICE_BACKEND` | `serial`, `openmp`, `pthreads`, `cuda`, `hip`, `sycl` | Backend for `FOR_ALL` and friends |
| `MATAR_HOST_BACKEND` | `serial`, `openmp`, `pthreads` | Backend for the `_HOST` macros |

```
cmake --preset cuda-hostomp     # CUDA device + OpenMP host: genuinely concurrent
cmake -B build -DMATAR_DEVICE_BACKEND=cuda -DMATAR_HOST_BACKEND=openmp
```

`examples/host_device_overlap.cpp` is a runnable walk-through of all of this: the `_HOST` argument forms, the host-only objects a host kernel may capture, a device `FOR_ALL` overlapped with a host `FOR_ALL_HOST` (with the timings printed), and the Dual-type handoff in both directions.

Both options are optional; leaving them unset keeps the historical behavior of setting `Kokkos_ENABLE_*` directly. Note that Kokkos permits only one host-parallel backend per build, so `openmp` and `pthreads` cannot be paired with each other, and selecting the same backend on both sides is valid but gives no concurrency (the two macro families then share one execution space — MATAR warns at configure time).

## Precision tiers

MATAR provides three floating-point type names whose meaning is fixed at configure time — code just uses the names, and the CMake flags decide what they are per build/architecture:

* `real_t` — the default working precision for field data
* `high_real_t` — fields that must stay accurate (coordinates, conserved-quantity sums)
* `low_real_t` — tolerant or bulk-storage fields (history buffers, gradients, output)

```
cmake --preset serial-fp32                 # real_t = float
cmake -B build -DMATAR_REAL=half           # real_t = Kokkos half_t (native on GPU backends)
cmake -B build -DMATAR_REAL=float -DMATAR_HIGH_REAL=double   # mixed
```

Notes:
* `half`/`bfloat16` map to the native 16-bit types on CUDA/HIP/SYCL and are transparently float-backed on CPU backends (the macros `MATAR_FP16_IS_EMULATED`/`MATAR_BF16_IS_EMULATED` report which at compile time). `quad` is `__float128`, host backends only. Non-Kokkos builds support only `double` and `float`; anything else fails at configure.
* User code only ever writes the three tier names — declare fields as `CArrayDevice<real_t>` (or `high_real_t`/`low_real_t`) and write constants as `real_t(0.5)`; the build flags decide what those names mean. The solvers in `solvers/` compute in `real_t`, so they run at whatever working precision the build selects.
* The MPI-aware types work at every tier: `MPICArrayKokkos<real_t>` halo exchange and `all_reduce` need nothing extra from the user. Internally, native 16-bit types travel as bytes and reduce by promoting to double on the wire (exact), and quad uses a custom MPI datatype and reduction op.

The Kokkos backend is selected with the standard Kokkos CMake variables (`Kokkos_ENABLE_OPENMP`, `Kokkos_ENABLE_CUDA`, `Kokkos_ENABLE_HIP`, `Kokkos_ARCH_*`, ...), which are passed through to the bundled Kokkos build. With `MATAR_ENABLE_KOKKOS=OFF`, MATAR is a dependency-free serial header-only library.

## Using MATAR as a third-party library
MATAR is header-only and exports a single CMake target, **`matar::matar`**. Linking it propagates everything a consumer needs: the include paths, the C++20 requirement, the `HAVE_KOKKOS`/`HAVE_CUDA`/`HAVE_HIP`/`HAVE_OPENMP`/`HAVE_THREADS`/`HAVE_MPI` compile definitions, and the Kokkos/MPI link dependencies. There is nothing to link manually and no MATAR library to build.

Consumers need **CMake >= 3.22** and a **C++20** compiler; both come from Kokkos 5. Whichever method you choose, set the Kokkos backend variables *before* MATAR is added, because the bundled Kokkos is configured as part of that step.

### Option 1: FetchContent
CMake downloads MATAR at configure time. Nothing needs to be vendored into your repository.

```cmake
cmake_minimum_required(VERSION 3.22)
project(myapp LANGUAGES CXX)

# Pick the Kokkos backend BEFORE MATAR is made available.
set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
    matar
    GIT_REPOSITORY https://github.com/lanl/MATAR.git
    GIT_TAG        <tag-or-commit>   # pin a release; avoid tracking a branch
)
FetchContent_MakeAvailable(matar)    # also configures the bundled Kokkos

add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE matar::matar)
```

MATAR carries Kokkos as its own submodule, and FetchContent updates submodules recursively by default, so the bundled Kokkos is fetched and built automatically. No extra steps are needed.

### Option 2: git submodule
Vendors MATAR into your repository, which pins the exact commit and allows offline builds.

```bash
git submodule add https://github.com/lanl/MATAR.git external/MATAR
git submodule update --init --recursive
```

`--recursive` is required: MATAR contains Kokkos as a nested submodule, and without it `external/MATAR/src/Kokkos/kokkos` is left empty and the MATAR build stops with an error telling you to re-run the command. Anyone cloning your project afterwards needs `git clone --recursive`, or the same `git submodule update --init --recursive` after cloning.

```cmake
cmake_minimum_required(VERSION 3.22)
project(myapp LANGUAGES CXX)

set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
add_subdirectory(external/MATAR)

add_executable(myapp main.cpp)
target_link_libraries(myapp PRIVATE matar::matar)
```

### Option 3: an installed MATAR
Against a MATAR installed with `cmake --install build/<preset> --prefix <prefix>`:

```cmake
find_package(Matar REQUIRED)         # configure with -DCMAKE_PREFIX_PATH=<prefix>
target_link_libraries(myapp PRIVATE matar::matar)
```

When the bundled Kokkos was used, it is installed into the same prefix and `MatarConfig.cmake` resolves it through `find_dependency(Kokkos)`.

### Selecting a backend, and reusing your own Kokkos
The backend is chosen with the standard Kokkos cache variables (`Kokkos_ENABLE_SERIAL`, `Kokkos_ENABLE_OPENMP`, `Kokkos_ENABLE_CUDA`, `Kokkos_ENABLE_HIP`, `Kokkos_ARCH_*`, ...), which are passed straight through to the bundled Kokkos build. Set them from your own `CMakeLists.txt` as shown above, or on the command line with `-DKokkos_ENABLE_CUDA=ON`.

If your project already provides Kokkos — through `find_package(Kokkos)` or an `add_subdirectory` of its own Kokkos copy — do that **before** adding MATAR. MATAR detects the existing `Kokkos::kokkos` target and links against it instead of building its bundled submodule, so only one Kokkos ends up in the build:

```cmake
find_package(Kokkos REQUIRED)        # your Kokkos wins
FetchContent_MakeAvailable(matar)    # MATAR reuses it
```

Other options worth knowing when embedding MATAR:

| Option | Effect |
|---|---|
| `MATAR_ENABLE_KOKKOS=OFF` | Host-only types; MATAR becomes a dependency-free header library and the parallel macros compile to plain serial loops |
| `MATAR_ENABLE_MPI=ON` | Adds `MPICArrayKokkos` and `CommunicationPlan` (requires MPI) |
| `MATAR_INSTALL` | Defaults to off when MATAR is embedded in another project, so it contributes no install rules to your package |

Note that while an embedded MATAR installs nothing, the *bundled Kokkos* still adds its own install rules, so `cmake --install` on your project will also deposit the Kokkos headers and CMake package files into your prefix. Use an external or parent-provided Kokkos if you need to keep it out of your install tree.

## Running codes in parallel
The OpenMP and pthreads backends run on as many threads as you give them. Set the count either with an environment variable or with a Kokkos command-line flag:
```
export OMP_NUM_THREADS=4                 # OpenMP backend
./mycode                                 

./mycode --kokkos-num-threads=4          # any host backend (OpenMP or pthreads)
KOKKOS_NUM_THREADS=4 ./mycode            # same, as an environment variable
```
(Kokkos 5 removed the old `--kokkos-threads` spelling; use `--kokkos-num-threads`.) The Kokkos flags only reach Kokkos if you pass the command-line arguments to `MATAR_INITIALIZE`, which wraps `Kokkos::initialize`:
```
int main(int argc, char* argv[])
{
    MATAR_INITIALIZE(argc, argv);

    // coding goes here

    MATAR_FINALIZE();

    return 0;
}
```
This still works when no arguments are given, so the same `main` works for every backend. The CUDA and HIP backends do not need a thread count.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
This program is open source under the BSD-3 License.

## Citation
```
@article{MATAR,
title = "{MATAR: A Performance Portability and Productivity Implementation of Data-Oriented Design with Kokkos}",
journal = {Journal of Parallel and Distributed Computing},
pages = {86-104},
volume = {157},
year = {2021},
author = {Daniel J. Dunning and Nathaniel R. Morgan and Jacob L. Moore and Eappen Nelluvelil and Tanya V. Tafolla and Robert W. Robey},
keywords = {Performance, Portability, Productivity, Memory Efficiency, GPUs, dense and sparse storage}
```

