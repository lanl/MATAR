Please read the homepage documentation first. The documentation listed here repeats what is listed on the homepage and adds a few additional details to help users with building and using MATAR.

We encourage developers to use Anaconda, making the build process as simple as possible. Anaconda can be installed on Mac and Linux OSs. At this time, Windows users must install Anaconda inside WSL-2. To use the anaconda package, follow the steps for your platform to install [anaconda](https://docs.anaconda.com/free/anaconda/install/index.html)/[miniconda](https://docs.conda.io/en/latest/miniconda.html)/[mamba](https://mamba.readthedocs.io/en/latest/installation.html). 


## Building MATAR inside Anaconda
It is advised to use the Anaconda package manager to build the MATAR examples and tests. Within a terminal, go to where you want to build MATAR, and follow the steps listed below here.

1. Create an environment and activate:
```
conda create --name MATAR
conda activate MATAR
```

2. Install needed packages:
   
If building MATAR examples and tests for a CPU backend in Kokkos, then the following packages are needed
```
conda install -c conda-forge cxx-compiler 
conda install -c conda-forge fortran-compiler
conda install cmake
```

To build examples for an Nvidia GPU, then install the following packages (and do not install anything shown in the CPU instructions above here),
```
conda install -c conda-forge "cxx-compiler=1.5.2" 
conda install -c conda-forge "fortran-compiler=1.5.2"
conda install cmake 
```
Here, the conda compiler version 1.5.2 is used, which will install gcc version 11.0. Omitting 1.5.2 will install gcc version 12 (at this time). We encourage users who are interested in running with the CUDA backend to use a gcc version with 1 number less than the CUDA library version. Now the CUDA libraries must be installed.
```
 conda install -c conda-forge cudatoolkit      
 conda install -c conda-forge cudatoolkit-dev
```

At this point, all necessary dependencies and third-party libraries are installed in the Conda environment (for the Kokkos CPU or NVidia GPU backends). 

3. Go to the scripts folder in the MATAR directory.  Then run the build script as:
```
source build-matar.sh --help
```

Which outputs:

```
Usage: source build-matar.sh [OPTION]
Required arguments:
  none

Optional arguments:
  --execution=<examples|test>. Default is 'all'
  --kokkos_build_type=<serial|openmp|pthreads|cuda|hip>.  Default is 'serial'
  --build_action=<full-app|set-env|install-kokkos|install-matar|matar>. Default is 'full-app'
  --machine=<darwin|chicoma|linux|mac> (default: none)
  --num_jobs=<number>: Number of jobs for 'make' (default: 1, on Mac use 1)
  --build_cores=<Integers greater than 0>. Default is set 1
  --help: Display this help message
```

To build MATAR with a kokkos backend you would need to provide `--kokkos_build_type` option. The command below builds the examples in MATAR using the serial version (which is the default) of Kokkos:

```
source build-matar.sh --kokkos_build_type=serial
```

This will build examples and tests in the folder `build-matar_serial`. The binaries for the examples are in bin folder inside that folder.

To build MATAR with the openMP parallel backend, use `--kokkos_build_type=openmp` option. 

```
source build-matar.sh --kokkos_build_type=openmp
```
The same logic applies to building the examples with the other Kokkos backends.

## Building MATAR without Anaconda
The user must install compilers and thirdparty libraries on a linux machine using `sudo app-get install`, or on a Mac using homebrew, or on an HPC maching by loading modules.  As part of the build script, we call scripts (inside the machines folder) to configure the enviroment variables. A note of caution when building on a Mac, the cmake outside anaconda is rescricted to serial compulation, use the default value of 1 and only use -j1 when recompiling the code.  For parallel compulation on a Mac, use the anaconda cmake.

If you need to simply rebuild the app and not get a new kokkos installation, simply type
```
source cmake_build_<example/test>.sh <args>
```
with the same arguments you would use with the build-matar.sh

If you log onto a machine for the first time (or get a new allocation) you will need to run
```
source setup-env.sh <args>
```
with the same arguments you would with build-mater.sh


## Building a simple project with MATAR
To use MATAR in your simple project with the serial Kokkos backend, you can copy paste all the `.h` files in `/src/include` into your local directory where your code is compiled.  Building a Kokkos backend requires using cmake and one the above approaches.
