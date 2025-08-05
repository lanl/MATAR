#!/bin/bash -e
show_help() {
    echo "Usage: source $(basename "$BASH_SOURCE") [OPTION]"
    echo "Valid options:"
    echo "  --execution=<examples|test>. Default is 'all'"
    echo "  --kokkos_build_type=<none|serial|openmp|pthreads|cuda|hip|serial_mpi|openmp_mpi|cuda_mpi|hip_mpi|>. Default is 'serial'"
    echo "  --build_action=<full-app|set-env|install-kokkos|install-matar|matar>. Default is 'full-app'"
    echo "  --machine=<darwin|chicoma|linux|mac>. Default is 'linux'"
    echo "  --intel_mkl=<enabled|disabled>. Default is 'disabled'"
    echo "  --build_cores=<Integers greater than 0>. Default is set 1"
    echo "  --trilinos=<enabled|disabled>. Default is 'disabled'"
    echo "  --debug=<enabled|disabled>. Default is 'disabled'"
    echo "  --help: Display this help message"
    echo " "
    echo " "
    echo " "
    echo "      --build_action                  The desired build step to be execute. The default action is 'full-app'"
    echo " "
    echo "          full-app                    builds MATAR from scratch, installing dependencies where necessary."
    echo "          set-env                     set appropriate environment variables and loads software modules (if necessary)"
    echo "          install-kokkos              builds and installs Kokkos if not already installed. Clones from github if necessary"
    echo "          install-matar               builds and installs Matar if not already installed."
    echo "          matar                       Generates CMake files and builds matar only (none of the dependencies)."
    echo " "
    echo "      --execution                     Builds the desired executables you want to run. The default action is 'all'"
    echo " "
    echo "          examples                    builds examples"
    echo "          test                        builds tests"
    echo "          benchmark                   builds benchmarks for MATAR"
    echo " "
    echo "      --kokkos_build_type             The desired kokkos parallel backend to use. The default is 'serial'"
    echo " "
    echo "          none                        No Kokkos backend"
    echo "          serial                      Serial Kokkos backend"
    echo "          openmp                      OpenMP Kokkos backend"
    echo "          pthreads                    pthreads Kokkos backend"
    echo "          cuda                        Cuda Kokkos backend"
    echo "          hip                         HIP Kokkos backend"
    echo "          serial_mpi                  Serial Kokkos backendi plus MPI"
    echo "          openmp_mpi                  OpenMP Kokkos backendi plus MPI"
    echo "          cuda_mpi                    Cuda Kokkos backendi plus MPI"
    echo "          hip_mpi                     HIP Kokkos backendi plus MPI"
    echo " "
    echo "      --machine                       The machine you are building for. The default is 'linux'"
    echo " "
    echo "          darwin                      The darwin cluster at LANL. Uses module loads for software"
    echo "          linux                       A general linux machine (that does not use modules)"
    echo "          mac                         A Mac computer. This option does not allow for cuda and hip builds, and build_cores will be set to 1"
    echo " "
    echo "      --intel_mkl                     Decides whether to build Trilinos using the Intel MKL library"
    echo " "
    echo "          enabled                     Links and builds Trilinos with the Intel MKL library"
    echo "          disabled                    Links and builds Trilinos using LAPACK and BLAS"
    echo " "
    echo "      --build_cores                   The number of build cores to be used by make and make install commands. The default is 1"
    echo " "
    echo "      --trilinos                      Decides if Trilinos is available for certain MATAR functionality"
    echo " "
    echo "          disabled                    Trilinos is not being used"
    echo "          enabled                     Trilinos will be linked with MATAR to enable relevant functionality"
    echo " "
    echo "      --debug                         Decides if debug build is enabled"
    echo " "
    echo "          disabled                    Release build (default)"
    echo "          enabled                     Debug build with debug symbols and no optimization"
    echo " "
    return 1
}

# Initialize variables with default values
build_action="full-app"
execution="examples"
machine="linux"
kokkos_build_type="cuda"
build_cores="1"
trilinos="disabled"
intel_mkl="disabled"
debug="enabled"

# Define arrays of valid options
valid_build_action=("full-app" "set-env" "install-matar" "install-kokkos" "matar")
valid_execution=("examples" "test" "benchmark")
valid_kokkos_build_types=("none" "serial" "openmp" "pthreads" "cuda" "hip" "serial_mpi" "openmp_mpi" "cuda_mpi" "hip_mpi")
valid_machines=("darwin" "chicoma" "linux" "mac")
valid_trilinos=("disabled" "enabled")
valid_intel_mkl=("disabled" "enabled")
valid_debug=("disabled" "enabled")

# Parse command line arguments
for arg in "$@"; do
    case "$arg" in
        --build_action=*)
            option="${arg#*=}"
            if [[ " ${valid_build_action[*]} " == *" $option "* ]]; then
                build_action="$option"
            else
                echo "Error: Invalid --build_action specified."
                show_help
                return 1
            fi
            ;;
        --execution=*)
            option="${arg#*=}"
            if [[ " ${valid_execution[*]} " == *" $option "* ]]; then
                execution="$option"
            else
                echo "Error: Invalid --execution specified."
                show_help
                return 1
            fi
            ;;
        --machine=*)
            option="${arg#*=}"
            if [[ " ${valid_machines[*]} " == *" $option "* ]]; then
                machine="$option"
            else
                echo "Error: Invalid --machine specified."
                show_help
                return 1
            fi
            ;;
        --kokkos_build_type=*)
            option="${arg#*=}"
            if [[ " ${valid_kokkos_build_types[*]} " == *" $option "* ]]; then
                kokkos_build_type="$option"
            else
                echo "Error: Invalid --kokkos_build_type specified."
                show_help
                return 1
            fi
            ;;
        --build_cores=*)
            option="${arg#*=}"
            if [ $option -ge 1 ]; then
                build_cores="$option"
            else
                echo "Error: Invalid --build_cores specified."
                show_help
                return 1
            fi
            ;;
        --trilinos=*)
            option="${arg#*=}"
            if [[ " ${valid_trilinos[*]} " == *" $option "* ]]; then
                trilinos="$option"
            else
                echo "Error: Invalid --kokkos_build_type specified."
                show_help
                return 1
            fi
            ;;
        --intel_mkl=*)
            option="${arg#*=}"
            if [[ " ${valid_intel_mkl[*]} " == *" $option "* ]]; then
                intel_mkl="$option"
            else
                echo "Error: Invalid --intel_mkl specified."
                show_help
                return 1
            fi
            ;;
        --debug=*)
            option="${arg#*=}"
            if [[ " ${valid_debug[*]} " == *" $option "* ]]; then
                debug="$option"
            else
                echo "Error: Invalid --debug specified."
                show_help
                return 1
            fi
            ;;
        --help)
            show_help
            return 1
            ;;
        *)
            echo "Error: Invalid argument or value specified."
            show_help
            return 1
            ;;
    esac
done

# Check for correct combos with mac
if [ "$machine" = "mac" ] && [ "$kokkos_build_type" = "cuda" ]; then
    echo "Error: Mac cannot build with Kokkos Cuda backend"
    show_help
    return 1
fi

if [ "$machine" = "mac" ] && [ "$kokkos_build_type" = "hip" ]; then
    echo "Error: Mac cannot build with Kokkos HIP backend"
    show_help
    return 1
fi

if [ "$machine" = "mac" ] && [ $build_cores -ne 1 ]; then
    echo "Error: Mac cannot be built in parallel. Setting build cores to default 1"
    # Nothing to do, default is already 1
fi

if [ "$trilinos" = "enabled" ] && [ "$kokkos_build_type" = "none" ]; then
    echo "Error: Kokkos none cannot be requested with Trilinos"
    show_help
    return 1
fi


echo "Building based on these argument options:"
echo "Build action - ${build_action}"
echo "Execution - ${execution}"
echo "Kokkos backend - ${kokkos_build_type}"
echo "make -j ${build_cores}"
echo "Trilinos - ${trilinos}"
echo "Intel MKL library - ${intel_mkl}"

cd "$( dirname "${BASH_SOURCE[0]}" )"

# Always setup the environment
source setup-env.sh ${machine} ${kokkos_build_type} ${build_cores}

# Next, do action based on args
if [ "$build_action" = "full-app" ]; then
    rm -rf ${builddir}
    if [ ! -d "${builddir}" ]
    then
        mkdir -p ${builddir}
    fi

    if [ "$trilinos" = "disabled" ]; then    
        source kokkos-install.sh ${kokkos_build_type}
    elif [ "$trilinos" = "enabled" ]; then    
        source trilinos-install.sh ${kokkos_build_type}  ${intel_mkl}
    fi
    source matar-install.sh ${kokkos_build_type} ${trilinos} ${debug}
    source cmake_build_${execution}.sh ${kokkos_build_type} ${trilinos} ${debug}
elif [ "$build_action" = "install-kokkos" ]; then
    source kokkos-install.sh ${kokkos_build_type}
elif [ "$build_action" = "install-matar" ]; then
    source matar-install.sh ${kokkos_build_type} ${debug}
elif [ "$build_action" = "matar" ]; then
    # Clean build directory (assumes there is a stale build)
    make -C "${EXAMPLE_BUILD_DIR}" distclean
    source cmake_build_${execution}.sh ${kokkos_build_type} ${trilinos} ${debug}
else
    echo "No build action, only setup the environment."
fi

cd ${basedir}
