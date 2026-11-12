### Make sure arguments are as we'd expect

### Load environment modules here
### Assign names as relevant

if [ "$1" = "hpc" ]
then
    mygcc="gcc/9.4.0"
    myclang="clang/13.0.0"
    mycuda="cuda/11.4.0"
    myrocm="rocm"

    module purge
    if [ "$2" = "cuda" ]
    then
        module purge
        module load ${mygcc}
        module load ${mycuda}
    elif [ "$2" = "hip" ]
    then
        module purge
        module load ${myclang}
        module load ${myrocm}
    else
        module load ${mygcc}
    fi
    module load cmake
    module -t list
fi


my_parallel=""
if [ "$2" != "none" ]
then
    my_parallel="kokkos-$2"
fi

my_build="build-examples"
if [ -z $3 ]
then
    my_build="${my_build}-${my_parallel}"
else
    my_build=$3
fi


export scriptdir=`pwd`

cd ..
export topdir=`pwd`
export basedir=${topdir}
export srcdir=${basedir}/src
export libdir=${basedir}/lib
export builddir=${basedir}/${my_build}
export installdir=${basedir}/install-kokkos

export KOKKOS_SOURCE_DIR=${srcdir}/Kokkos/kokkos
export KOKKOS_BUILD_DIR=${builddir}/kokkos
export KOKKOS_INSTALL_DIR=${installdir}/kokkos-${my_parallel}

export MATAR_SOURCE_DIR=${srcdir}
export MATAR_BUILD_DIR=${builddir}
export MATAR_INSTALL_DIR=${installdir}/matar

cd $scriptdir



