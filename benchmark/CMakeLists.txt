cmake_minimum_required(VERSION 3.1.3)

project (matarbenchmark)

set(benchmark_DIR "benchmark/build")
find_package(benchmark REQUIRED)
find_package(Matar REQUIRED)

if (KOKKOS)
  find_package(Kokkos REQUIRED) #new

  if (CUDA)
    add_definitions(-DHAVE_CUDA=1)
  elseif (HIP)
    add_definitions(-DHAVE_HIP=1)
  elseif (OPENMP)
    add_definitions(-DHAVE_OPENMP=1)
  elseif (THREADS)
    add_definitions(-DHAVE_THREADS=1)
  endif()
endif()

find_package(Kokkos REQUIRED) #new

set(This matar_benchmark)

set(Sources
    src/serial_types_benchmark.cpp
)

add_executable(${This} ${Sources})

target_link_libraries(${This}
    matar
    benchmark::benchmark
)

if (KOKKOKS)
  target_link_libraries(${This} matar Kokkos::kokkos benchmark::benchmark)
endif ()
