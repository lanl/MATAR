include(FetchContent)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

add_executable(BM_CArray src/CArray_benchmark.cpp)
target_link_libraries(BM_CArray matar::matar benchmark::benchmark)

if(MATAR_ENABLE_KOKKOS)
    add_executable(BM_CArrayDevice src/CArrayDevice_benchmark.cpp)
    target_link_libraries(BM_CArrayDevice matar::matar benchmark::benchmark)
endif()
