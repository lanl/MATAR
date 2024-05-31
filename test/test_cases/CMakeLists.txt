cmake_minimum_required(VERSION 3.1.3)

set(This mater_test)

set(Sources
    standard_types_tests.cpp
)

add_executable(${This} ${Sources})
target_link_libraries(${This}
    matar
    gtest_main
)




add_executable(test_CArrayKokkos test_CArrayKokkos.cpp)
target_link_libraries(test_CArrayKokkos matar gtest_main)


if (KOKKOKS)
    target_link_libraries(${This} Kokkos::kokkos)
    target_link_libraries(test_CArrayKokkos Kokkos::kokkos)
endif()
