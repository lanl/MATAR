include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
# On Windows: keep gtest from overriding the parent project's runtime settings.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_subdirectory(test_cases)

add_executable(test_csr CSRArray.cpp)
target_link_libraries(test_csr matar::matar gtest_main)

add_executable(test_csc CSCArray.cpp)
target_link_libraries(test_csc matar::matar gtest_main)

include(GoogleTest)
gtest_discover_tests(test_csr DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_csc DISCOVERY_MODE PRE_TEST)
