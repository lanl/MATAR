cmake_minimum_required(VERSION 3.1.3)

project (laplace)

if (NOT KOKKOS)
  add_executable(test_carray_right main_carray_right.cpp)
  add_executable(test_carray_wrong main_carray_wrong.cpp)
  add_executable(test_farray_right main_farray_right.cpp)
  add_executable(test_farray_wrong main_farray_wrong.cpp)

  target_link_libraries(test_carray_right matar)
  target_link_libraries(test_carray_wrong matar)
  target_link_libraries(test_farray_right matar)
  target_link_libraries(test_farray_wrong matar)
endif()

if (KOKKOS)
  #add_executable()

  add_definitions(-DHAVE_KOKKOS=1)
  #set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_KOKKOS)
  if (CUDA)
    #add_definitions(-DHAVE_CUDA=1)
    set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_CUDA)
  elseif (HIP)
    #add_definitions(-DHAVE_HIP=1)
    set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_HIP)
  elseif (OPENMP)
    #add_definitions(-DHAVE_OPENMP=1)
    set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_OPENMP)
  endif()

  #target_link_libraries()
endif()
