#add_library(matar INTERFACE)

#target_sources(matar INTERFACE matar.h)

#target_include_directories(matar INTERFACE ./MATAR)
add_library(matar matar.cpp matar.h)

if (KOKKOS)
  add_definitions(-DHAVE_KOKKOS=1)
  #set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_KOKKOS)
  if (CUDA)
    add_definitions(-DHAVE_CUDA=1)
    #set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_CUDA)
  elseif (OPENMP)
    add_definitions(-DHAVE_OPENMP=1)
    #set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_OPENMP)
  endif()

  #target_link_libraries(matar INTERFACE Kokkos::kokkos)
  target_link_libraries(matar  Kokkos::kokkos)
endif()

