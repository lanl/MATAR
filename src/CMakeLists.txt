cmake_minimum_required(VERSION 3.12)
set (CMAKE_CXX_STANDARD 17)

if (KOKKOS)
  find_package(Kokkos REQUIRED) #new
endif ()

add_library(matar matar.cpp matar.h aliases.h host_types.h kokkos_types.h macros.h)

if (KOKKOS)
  set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_KOKKOS)
  if (CUDA)
    set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_CUDA)
  elseif (HIP)
    set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_HIP)
  elseif (OPENMP)
    set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_OPENMP)
  elseif (THREADS)
    set_target_properties(matar PROPERTIES COMPILE_DEFINITIONS HAVE_THREADS)
  endif()

  target_link_libraries(matar  Kokkos::kokkos)
endif()

target_include_directories(matar PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  
    $<INSTALL_INTERFACE:include>
    #${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(matar PROPERTIES PUBLIC_HEADER matar.h)

# Create install instructions
install(TARGETS
        matar
    EXPORT MatarConfig
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDE DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
export(TARGETS
        matar
    NAMESPACE matar::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/MatarConfig.cmake"
)
install(EXPORT
        MatarConfig
    DESTINATION lib/cmake/matar
    NAMESPACE matar::
)
