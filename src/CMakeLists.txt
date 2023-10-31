cmake_minimum_required(VERSION 3.12)
set (CMAKE_CXX_STANDARD 17)

#add_library(matar INTERFACE matar.h aliases.h host_types.h kokkos_types.h macros.h)
#add_library(matar matar.cpp)
#set_target_properties(matar PROPERTIES PUBLIC_HEADER "matar.h;macros.h;host_types.h;kokkos_types.h;aliases.h")

if (KOKKOS)
  find_package(Kokkos REQUIRED) #new
endif ()

add_library(matar INTERFACE)
target_include_directories(matar INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>  
    $<INSTALL_INTERFACE:include>
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/MatarConfigVersion.cmake"
    VERSION 1.0
    COMPATIBILITY AnyNewerVersion
)

install(TARGETS matar
    EXPORT matarTargets
    LIBRARY DESTINATION lib COMPONENT Runtime
    ARCHIVE DESTINATION lib COMPONENT Development
    RUNTIME DESTINATION bin COMPONENT Runtime
    PUBLIC_HEADER DESTINATION include COMPONENT Development
    BUNDLE DESTINATION bin COMPONENT Runtime
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/MatarConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/MatarConfig.cmake"
    INSTALL_DESTINATION lib/cmake/matar
)

install(EXPORT matarTargets DESTINATION lib/cmake/matar)
install(FILES "${PROJECT_BINARY_DIR}/MatarConfigVersion.cmake"
              "${PROJECT_BINARY_DIR}/MatarConfig.cmake"
        DESTINATION lib/cmake/matar)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)
