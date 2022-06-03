
set(matar_SOURCES
  matar.cpp
  )

target_sources(matar PRIVATE ${matar_SOURCES})

set(matar_HEADERS
  matar.h
  macros.h
  )

list(TRANSFORM matar_HEADERS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

set_target_properties(matar PROPERTIES
  PUBLIC_HEADER "${matar_HEADERS}")

target_include_directories(matar PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)
