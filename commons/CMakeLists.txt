cmake_minimum_required(VERSION 3.21)
project(commons) 


# Careful, maybe remove this later and add proper exporting
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)


# -- C++ commons
# Define the commons library
add_library(opendarts_commons SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/src/commons.cpp)


set_target_properties(opendarts_commons PROPERTIES
    POSITION_INDEPENDENT_CODE ON  # Recommended for shared libraries
)

# Define the include directories to propagate when linking
target_include_directories(opendarts_commons
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(opendarts_commons PRIVATE
  opendarts_compiler_flags
)

install(TARGETS opendarts_commons 
  CONFIGURATIONS RELEASE DEBUG
  DESTINATION ./
)

# -- End C++ commons


# -- Python binding via pybind11
pybind11_add_module(commons 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/py_commons.cpp
)

# Link pybind11 to the commons module
target_link_libraries(commons PRIVATE
  opendarts_compiler_flags
  opendarts_commons 
  pybind11::module)


install(TARGETS commons 
  CONFIGURATIONS RELEASE RELWITHDEBINFO DEBUG ""
  DESTINATION ./
)
