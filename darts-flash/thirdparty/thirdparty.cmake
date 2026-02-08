# Thirdparty libraries ---------------------------------------------------------
# Imports all thirdparty libraries so that they can be used in the 
# project.
# ------------------------------------------------------------------------------


# Initialize reporting ---------------------------------------------------------
# Reports we started looking for thirdparty libraries and initializes the flag 
# that checks if all have been found 
message(CHECK_START "Importing thirdparty libraries")
unset(thirdparty_missing_components)
# ------------------------------------------------------------------------------

# Adds eigen3 ------------------------------------------------------------------
message(CHECK_START "   Importing Eigen3")
if(NOT DEFINED Eigen3_DIR)
  set(Eigen3_DIR "${CMAKE_SOURCE_DIR}/thirdparty/install/share/eigen3/cmake/")
endif(NOT DEFINED Eigen3_DIR)

# find_package requires absolute paths to work, make sure the path is absolute
message(STATUS "      Converting Eigen3 search path to absolute path")
message(STATUS "         Input path: ${Eigen3_DIR}")
file(REAL_PATH "${Eigen3_DIR}" Eigen3_DIR BASE_DIRECTORY "${CMAKE_BINARY_DIR}")
message(STATUS "         Absolute path: ${Eigen3_DIR}")

# Find Eigen3
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
if (TARGET Eigen3::Eigen)
  message(STATUS "      Found Eigen3: TRUE")
else()
  message(FATAL_ERROR "      Found Eigen3: FALSE")
endif (TARGET Eigen3::Eigen)
# find_package(PkgConfig)
# pkg_search_module(Eigen3 REQUIRED eigen3)

# Some user feedback info
message(STATUS "      Include directories: ${EIGEN3_INCLUDE_DIR}")

message(CHECK_PASS "done!")
# ------------------------------------------------------------------------------

# Adds pybind11 ----------------------------------------------------------------
message(CHECK_START "   Importing pybind11")
# set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
# find_package(pybind11 CONFIG REQUIRED)

# add_library(pybind11 INTERFACE)
# target_include_directories(pybind11 INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/pybind11")
# Disable pybind11 tests and exclude them from the default build
set(PYBIND11_TEST OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/pybind11 EXCLUDE_FROM_ALL)

# # find_package requires absolute paths to work, make sure the path is absolute
# message(STATUS "      Converting pybind11 search path to absolute path")
# message(STATUS "         Input path: ${pybind11_DIR}")
# file(REAL_PATH "${pybind11_DIR}" pybind11_DIR BASE_DIRECTORY "${CMAKE_BINARY_DIR}")
# message(STATUS "         Absolute path: ${pybind11_DIR}")
# find_package(pybind11 CONFIG REQUIRED)

# Some user feedback info
message(STATUS "      Include directories: ${pybind11_INCLUDE_DIRS}")

message(CHECK_PASS "done!")
# ------------------------------------------------------------------------------


# Finalize reporting and check if all libraries have been added ----------------
message(CHECK_PASS "done!")
# ------------------------------------------------------------------------------
