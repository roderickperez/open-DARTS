cmake_minimum_required(VERSION 3.21)

project(DARTS-flash)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Load cmake modules -----------------------------------------------------------
# set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/settings/cmake/modules/")

include(FeatureSummary)
# include(GetGitRevisionDescription)  # to extract git information (available in source)
# ------------------------------------------------------------------------------

# Set C++ version 
set(CMAKE_CXX_STANDARD 17)

# # Set default compile flags for GCC
# if(CMAKE_COMPILER_IS_GNUCXX)
#     message(STATUS "GCC detected, adding compile flags")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -fPIC")
# endif(CMAKE_COMPILER_IS_GNUCXX)

# Set compiler specific variables
# Used for compiler flags and other cross compiler variables
message(STATUS "  Setting compiler and linker flags for: ${CMAKE_CXX_COMPILER_ID}")
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # Compiler flags 
  add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)
  set(dartsflash_cxx_compiler_flags, "-W3;-WX")

  # Linker flags (currently not needed)
  # set(opendarts_cxx_linker_flags, "-W3")
  
  # Compiler specific helper variables 
   
  # Flags cmake to import external libraries as msvc like files: .lib and .dll
  # instead of .a and .so
  set(import_externals_as_msvc TRUE)
  
  add_compile_definitions(_USE_MATH_DEFINES)

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(STATUS "  Setting GNU compiler and linker flags")
  # Compiler flags 
  set(dartsflash_cxx_compiler_flags "-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused;-Werror;-fPIC;-fvisibility=hidden")
  
  if(SET_CXX11_ABI_0)
    list(APPEND dartsflash_cxx_compiler_flags "-D_GLIBCXX_USE_CXX11_ABI=0")
  endif()

  # Linker flags (currently not needed)
  # set(opendarts_cxx_linker_flags, "-W3")
  
  # Compiler specific helper variables (currently not needed)

elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(STATUS "  Setting Clang compiler and linker flags")
  # Compiler flags 
  set(dartsflash_cxx_compiler_flags "-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused;-Werror;-fPIC;-fvisibility=hidden")

  if(SET_CXX11_ABI_0)
    list(APPEND dartsflash_cxx_compiler_flags "-D_GLIBCXX_USE_CXX11_ABI=0")
  endif()
  
  # Compiler specific helper variables (currently not needed)
  
endif()

message(STATUS "   compiler flags: ${dartsflash_cxx_compiler_flags}")

# Setup options
option(ENABLE_TESTING "Build and copy testing stuff" OFF)  # by default tests are not built, to save time
add_feature_info(ENABLE_TESTING ENABLE_TESTING "Build and copy testing stuff")  # add option as feature for pretty display

# Report options for this build
feature_summary(WHAT ENABLED_FEATURES DISABLED_FEATURES PACKAGES_FOUND)
feature_summary(FILENAME ${CMAKE_CURRENT_BINARY_DIR}/features.log WHAT ALL)
message(STATUS "The list of features was also saved to file: ${CMAKE_CURRENT_BINARY_DIR}/features.log\n")

# Setup testing ----------------------------------------------------------------

# Report Enable or disable testing via option flag
if(ENABLE_TESTING)
  enable_testing()
endif()
# ------------------------------------------------------------------------------


# Build dummy library to contain compiler flags --------------------------------
add_library(dartsflash_compiler_flags INTERFACE)

# set(dartsflash_cxx_compiler_flags, "-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused;-fPIC")
# set(dartsflash_cxx_linker_flags, "")

target_compile_features(dartsflash_compiler_flags
  INTERFACE
  cxx_std_11
)

target_compile_options(dartsflash_compiler_flags 
  INTERFACE
    ${dartsflash_cxx_compiler_flags}
)

# target_link_options(dartsflash_compiler_flags 
#   INTERFACE
#     ${dartsflash_cxx_linker_flags}
# )

# ------------------------------------------------------------------------------

# Thirdparty dependencies ------------------------------------------------------
include(thirdparty/thirdparty.cmake)
# ------------------------------------------------------------------------------

add_subdirectory(cpp)
add_subdirectory(tests)
add_subdirectory(examples)
