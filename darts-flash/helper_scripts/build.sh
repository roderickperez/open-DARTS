# Setup shell script run -------------------------------------------------------
# Exit when any command fails
set -e  
# ------------------------------------------------------------------------------

################################################################################
# Help info                                                                    #
################################################################################
Help_Info()
{
  echo "$(basename "$0") [-h] [-c] [-t] [-p] [-s] [-w] [-r] [-e EIGEN3_DIRECTORY] [-d INSTALL CONFIGURATION] [-j NUM THREADS] [-g g++-13]"
  echo "   Script to install opendarts-linear_solvers on macOS."
  echo "USAGE: "
  echo "   -h : displays this help menu."
  echo "   -c : cleans up build to prepare a new fresh build."
  echo "   -t : Disable testing: ctest of dartsflash."
  echo "   -p : Skip building python interface."
  echo "   -s : Disable test-suite: pytest of dartsflash."
  echo "   -w : Enable generation of python wheel."
  echo "   -r : Skip building requirements (if you have them already installed)."
  echo "   -e : Path to Eigen3, example: ../thirdparty/install/share/eigen3/cmake/"
  echo "   -d : Mode in which project is installed [Release, Debug]."
  echo "   -j : Number of threads for compilation."
  echo "   -g : Specify a compiler (g++) version, example: g++-13"
}
################################################################################
# Main program                                                                 #
################################################################################

# Read input arguments ---------------------------------------------------------
clean_mode=false  # Set mode to clean up, cleans build to prepare for fresh new build
testing=true     # Whether to enable the testing (ctest) of solvers.
build_python=true # Whether to build the python interface.
test_python=true  # Whether to enable the test-suite (pytest).
wheel=false       # Whether to generate python wheel.
skip_req=false    # Skip building requirements.
config="Release"  # Default configuration (install).
NT=8              # Number of threads by default 8
gpp_version=g++   # Version of g++
special_gpp=false # Whether a special compiler version (g++) is specified.
eigen3_dir="../thirdparty/install/share/eigen3/cmake/" # default paths to dependency Eigen3

while getopts ":hctpswre:p:d:j:g:" option; do
    case "$option" in
        h) # Display help
           Help_Info
           exit;;
        c) # Clean mode
           clean_mode=true;;
        t) # Testing
           testing=false;;
        p) # Build python interface
           build_python=false;;
        s) # Test python test-suite
           test_python=false;;
        w) # Generate wheel
           wheel=true;;
        r) # skip buildrequirements
           skip_req=true;;
        e) # path to Eigen3
           eigen3_dir=${OPTARG};;
        d) # Select a mode
           config=${OPTARG};;
        j) # Number of threads
           NT=${OPTARG};; 
        g) # gpp version
           special_gpp=true
           gpp_version=${OPTARG};;
    esac
done
# ------------------------------------------------------------------------------

# Amend the path if necessary --------------------------------------------------
# If the script is called from inside the folder helper_scripts, then place us 
# at the root directory darts-flash.
if [[ "$(basename $PWD)" == "helper_scripts" ]]; then
    cd ../
fi
# ------------------------------------------------------------------------------

# Build loop -------------------------------------------------------------------
if [[ "$clean_mode" == true ]]; then
    # Cleaning build to prepare a fresh build
    echo -e "\n   Cleaning build and install folders"
    rm -rf build install
    rm -f dartsflash/*.so
    rm -rf dist
else
    if [[ "$skip_req" == false ]]; then
        # Update submodules
        echo -e "\n- Update submodules: START\n"
        rm -rf thirdparty/eigen thirdparty/pybind11
        git submodule sync --recursive
        git submodule update --recursive --init
        echo -e "\n- Update submodules: DONE! \n"
        
        echo -e "\n-- Building eigen: START\n"
        # Setup path for building eigen
        cd thirdparty
        mkdir -p build/eigen

        # Build eigen
        cd build/eigen
        cmake -D CMAKE_INSTALL_PREFIX=../../install ../../eigen/
        make install

        # Return to root dir
        cd ../../../
        echo -e "\n-- Building eigen: DONE!\n"

        echo -e "\n-- Building pybind11: START\n"
        # Setup path for building pybind11
        cd thirdparty
        mkdir -p build/pybind11

        # Build pybind11
        cd build/pybind11
        cmake_options_pybind11=" -D CMAKE_INSTALL_PREFIX=../../install -D PYBIND11_TEST=OFF -D BUILD_TESTING=OFF"
        if [[ "$special_gpp" == true ]]; then
            cmake_options_pybind11+=" -D CMAKE_CXX_COMPILER=${gpp_version}"
        fi
        cmake $cmake_options_pybind11 ../../pybind11/
        make install

        # Return to root dir
        cd ../../../
        echo -e "\n-- Building pybind11: DONE!\n"

    else
        echo -e "\n- Requirements already installed \n"
    fi

    echo -e "\n- Building opendarts-flash: START\n"
    # Setup build folder
    mkdir -p build
    cd build

    # Setup build with cmake
    cmake_options=" -D Eigen3_DIR=${eigen3_dir} -D SET_CXX11_ABI_0=TRUE -D CMAKE_BUILD_TYPE=${config} -D CMAKE_INSTALL_PREFIX=../install"

    if [[ "$testing" == true ]]; then
        cmake_options+=" -D ENABLE_TESTING=ON"
    fi
    if [[ "$build_python" == false ]]; then
        cmake_options+=" -D OPENDARTS_FLASH_BUILD_PYTHON=FALSE"
    fi
    if [[ "$special_gpp" == true ]]; then
        cmake_options+=" -D CMAKE_CXX_COMPILER=${gpp_version}"
    fi

    echo "CMake options: $cmake_options" # Report to user the CMake options
    cmake $cmake_options ..

    # Build and install libdarts-flash
    make install -j $NT

    # Test
    if [[ "$testing" == true ]]; then
        ctest
    fi

    cd ..
    echo -e "\n- Building opendarts-flash: DONE!\n"

    if [[ "$build_python" == true ]]; then
        echo -e "\n- Building python package dartsflash: START\n"
        # Copy .so
        if [[ "$config" == "Debug" ]]; then
            cp ./install/lib/debug/*.so ./dartsflash/
        else
            cp ./install/lib/*.so ./dartsflash/
        fi

        if [[ "$test_python" == true ]]; then
            python3 -m pip install .[test]
            pytest tests/python
        else
            python3 -m pip install .
        fi

        # Build dartsflash.whl
        if [[ "$wheel" == true ]]; then
            python3 -m pip install build
            python3 -m build
            echo -e "-- Python wheel generated! \n"
        fi

        echo -e "\n- Building python package dartsflash: DONE!\n"
    fi
fi
# ------------------------------------------------------------------------------