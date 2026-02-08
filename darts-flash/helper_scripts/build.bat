@echo off
setlocal enabledelayedexpansion

REM Read input arguments ---------------------------------------------
set clean_mode=false
set testing=true
set wheel=false
set build_python=true
set test_python=true
set skip_req=false
set config=Release
set NT=8

:parse_args
if "%~1"=="" goto :process_input
set option=%1
shift
if "%option%"=="-h" goto :help_info
if "%option%"=="-c" set clean_mode=true & goto parse_args
if "%option%"=="-t" set testing=false & goto parse_args
if "%option%"=="-w" set wheel=true & goto parse_args
if "%option%"=="-p" set build_python=false & goto parse_args
if "%option%"=="-s" set test_python=false & goto parse_args
if "%option%"=="-r" set skip_req=true & goto parse_args
if "%option%"=="-d" set config=%1 & shift & goto parse_args
if "%option%"=="-j" set NT=%1 & shift & goto parse_args
goto parse_args

:process_input
echo - Report configuration of this script: START
echo    config = %config%
echo    testing = %testing%
echo    generate python wheel = %wheel%
echo - Report configuration of this script: DONE!
REM ----------------------------------------------------------------

if %clean_mode%==true (
  echo - Cleaning up
  rmdir /s /q build 2> NUL
  rmdir /s /q install 2> NUL
  rmdir /s /q dist 2> NUL
  del dartsflash\*.pyd 2> NUL
  goto :eof
)

if %skip_req%==false (
    echo - Update submodules: START
    rmdir /s /q thirdparty/eigen thirdparty/pybind11 2> NUL
    git submodule update --recursive --init || goto :error
    echo - Update submodules: DONE!

    echo - Install requirements: START
    echo -- Build Eigen 3
    cd thirdparty
    mkdir build
    cd build
    mkdir eigen
    cd eigen
    cmake -D CMAKE_INSTALL_PREFIX=../../install ../../eigen/ || goto :error
    msbuild INSTALL.vcxproj /p:Configuration=Release /p:Platform=x64 -maxCpuCount:1 || goto :error
    cd ..\..\..
    echo - Install requirements: DONE ! 
)

echo - Building open-darts-flash: START
rmdir build 2> NUL
mkdir build
cd build

REM Setup build with CMake
set cmake_options=-DCMAKE_INSTALL_PREFIX=../install -DEigen3_DIR=../thirdparty/install/share/eigen3/cmake/ -DCMAKE_BUILD_TYPE=%config% -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE
if %testing%==true (
  set cmake_options=%cmake_options% -D ENABLE_TESTING=TRUE
)
if %build_python%==true (
  set cmake_options=%cmake_options% -D OPENDARTS_FLASH_BUILD_PYTHON=TRUE
) else (
  set cmake_options=%cmake_options% -D OPENDARTS_FLASH_BUILD_PYTHON=FALSE
)
echo CMake options: %cmake_options%

cmake %cmake_options% .. || goto :error

REM build and install
msbuild INSTALL.vcxproj /p:Configuration=Release /p:Platform=x64 -maxCpuCount:8 || goto :error

if %testing%==true ctest -C Release

cd ..
echo - Building open-darts-flash: DONE!

if %build_python%==true (
  echo - Building python package darts-flash: START
  REM copy and rename pyd
  copy install\lib\*.pyd dartsflash || goto :error

  if %wheel%==true (
    REM make sure wheel package could be installed
    python -m pip install wheel 
    python -m pip install build 
    python -m build || goto :error
    echo -- Python wheel generated!
  )

  REM test python test-suite
  if %test_python%==true (
    python -m pip install .[test]
    pytest tests\python
  ) else (
    python -m pip install .
  )

  echo - Building python package darts-flash: DONE!
)

rem || goto :error checks exit code of command 
rem if one of the commands fails, interrupt batch and return error code
:error
echo Build finished with error code %errorlevel%.
exit /b %errorlevel%
goto :eof

REM Help info --------------------------------------------------------
:help_info
echo helper_scripts\build.bat [-h] [-c] [-t] [-w] [-r] [-d INSTALL CONFIGURATION] [-j NUM THREADS]
echo    Script to install dartsflash on Windows.
echo USAGE: 
echo    -h : displays this help menu.
echo    -c : cleans up build to prepare a new fresh build. Default: don't clean
echo    -t : Skip testing: ctest of solvers. Default: test
echo    -w : Enable generation of python wheel. Default: false
echo    -r : Skip building thirdparty libraries (if you have them already compiled). Default: false
echo    -d MODE   : Configuration for C++ code [Release, Debug]. Example: -d Debug
echo    -j N      : Set number of threads (N) for compilation. Default: 8. Example: -j 4
goto :eof
REM ----------------------------------------------------------------
