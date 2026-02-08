:: Compilation file for MinGW on Windows

:parameters
set CONFIG=Release

:compilers
set CC=gcc
set CXX=g++

:directories
set FLASH_DIR=%~dp0..\
echo %FLASH_DIR%

:submodules
rem - Update submodules: START
if exist %FLASH_DIR%thirdparty\eigen (
    rmdir /s /q %FLASH_DIR%thirdparty\eigen
)
if exist %FLASH_DIR%thirdparty\pybind11 (
    rmdir /s /q %FLASH_DIR%thirdparty\pybind11
)
git submodule update --recursive --remote --init
rem - Update submodules: DONE!

rem - Install requirements: START
if not exist %FLASH_DIR%thirdparty\build (
    mkdir %FLASH_DIR%thirdparty\build
)
if not exist %FLASH_DIR%thirdparty\install (
    mkdir %FLASH_DIR%thirdparty\install
)

:eigen3
rem -- Build Eigen 3
if not exist %FLASH_DIR%thirdparty\build\eigen (
    mkdir %FLASH_DIR%thirdparty\build\eigen
)
if not exist %FLASH_DIR%thirdparty\install\eigen (
    mkdir %FLASH_DIR%thirdparty\install\eigen
)

chdir %FLASH_DIR%thirdparty\build\eigen
cmake ^
  -G "Unix Makefiles" ^
  -D CMAKE_INSTALL_PREFIX=%FLASH_DIR%thirdparty\install\eigen ^
  -D CMAKE_BUILD_TYPE=%CONFIG% ^
   %FLASH_DIR%thirdparty\eigen || goto :error

make install || goto :error

chdir %FLASH_DIR%
rem - Install requirements: DONE ! 

:flash
rem - Building open-darts-flash: START
if not exist %FLASH_DIR%build (
    mkdir %FLASH_DIR%build
)
del /S/Q %FLASH_DIR%build\*
chdir %FLASH_DIR%build

cmake ^
  -G "Unix Makefiles" ^
  -D CMAKE_INSTALL_PREFIX=%FLASH_DIR%..\install ^
  -D CMAKE_BUILD_TYPE=%CONFIG% ^
  -D Eigen3_DIR=%FLASH_DIR%thirdparty\install\eigen\share\eigen3\cmake\ ^
  -D CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE ^
  -D ENABLE_TESTING=TRUE ^
   %FLASH_DIR% || goto :error

make -j8 install || goto :error

rem - Tests
ctest -C Release
chdir %FLASH_DIR%
rem - Building open-darts-flash: DONE!
goto :end

:python
rem - Building python package darts-flash: START
rem Copy and rename pyd
chdir %FLASH_DIR%\build
if not exist %FLASH_DIR%build\dartsflash (
    mkdir %FLASH_DIR%build\dartsflash
)
copy %FLASH_DIR%..\install\lib\*.pyd dartsflash || goto :error
copy %FLASH_DIR%dartsflash\* dartsflash || goto :error

rem make sure wheel package could be installed
python -m pip install wheel 
python -m pip install build 

python %FLASH_DIR%setup.py build bdist_wheel --plat-name=win-amd64 || goto :error
rem -- Python wheel generated!

rem install python package
python -m pip install %FLASH_DIR%

rem - Building python package darts-flash: DONE!

rem || goto :error checks exit code of command 
rem if one of the commands fails, interrupt batch and return error code

:error
echo Build finished with error code %errorlevel%.
exit /b %errorlevel%
chdir %FLASH_DIR%

:end

