rem update submodules
git submodule update --recursive --remote --init


rem 2 Compile engines
cd darts-engines
msbuild darts-engines.vcxproj /p:Configuration=Release /p:Platform=x64 -maxCpuCount:2 
cd ..

rem 3 Compile discretizer
cd darts-discretizer
msbuild darts-discretizer.vcxproj /p:Configuration=Release /p:Platform=x64 -maxCpuCount:2 
cd ..

rem 4 Build wheel
echo 'build darts.whl for windows started'
rem copy VS redist libraries 
rem copy $env:VCToolsRedistDir\x64\Microsoft.VC143.CRT\msvcp140.dll .\darts
rem copy $env:VCToolsRedistDir\x64\Microsoft.VC143.CRT\vcruntime140.dll .\darts
rem copy $env:VCToolsRedistDir\x64\Microsoft.VC143.OpenMP\vcomp140.dll .\darts
python darts-package/darts/print_build_info.py
python setup.py build bdist_wheel --plat-name=win-amd64