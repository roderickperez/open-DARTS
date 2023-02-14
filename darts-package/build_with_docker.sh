#!/bin/bash
CUDA_VERSION=11.2.2
DARTS_ROOT=`realpath ..`
ENGINES=$DARTS_ROOT/darts-engines
SOLVERS=$DARTS_ROOT/darts-linear-solvers
AMGX=$SOLVERS/lib/AMGX
PHYSICS=$DARTS_ROOT/darts-physics
FLASH=$DARTS_ROOT/darts-flash
PACKAGE=$DARTS_ROOT/darts-package


# build AMGX
# cd $AMGX
# rm -rf build
# mkdir build
# docker run -v $AMGX:/amgx -w /amgx/build -it --rm cuda$CUDA_VERSION-devel-py cmake ../ -DCUDA_ARCH="75 86" -DCMAKE_NO_MPI=True
# docker run -v $AMGX:/amgx -w /amgx/build -it --rm cuda$CUDA_VERSION-devel-py make -j 16 amgxsh
# cp $AMGX/build/libamgxsh.so $PACKAGE/darts

# build SOLVERS
# cd $SOLVERS
# make clean
# docker run -v $DARTS_ROOT:/darts -w /darts/darts-linear-solvers -it --rm cuda$CUDA_VERSION-devel-py make -j 16 gpu

# # # build ENGINES
# cd $ENGINES
# make clean
# ./update_private_artifacts_local.sh
# docker run -v $DARTS_ROOT:/darts -w /darts/darts-engines -it --rm cuda$CUDA_VERSION-devel-py make -j 10 gpu

# #build FLASH
# cd $FLASH
# make clean
# docker run -v $DARTS_ROOT:/darts -w /darts/darts-flash -it --rm cuda$CUDA_VERSION-devel-py make -j 10


# # build PHYSICS
# cd $PHYSICS
# make clean
# ./update_private_artifacts_local.sh
# docker run -v $DARTS_ROOT:/darts -w /darts/darts-physics -it --rm cuda$CUDA_VERSION-devel-py make -j 10

# build PACKAGE
docker run -v $DARTS_ROOT:/darts -w /darts/darts-package -it --rm cuda$CUDA_VERSION-devel-py python3 setup.py clean
docker run -v $DARTS_ROOT:/darts -w /darts/darts-package -it --rm cuda$CUDA_VERSION-devel-py python3 setup.py build bdist_wheel


docker build --rm -f "Dockerfile_deploy" -t darts_cuda$CUDA_VERSION-deploy2 --build-arg VERSION=$CUDA_VERSION "."

#run deploy docker image on target system
#docker run --rm --gpus all -w /dcse2020 -e PYTHONPATH=/dcse2020 dcse2020:cuda10.2-deploy python3 src/main.py
