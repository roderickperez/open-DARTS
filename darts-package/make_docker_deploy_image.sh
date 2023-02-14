CUDA_VERSION=11.2.2

docker build --rm -f "Dockerfile_deploy" -t darts_cuda$(CUDA_VERSION)-deploy --build-arg cuda-version=$(CUDA_VERSION) "."