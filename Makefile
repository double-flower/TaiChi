all: spmv

#CUDA_PARAMETERS
ARCH = 70 
NVCC_FLAGS = -O3 -w -m64  -Xcompiler -fopenmp -gencode=arch=compute_$(ARCH),code=sm_$(ARCH) -gencode=arch=compute_$(ARCH),code=compute_$(ARCH)
CUDA_INSTALL_PATH = /usr/local/cuda-10.0
CUDA_CC = ${CUDA_INSTALL_PATH}/bin/nvcc
CUDA_INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/samples/common/inc 
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcusparse -lgomp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml

TaiChi.o: TaiChi.cu
	$(CUDA_CC) $(NVCC_FLAGS) -o TaiChi.o -c TaiChi.cu $(CUDA_INCLUDES) 
main.o: main.cu
	$(CUDA_CC) -ccbin g++ $(NVCC_FLAGS) -o main.o -c main.cu  $(CUDA_INCLUDES) 
morohutils.o: morohutils.cu
	$(CUDA_CC) -ccbin g++ $(NVCC_FLAGS) -o morohutils.o -c morohutils.cu $(CUDA_INCLUDES) 
mmio.o: mmio.cpp
	$(CUDA_CC) -ccbin g++ $(NVCC_FLAGS) -o mmio.o -c mmio.cpp $(CUDA_INCLUDES) 
spmv: TaiChi.o main.o morohutils.o mmio.o
	$(CUDA_CC) $(NVCC_FLAGS) TaiChi.o main.o morohutils.o mmio.o -o spmv $(CUDA_INCLUDES) $(CUDA_LIBS) 
clean:
	rm *.o 
