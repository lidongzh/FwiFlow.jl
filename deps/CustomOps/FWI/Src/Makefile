CXX=nvcc
CFLAGS=-O3 -std=c++11 -I/usr/local/cuda/include -I../rapidjson -Xcompiler -fPIC
LFLGAS=-L/usr/local/cuda/lib64 -lnvrtc -lcuda -lcudart -lcufft

BUILD_DIR="./build/"

CUFD_CPP_SRC= Parameter.cpp
CUFD_CU_SRC= libCUFD.cu el_stress.cu el_velocity.cu \
	el_stress_adj.cu el_velocity_adj.cu Model.cu Cpml.cu utilities.cu \
	Src_Rec.cu Boundary.cu

CUFD_CPP_OBJ:= $(patsubst %.cpp,${BUILD_DIR}%.o,$(CUFD_CPP_SRC))
CUFD_CU_OBJ:= $(patsubst %.cu,${BUILD_DIR}%.o,$(CUFD_CU_SRC))
CUFD_HEADERS= Parameter.h Model.h Cpml.h utilities.h Src_Rec.h Boundary.h

${BUILD_DIR}%.o: %.cpp $(CUFD_HEADERS)
	$(CXX) -c $< $(CFLAGS) $(LFLGAS) -o $@

${BUILD_DIR}%.o: %.cu $(CUFD_HEADERS)
	$(CXX) -c $< $(CFLAGS) $(LFLGAS) -o $@

libCUFD: $(CUFD_CU_OBJ) $(CUFD_CPP_OBJ) $(CUFD_HEADERS)
	$(CXX) -shared $(CUFD_CU_OBJ) $(CUFD_CPP_OBJ) $(CFLAGS) $(LFLGAS) -o ${BUILD_DIR}libCUFD.so

clean:
	rm -f ${BUILD_DIR}*.o ${BUILD_DIR}libCUFD

make_dir:
	@mkdir -p ${BUILD_DIR}

all: clean make_dir libCUFD

