CXX=nvcc
CFLAGS=-O3 -std=c++11 -I/usr/local/cuda/include -I../rapidjson
LFLGAS=-L/usr/local/cuda/lib64 -lnvrtc -lcuda -lcudart -lcufft

#CXX=/cm/shared/apps/cuda90/toolkit/9.0.176/bin/nvcc
#CFLAGS=-O3 -std=c++11 -I/cm/shared/apps/cuda90/toolkit/9.0.176/include -I../rapidjson
#LFLGAS=-L/cm/shared/apps/cuda90/toolkit/9.0.176/lib64 -lnvrtc -lcuda -lcudart -lcufft

CUFD_CPP_SRC= Parameter.cpp
CUFD_CU_SRC= main.cu add_source.cu el_stress.cu el_velocity.cu \
	ac_pressure.cu ac_velocity.cu Model.cu Cpml.cu utilities.cu \
	Src_Rec.cu recording.cu Boundary.cu ac_pressure_adj.cu ac_velocity_adj.cu \
	res_injection.cu image_vel.cu image_vel_time.cu ac_adj_push.cu

CUFD_CPP_OBJ:= $(patsubst %.cpp,%.o,$(CUFD_CPP_SRC))
CUFD_CU_OBJ:= $(patsubst %.cu,%.o,$(CUFD_CU_SRC))
CUFD_HEADERS= Parameter.h Model.h Cpml.h utilities.h Src_Rec.h Boundary.h

%.o: %.cpp $(CUFD_HEADERS)
	$(CXX) -c $< $(CFLAGS) $(LFLGAS) -o $@

%.o: %.cu $(CUFD_HEADERS)
	$(CXX) -c $< $(CFLAGS) $(LFLGAS) -o $@

CUFD: $(CUFD_CU_OBJ) $(CUFD_CPP_OBJ) $(CUFD_HEADERS)
	$(CXX) $(CUFD_CU_OBJ) $(CUFD_CPP_OBJ) $(CFLAGS) $(LFLGAS) -o CUFD


clean:
	rm -f *.o CUFD

all: clean CUFD

