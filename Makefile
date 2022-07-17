CC=gcc
#LFLAGS?=-Wall -lpthread mlpack/build/lib/libmlpack.so -larmadillo 
#LDFLAGS=-L./mlpack/build/lib
#CFLAGS=-std=c++0x -Wall -D __STDC_LIMIT_MACROS -D __STDC_FORMAT_MACROS -O3 -g
CFLAGS=-std=c++0x -Wall -D __STDC_LIMIT_MACROS -D __STDC_FORMAT_MACROS -O0 -g3
#CFLAGS+=-std=c++11 -fopenmp -larmadillo 
#CFLAGS+=-lmlpack -larmadillo -lboost_serialization -std=c++11 -fopenmp
CXX=g++

INCLUDE=-Iminisat -Iminisat/minisat/core \
		-Iminisat/minisat/mtl -Iminisat/minisat/simp \
		-Iaiger -Imlpack/build/include/ \
#-Iensmallen/build/include

all:	ic3

ic3:	minisat/build/dynamic/lib/libminisat.so aiger/aiger.o Model.o IC3.o model2graph.o
	$(CXX) $(CFLAGS) $(INCLUDE) -o IC3 \
		aiger.o Model.o IC3.o model2graph.o \
		minisat/build/release/lib/libminisat.a \
		-lz3

#-lmlpack \
#-lboost_serialization \
#-fopenmp \
#-larmadillo \
#-fpermissive

.c.o:
	$(CC) -O0 -g3 $(INCLUDE) $< -c 

.cpp.o:	
	$(CXX) $(CFLAGS) $(INCLUDE) $< -c

clean:
	rm -f *.o ic3

dist:
	cd ..; tar cf ic3ref/IC3ref.tar ic3ref/*.h ic3ref/*.cpp ic3ref/Makefile ic3ref/LICENSE ic3ref/README; gzip ic3ref/IC3ref.tar
