all: *.cpp *.cu
	nvcc $< *.cu -g -o result `pkg-config --cflags --libs cuda`
