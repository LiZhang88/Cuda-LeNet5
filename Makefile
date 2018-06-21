all:
	/usr/local/cuda/bin/nvcc *.cu  -arch=compute_20 #-Wno-deprecated-gpu-targets

run:
	./CNN
clean:
	rm CNN
