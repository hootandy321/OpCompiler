CC = g++
ARCH = sm_80

TARGET = build/stream
SRC = stream.cu

$(TARGET): $(SRC)
	mkdir -p build
	nvcc -std=c++11 -ccbin=$(CC) $(SRC) -arch=$(ARCH) -o $(TARGET)

.PHONY: clean
clean:
	rm -rf build
