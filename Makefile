CFLAGS = -Wall -g -std=c99 -pedantic -O3
LDLIBS = -lm -lpthread

all: irisnn winenn mnistnn

irisnn: matrix.o neuralnet.o

winenn: matrix.o neuralnet.o

clean:
	rm -f irisnn winenn mnistnn
	rm -f matrix.o neuralnet.o
