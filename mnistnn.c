#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "matrix.h"

#define TRAIN_ATTRS "mnist/train-images-idx3-ubyte"
#define TRAIN_LABELS "mnist/train-labels-idx1-ubyte"
#define TEST_ATTRS "mnist/mnist/t10k-images-idx3-ubyte"
#define TEST_LABELS "mnist/t10k-labels-idx1-ubyt"

uint32_t swapint(uint32_t n) {
  uint32_t a = n & 0xff;
  uint32_t b = n >> 8 & 0xff;
  uint32_t c = n >> 16 & 0xff;
  uint32_t d = n >> 24 & 0xff;
  return a << 24 | b << 16 | c << 8 | d;
}

uint32_t intread(FILE *fin) {
  uint32_t n;
  fread(&n, 1, 4, fin);
  return swapint(n);
}

matrix_t *load_X_train() {
  uint32_t n;
  unsigned char c[5];
  unsigned char *d;
  uint32_t rows, cols;
  FILE *fin = fopen(TRAIN_ATTRS, "rb");
  if (intread(fin) != 0x803) {
    fprintf(stderr, "Incorrect file\n");
    exit(EXIT_FAILURE);
  }
  n = intread(fin);
  rows = intread(fin);
  cols = intread(fin);
  d = c;
  c[4] = 0;
  printf("%s\n", c);
  printf("%d %d %d\n", n, rows, cols);
  fclose(fin);
  return NULL;
}

int main() {
  load_X_train();
  return 0;
}
