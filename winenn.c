#include <stdio.h>
#include <stdlib.h>
#include "neuralnet.h"
#include "utils.h"

#define N_ROWS 178
#define N_COLS 13
#define LAYERS {13, 8, 3}
#define M 3

int main(int argc, char **argv) {
  int i;
  int n_nodes[M] = LAYERS;
  int epochs = atoi(argv[1]);
  int nabla = atof(argv[2]);
  neuralnet_t *nn;
  matrix_t *error;
  matrix_t *y_hat;
  matrix_t *X = matrix_load("wine/attrs.txt", N_ROWS, N_COLS);
  matrix_t *y = matrix_load("wine/labels.txt", N_ROWS, 1);
  matrix_t *Y = matrix_new(N_ROWS, 3, FALSE);
  for (i = 0; i < N_ROWS; i++) {
    matrix_set(Y, i, 0, matrix_get(y, i, 0) == 1 ? 1.0 : 0.0);
    matrix_set(Y, i, 1, matrix_get(y, i, 0) == 2 ? 1.0 : 0.0);
    matrix_set(Y, i, 2, matrix_get(y, i, 0) == 3 ? 1.0 : 0.0);
  }
  matrix_center(X, TRUE);
  nn = neuralnet_new(n_nodes, M, sigmoid, d_sigmoid);
  neuralnet_fit(nn, X, Y, epochs, nabla);
  y_hat = neuralnet_predict(nn, X);
  error = model_error(Y, y_hat);
  matrix_print(error);
  matrix_free(error);
  matrix_free(y_hat);
  matrix_free(X);
  matrix_free(Y);
  neuralnet_free(nn);
  return 0;
}
