#include <stdio.h>
#include <stdlib.h>
#include "neuralnet.h"
#include "utils.h"

#define IRIS_ATTR_FILE "iris/iris-attributes.csv"
#define N_ROWS 150
#define N_COLS 4
#define LAYERS {4, 8, 5, 3}
#define M 4

int main(int argc, char **argv) {
  int i;
  int n_nodes[M] = LAYERS;
  if (argc < 3) {
    printf("Usage: %s EPOCHS LEARNING_RATE\n", argv[0]);
    exit(EXIT_FAILURE);
  }
  int epochs = atoi(argv[1]);
  double nabla = atof(argv[2]);
  neuralnet_t *nn;
  matrix_t *error;
  matrix_t *y_hat;
  matrix_t *X = matrix_load("iris/attrs.txt", 150, 4);
  /* Three columns are build for output data.
     one for each flower kind. */
  matrix_t *Y = matrix_new(N_ROWS, 3, FALSE);
  for (i = 0; i < N_ROWS; i++) {
    matrix_set(Y, i, 0, i < 50 ? 1.0 : 0.0);
    matrix_set(Y, i, 1, i >= 50 && i < 100 ? 1.0 : 0.0);
    matrix_set(Y, i, 2, i >= 100 ? 1.0 : 0.0);
  }
  /* Data is centered by mean and
     standard deviation adjustment */
  matrix_center(X, TRUE);
  nn = neuralnet_new(n_nodes, M, sigmoid, d_sigmoid);
  neuralnet_fit(nn, X, Y, epochs, nabla);
  y_hat = neuralnet_predict(nn, X);
  error = model_error(Y, y_hat);
  /*
  matrix_print(y_hat);
  */
  matrix_print(error);
  matrix_free(error);
  matrix_free(y_hat);
  matrix_free(X);
  matrix_free(Y);
  neuralnet_free(nn);
  return EXIT_SUCCESS;
}
