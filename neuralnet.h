#ifndef _NEURALNET_H
#define _NEURALNET_H 1

#include "matrix.h"

typedef struct {
  int n_layers;
  int *n_nodes;
  matrix_t **w;
  matrix_t **b;
  matrix_t **z;
  matrix_t **a;
  double (*actfunc)(double);
  double (*d_actfunc)(double);
} neuralnet_t;

double sigmoid(double x);
double d_sigmoid(double x);

neuralnet_t *neuralnet_new(int *n_nodes, int n_layers, 
  double (*actfunc)(double), double (*d_actfunc)(double));
void neuralnet_free(neuralnet_t *nn);
void neuralnet_forward(neuralnet_t *nn, matrix_t *X);
void neuralnet_backpropagation(neuralnet_t *nn, matrix_t *X, matrix_t *Y, double nabla);
void neuralnet_fit(neuralnet_t *nn, matrix_t *X, matrix_t *Y, int epochs, double nabla);
matrix_t *neuralnet_predict(neuralnet_t *nn, matrix_t *X);

matrix_t *model_error(matrix_t *Y, matrix_t *Y_hat);
#endif /* _NEURALNET_H */
