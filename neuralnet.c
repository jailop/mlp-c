#include "neuralnet.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

double d_sigmoid(double x) {
  double value = sigmoid(x);
  return value * (1 - value);
}

neuralnet_t *neuralnet_new(int *n_nodes, int n_layers,
  double (*actfunc)(double), double (*d_actfunc)(double)) {
  int i;
  neuralnet_t *nn = malloc(sizeof(neuralnet_t));
  nn->n_layers = n_layers;
  nn->n_nodes = malloc(sizeof(int) * n_layers); 
  for (i = 0; i < n_layers; i++)
    nn->n_nodes[i] = n_nodes[i];
  nn->w = malloc(sizeof(matrix_t *) * n_layers);
  nn->b = malloc(sizeof(matrix_t *) * n_layers);
  nn->z = malloc(sizeof(matrix_t *) * n_layers);
  nn->a = malloc(sizeof(matrix_t *) * n_layers);
  for (i = 1; i < n_layers; i++) {
    nn->w[i] = matrix_new(nn->n_nodes[i - 1], nn->n_nodes[i], TRUE);
    nn->b[i] = matrix_new(1, nn->n_nodes[i], TRUE);
    nn->z[i] = matrix_new(1, nn->n_nodes[i], FALSE);
    nn->a[i] = matrix_new(1, nn->n_nodes[i], FALSE);
  }
  nn->actfunc = actfunc;
  nn->d_actfunc = d_actfunc;
  return nn;
}

void neuralnet_free(neuralnet_t *nn) {
  int i;
  for (i = 1; i < nn->n_layers; i++) {
    matrix_free(nn->w[i]);
    matrix_free(nn->b[i]);
    matrix_free(nn->z[i]);
    matrix_free(nn->a[i]);
  }
  freemem(nn->w);
  freemem(nn->b);
  freemem(nn->z);
  freemem(nn->a);
  freemem(nn->n_nodes);
  freemem(nn);
}

void neuralnet_forward(neuralnet_t *nn, matrix_t *X) {
  int m;
  matrix_t *m1;
  nn->a[0] = X; 
  for (m = 1; m < nn->n_layers; m++) {
    matrix_free(nn->z[m]);
    matrix_free(nn->a[m]);
    m1 = matrix_prod(nn->a[m - 1], nn->w[m]);
    nn->z[m] = matrix_sum_row(m1, nn->b[m]->data, FALSE);
    nn->a[m] = matrix_foreach(nn->z[m], nn->actfunc, FALSE);
    matrix_free(m1);
  }
}

void neuralnet_backpropagation(neuralnet_t *nn, matrix_t *X, matrix_t *Y, double nabla) {
  int m;
  matrix_t *m1, *m2, *m3;
  neuralnet_forward(nn, X);
  /* dC/da */
  m1 = matrix_scale(Y, -1.0, FALSE);
  matrix_sum(m1, nn->a[nn->n_layers - 1], TRUE);
  for (m = nn->n_layers - 1; m > 0; m--) {
    /* da / dz */
    m2 = matrix_foreach(nn->z[m], nn->d_actfunc, FALSE);
    matrix_hadamart_prod(m1, m2, TRUE);
    matrix_free(m2);
    /* Updating w */
    m2 = matrix_transpose(nn->a[m - 1]);
    m3 = matrix_prod(m2, m1);
    matrix_scale(m3, -(nabla / m1->n_rows), TRUE);
    matrix_sum(nn->w[m], m3, TRUE);
    matrix_free(m2);
    matrix_free(m3);
    /* Updating b */
    m2 = matrix_new(1, m1->n_rows, FALSE);
    matrix_foreach_set(m2, 1, TRUE);
    m3 = matrix_prod(m2, m1);
    matrix_scale(m3, -(nabla / m1->n_rows), TRUE);
    matrix_sum(nn->b[m], m3, TRUE);
    matrix_free(m2);
    matrix_free(m3);
    /* Updating persistent chain */
    m2 = matrix_transpose(nn->w[m]);
    m3 = matrix_prod(m1, m2);
    matrix_free(m2);
    matrix_free(m1);
    m1 = m3;
  }
  matrix_free(m1);
}

matrix_t *model_error(matrix_t *Y, matrix_t *Y_hat) {
  matrix_t *m1, *m2, *m3;
  m1 = matrix_scale(Y, -1.0, FALSE);
  matrix_sum(m1, Y_hat, TRUE);
  m2 = matrix_transpose(m1);
  m3 = matrix_prod(m2, m1);
  matrix_scale(m3, 1.0 / (Y->n_rows * Y->n_cols), TRUE);
  matrix_free(m1);
  matrix_free(m2);
  return m3;
}

void neuralnet_fit(neuralnet_t *nn, matrix_t *X, matrix_t *Y, int epochs, double nabla) {
  int i;
  for (i = 0; i < epochs; i++)
    neuralnet_backpropagation(nn, X, Y, nabla);
}

matrix_t *neuralnet_predict(neuralnet_t *nn, matrix_t *X) {
  matrix_t *y_hat;
  neuralnet_forward(nn, X);
  y_hat = matrix_scale(nn->a[nn->n_layers - 1], 1.0, FALSE);
  return y_hat;
}
