#include "matrix.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>

#define matrix_pos(m, row, col) ((row) * (m)->n_cols + col) 

void matrix_set(matrix_t *m, int row, int col, double value) {
  int i;
  if (row > m->n_rows - 1 || col > m->n_cols) {
    fprintf(stderr, "Matrix set value error: out of bounds\n");
    exit(EXIT_FAILURE);
  }
  i = matrix_pos(m, row, col);
  m->data[i] = value;
}

double matrix_get(matrix_t *m, int row, int col) {
  int i = matrix_pos(m, row, col);
  return m->data[i];
}

matrix_t *matrix_new(int n_rows, int n_cols, int init) {
  int i, j;
  matrix_t *m = malloc(sizeof(matrix_t));
  m->data = malloc(sizeof(double) * n_rows * n_cols);
  m->n_rows = n_rows;
  m->n_cols = n_cols;
  if (init) {
    for (i = 0; i < m->n_rows; i++)
      for (j = 0; j < m->n_cols; j++)
        matrix_set(m, i, j, (double) rand() / (double) RAND_MAX);      
  }
  return m;
}

matrix_t *matrix_foreach(matrix_t *m, double (*func)(double), int mutate) {
  int i, j;
  double v;
  matrix_t *r;
  if (!mutate)
    r = matrix_new(m->n_rows, m->n_cols, FALSE);
  else
    r = m;
  for (i = 0; i < m->n_rows; i++)
    for (j = 0; j < m->n_cols; j++) {
      v = func(matrix_get(m, i, j));
      matrix_set(r, i, j, v);
    }
  return r;
}

matrix_t *matrix_foreach_set(matrix_t *m, double v, int mutate) {
  int i, j;
  matrix_t *r;
  if (!mutate)
    r = matrix_new(m->n_rows, m->n_cols, FALSE);
  else
    r = m;
  for (i = 0; i < m->n_rows; i++)
    for (j = 0; j < m->n_cols; j++) 
      matrix_set(r, i, j, v);
  return r;
}

matrix_t *matrix_hadamart_prod(matrix_t *a, matrix_t *b, int mutate) {
  int i, j;
  double v;
  matrix_t *c;
  if (a->n_cols != b->n_cols && a->n_rows != b->n_rows) {
    printf("Invalid Hadamard product: dimensions mismatch\n");
    printf("(%d, %d)-(%d, %d)\n", a->n_rows, a->n_cols, b->n_rows, b->n_cols);
    exit(EXIT_FAILURE);
  }
  if (!mutate)
    c = matrix_new(a->n_rows, a->n_cols, FALSE);
  else
    c = a;
  for (i = 0; i < a->n_rows; i++)
    for (j = 0; j < a->n_cols; j++) {
      v = matrix_get(a, i, j) * matrix_get(b, i, j);
      matrix_set(c, i, j, v);
    }
  return c;
}

matrix_t *matrix_sum(matrix_t *a, matrix_t *b, int mutate) {
  int i, j;
  double v;
  matrix_t *c;
  if (a->n_cols != b->n_cols && a->n_rows != b->n_rows) {
    printf("Invalid sum: dimensions mismatch\n");
    printf("(%d, %d)-(%d, %d)\n", a->n_rows, a->n_cols, b->n_rows, b->n_cols);
    exit(EXIT_FAILURE);
  }
  if (!mutate)
    c = matrix_new(a->n_rows, a->n_cols, FALSE);
  else
    c = a;
  for (i = 0; i < a->n_rows; i++)
    for (j = 0; j < a->n_cols; j++) {
      v = matrix_get(a, i, j) + matrix_get(b, i, j);
      matrix_set(c, i, j, v);
    }
  return c;
}

matrix_t *matrix_sum_row(matrix_t *a, double *row, int mutate) {
  int i, j;
  double v;
  matrix_t *b;
  if (!mutate)
    b = matrix_new(a->n_rows, a->n_cols, FALSE);
  else
    b = a;
  for (i = 0; i < a->n_rows; i++)
    for (j = 0; j < b->n_cols; j++) {
      v = matrix_get(a, i, j) + row[j];
      matrix_set(b, i, j, v);
    }
  return b;
}

matrix_t *matrix_scale(matrix_t *a, double factor, int mutate) {
  int i, j;
  double v;
  matrix_t *b;
  if (!mutate)
    b = matrix_new(a->n_rows, a->n_cols, FALSE);
  else
    b = a;
  for (i = 0; i < a->n_rows; i++)
    for (j = 0; j < a->n_cols; j++) {
      v = factor * matrix_get(a, i, j);
      matrix_set(b, i, j, v);
    }
  return b;
}

void matrix_free(matrix_t *m) {
  freemem(m->data);
  freemem(m);
}

void matrix_print(matrix_t *m) {
  int i, j;
  printf("Matrix(%d rows, %d cols)\n", m->n_rows, m->n_cols);
  for (i = 0; i < m->n_rows; i++) {
    for (j = 0; j < m->n_cols; j++)
      printf("%.12f ", matrix_get(m, i, j));
    printf("\n");
  }
}

matrix_t *matrix_load(const char *filename, int n_rows, int n_cols) {
  int i, j;
  double value;
  char c;
  FILE *fin = fopen(filename, "r");
  matrix_t *m = matrix_new(n_rows, n_cols, FALSE);
  for (i = 0; i < n_rows; i++)
    for (j = 0; j < n_cols; j++) {
      fscanf(fin, "%lf%c", &value, &c);
      matrix_set(m, i, j, value);
    }
  fclose(fin);
  return m;
}

struct matprod {
  int i;
  int j;
  matrix_t *a;
  matrix_t *b;
  matrix_t *m;
};

void *matrix_prod_vec(void *arg) {
  struct matprod mp = *(struct matprod *) arg;
  double sum = 0.0;
  int k;
  for (k = 0; k < mp.a->n_cols; k++) {
    sum += matrix_get(mp.a, mp.i, k) * matrix_get(mp.b, k, mp.j);
  }
  matrix_set(mp.m, mp.i, mp.j, sum);
  printf("%f ", sum);
  return NULL;
}

#define THREADS 4

matrix_t *matrix_prod(matrix_t *a, matrix_t *b) {
  int i, j, k;
  /*
  pthread_t tid[THREADS];
  int pc = 0;
  */
  matrix_t *m;
  if (a->n_cols != b->n_rows) {
    printf("Invalid product: dimensions mismatch\n");
    printf("(%d, %d)-(%d, %d)\n", a->n_rows, a->n_cols, b->n_rows, b->n_cols);
    exit(EXIT_FAILURE);
  }
  m = matrix_new(a->n_rows, b->n_cols, FALSE);
  #pragma omp target
  #pragma omp loop
  for (i = 0; i < a->n_rows; i++) 
    for (j = 0; j < b->n_cols; j++)  {
      double value = 0.0;
      for (k = 0; k < a->n_cols; k++)
        value += matrix_get(a, i, k) * matrix_get(b, k, j);
      matrix_set(m, i, j, value);
      /*
      struct matprod mp;
      mp.i = i;
      mp.j = j;
      mp.a = a;
      mp.b = b;
      mp.m = m;
      pthread_create(&tid[pc], NULL, &matrix_prod_vec, &mp);
      pc++;
      if (pc == THREADS) {
        for (k = 0; k < THREADS; k++)
          pthread_join(tid[k], NULL);
        pc = 0;
      }
      */
    }
  return m;
}

matrix_t *matrix_transpose(matrix_t *mat) {
  int i, j;
  matrix_t *m = matrix_new(mat->n_cols, mat->n_rows, FALSE);
  for (i = 0; i < m->n_rows; i++)
    for (j = 0; j < m->n_cols; j++)
      matrix_set(m, i, j, matrix_get(mat, j, i));
  return m;
}

matrix_t *matrix_center(matrix_t *mat, int mutate) {
  int i, j;
  double sum, mean, dev, v;
  matrix_t *res;
  if (mutate)
    res = mat;
  else
    res = matrix_new(mat->n_rows, mat->n_cols, FALSE);
  for (j = 0; j < mat->n_cols; j++) {
    sum = 0.0;
    for (i = 0; i < mat->n_rows; i++)
      sum += matrix_get(mat, i, j);
    mean = sum / mat->n_rows;
    sum = 0.0;
    for (i = 0; i < mat->n_rows; i++) {
      v = matrix_get(mat, i, j) - mean;
      sum += v * v;
    }
    dev = sqrt(sum / mat->n_rows);
    for (i = 0; i < mat->n_rows; i++) {
      v = (matrix_get(mat, i, j) - mean) / dev;
      matrix_set(res, i, j, v);
    }
  }
  return res;
}
