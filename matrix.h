#ifndef _MATRIX_H
#define _MATRIX_H 1

/* Routines for matrix algreba
 *
 * The purpose of these routines is to provide support in building multilayer
 * perceptrons to avoid dependencies and keeping the code light.They have a
 * educative character, i.e. they are not intetended to be used in research or
 * production environments.
 *
 * (c) 2021 - Jaime Lopez <jailop AT protonmail DOT com>
 */

typedef struct {
  double *data;
  int n_rows;
  int n_cols;
} matrix_t;

matrix_t *matrix_new(int n_rows, int n_cols, int init);
matrix_t *matrix_load(const char *filename, int n_rows, int n_cols);
void matrix_print(matrix_t *m);
void matrix_free(matrix_t *m);
double matrix_get(matrix_t *m, int row, int col);
void matrix_set(matrix_t *m, int row, int col, double value);
matrix_t *matrix_foreach_set(matrix_t *m, double v, int mutate);
matrix_t *matrix_foreach(matrix_t *m, double (*func)(double), int mutate);
matrix_t *matrix_transpose(matrix_t *mat);
matrix_t *matrix_center(matrix_t *mat, int mutate);
matrix_t *matrix_scale(matrix_t *a, double factor, int mutate);
matrix_t *matrix_sum(matrix_t *a, matrix_t *b, int mutate);
matrix_t *matrix_sum_row(matrix_t *a, double *row, int mutate);
matrix_t *matrix_hadamart_prod(matrix_t *a, matrix_t *b, int mutate);
matrix_t *matrix_prod(matrix_t *a, matrix_t *b);

#endif /* _MATRIX_H */
