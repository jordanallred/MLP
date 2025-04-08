#ifndef MATRIX_H
#define MATRIX_H

typedef struct
{
    double ** pp_data;
    int       rows;
    int       columns;
} matrix_t;

matrix_t * matrix_create (int rows, int columns);
void       matrix_free (matrix_t * p_matrix);
matrix_t * matrix_multiply (matrix_t * p_matrix_1, matrix_t * p_matrix_2);
matrix_t * matrix_add (matrix_t * p_matrix_1, matrix_t * p_matrix_2);
void       matrix_visualize (matrix_t * p_matrix);
matrix_t * matrix_copy (matrix_t * p_input);

#endif // MATRIX_H
