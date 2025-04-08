#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double ** allocate_2d_array (int rows, int columns)
{
    double ** p_array = calloc(rows, sizeof(double *));

    for (int index = 0; index < rows; index++)
    {
        p_array[index] = calloc(columns, sizeof(double));
    }

    return p_array;
}

static double ** copy_2d_array (double ** p_array_1, int rows, int columns)
{
    double ** p_array_2 = calloc(rows, sizeof(double *));

    for (int row = 0; row < rows; row++)
    {
        p_array_2[row] = calloc(columns, sizeof(double));
        for (int column = 0; column < rows; column++)
        {
            p_array_2[row][column] = p_array_1[row][column];
        }
    }

    return p_array_2;
}

static void free_2d_array (double ** array, int rows)
{
    for (int row = 0; row < rows; row++)
    {
        free(array[row]);
    }
    free(array);
}

matrix_t * matrix_create (int rows, int columns)
{
    matrix_t * p_matrix = calloc(1, sizeof(matrix_t));
    p_matrix->rows      = rows;
    p_matrix->columns   = columns;
    p_matrix->pp_data   = allocate_2d_array(rows, columns);

    return p_matrix;
}

void matrix_free (matrix_t * p_matrix)
{
    free_2d_array(p_matrix->pp_data, p_matrix->rows);
    free(p_matrix);
}

matrix_t * matrix_multiply (matrix_t * p_matrix_1, matrix_t * p_matrix_2)
{
    if (p_matrix_1->columns != p_matrix_2->rows)
    {
        (void)fprintf(stderr,
                      "Incompatible matrices (%i, %i) • (%i, %i)\n",
                      p_matrix_1->rows,
                      p_matrix_1->columns,
                      p_matrix_2->rows,
                      p_matrix_2->columns);
        return NULL;
    }

    matrix_t * p_matrix_3 = calloc(1, sizeof(matrix_t));

    p_matrix_3->rows    = p_matrix_1->rows;
    p_matrix_3->columns = p_matrix_2->columns;
    p_matrix_3->pp_data
        = allocate_2d_array(p_matrix_3->rows, p_matrix_3->columns);

    for (int row = 0; row < p_matrix_3->rows; row++)
    {
        for (int column = 0; column < p_matrix_3->columns; column++)
        {
            for (int index = 0; index < p_matrix_1->columns; index++)
            {
                p_matrix_3->pp_data[row][column]
                    += (p_matrix_1->pp_data[row][index]
                        * p_matrix_2->pp_data[index][column]);
            }
        }
    }

    return p_matrix_3;
}

matrix_t * matrix_add (matrix_t * p_matrix_1, matrix_t * p_matrix_2)
{
    if ((p_matrix_1->rows != p_matrix_2->rows)
        || (p_matrix_1->columns != p_matrix_2->columns))
    {
        (void)fprintf(stderr,
                      "Incompatible matrices (%i, %i) + (%i, %i)\n",
                      p_matrix_1->rows,
                      p_matrix_1->columns,
                      p_matrix_2->rows,
                      p_matrix_2->columns);
        return NULL;
    }

    matrix_t * p_matrix_3 = calloc(1, sizeof(matrix_t));
    p_matrix_3->rows      = p_matrix_1->rows;
    p_matrix_3->columns   = p_matrix_1->columns;
    p_matrix_3->pp_data   = copy_2d_array(
        p_matrix_1->pp_data, p_matrix_1->rows, p_matrix_1->columns);

    for (int row = 0; row < p_matrix_3->rows; row++)
    {
        for (int column = 0; column < p_matrix_3->columns; column++)
        {
            // since p_matrix_1->data has been copied over, we can increment
            p_matrix_3->pp_data[row][column]
                += p_matrix_2->pp_data[row][column];
        }
    }

    return p_matrix_3;
}

void matrix_visualize (matrix_t * p_matrix)
{
    for (int row = 0; row < p_matrix->rows; row++)
    {
        for (int column = 0; column < p_matrix->columns; column++)
        {
            printf("%f\t", p_matrix->pp_data[row][column]);
        }
        printf("\n");
    }
    printf("\n");
}

matrix_t * matrix_copy (matrix_t * p_input)
{
    matrix_t * p_output = calloc(1, sizeof(matrix_t));
    p_output->rows = p_input->rows;
    p_output->columns = p_input->columns;
    p_output->pp_data = copy_2d_array(p_input->pp_data, p_output->rows, p_output->columns);

    return p_output;
}