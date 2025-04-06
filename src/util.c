#include <stdlib.h>

double ** allocate_2d_array (int rows, int cols)
{
    double ** p_array = calloc(rows, sizeof(double *));

    for (int index = 0; index < rows; index++)
    {
        p_array[index] = calloc(cols, sizeof(double));
    }

    return p_array;
}
