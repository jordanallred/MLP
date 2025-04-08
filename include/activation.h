#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "matrix.h"

typedef enum
{
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX,
    LINEAR
} activation_t;

typedef matrix_t * (*activation_function_t)(matrix_t * p_input);

activation_function_t activation_get_function (activation_t activation);

#endif // ACTIVATION_H
