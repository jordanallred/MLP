#include "activation.h"

#include <stddef.h>
#include <stdlib.h>

static matrix_t * activation_sigmoid (matrix_t * p_input);
static matrix_t * activation_relu (matrix_t * p_input);
static matrix_t * activation_tanh (matrix_t * p_input);
static matrix_t * activation_softmax (matrix_t * p_input);
static matrix_t * activation_linear (matrix_t * p_input);

activation_function_t activation_get_function (activation_t activation)
{
    switch (activation)
    {
        case SIGMOID:
            return activation_sigmoid;
        case RELU:
            return activation_relu;
        case TANH:
            return activation_tanh;
        case SOFTMAX:
            return activation_softmax;
        case LINEAR:
            return activation_linear;
        default:
            return NULL;
    }
}

static matrix_t * activation_sigmoid (matrix_t * p_input)
{
    return NULL;
}

static matrix_t * activation_relu (matrix_t * p_input)
{
    return NULL;
}

static matrix_t * activation_tanh (matrix_t * p_input)
{
    return NULL;
}

static matrix_t * activation_softmax (matrix_t * p_input)
{
    return NULL;
}

static matrix_t * activation_linear (matrix_t * p_input)
{
    matrix_t * p_output = matrix_copy(p_input);

    return p_output;
}
