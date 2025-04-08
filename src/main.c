#include "mlp.h"

#include <stdlib.h>

int main (void)
{

    int          num_layers         = 4;
    int          layer_sizes[]      = { 100, 64, 64, 10 };
    activation_t activation_types[] = { LINEAR, LINEAR, LINEAR };
    double       learning_rate      = 0.01;
    double       lambda             = 5.0;

    mlp_t * p_mlp = mlp_create(
        num_layers, layer_sizes, activation_types, learning_rate, lambda);

    matrix_t * p_input = matrix_create(1, 100);

    for (int index = 0; index < p_input->columns; index++)
    {
        p_input->pp_data[0][index] = index;
    }

    matrix_t * p_target = matrix_create(1, 10);

    for (int index = 0; index < p_target->columns; index++)
    {
        p_target->pp_data[0][index] = index;
    }

    matrix_t ** pp_input = calloc(1, sizeof(matrix_t *));
    pp_input[0]          = p_input;

    matrix_t ** pp_target = calloc(1, sizeof(matrix_t *));
    pp_target[0]          = p_target;

    mlp_train(p_mlp, pp_input, pp_target, 1, 100);

    mlp_free(p_mlp);

    matrix_free(p_input);
    matrix_free(p_target);
    free(pp_input);
    free(pp_target);

    return 0;
}