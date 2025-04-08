#include "mlp.h"

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

    matrix_t * p_output = mlp_predict(p_mlp, p_input);

    matrix_visualize(p_output);

    matrix_free(p_output);

    mlp_free(p_mlp);

    return 0;
}