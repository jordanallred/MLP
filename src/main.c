#include <mlp.h>

int main (void)
{

    int          num_layers         = 4;
    int          layer_sizes[]      = { 100, 64, 64, 10 };
    activation_t activation_types[] = { SIGMOID, RELU };
    double       learning_rate      = 0.01;
    double       lambda             = 5.0;

    mlp_t * p_mlp = mlp_create(
        num_layers, layer_sizes, activation_types, learning_rate, lambda);
    mlp_free(p_mlp);

    return 0;
}