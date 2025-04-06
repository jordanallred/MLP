#include <mlp.h>

int main (void)
{

    int num_layers = 4;
    int layer_sizes[] = {100, 64, 64, 10};
    ActivationType activation_types[] = {SIGMOID, RELU};
    double learning_rate = 0.01;
    double lambda = 5.0;

    MLP * p_mlp = create_mlp(num_layers, layer_sizes, activation_types, learning_rate, lambda);
    free_mlp(p_mlp);

    return 0;
}