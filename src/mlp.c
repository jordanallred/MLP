#include "mlp.h"
#include "util.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

MLP * create_mlp (int            num_layers,
                  int *          layer_sizes,
                  activation_t * activation_types,
                  double         learning_rate,
                  double         lambda)
{
    if (num_layers <= 0)
    {
        (void)fprintf(
            stderr, "Number of layers must be positive (was %i)\n", num_layers);
        return NULL;
    }

    if (NULL == layer_sizes)
    {
        (void)fprintf(stderr, "Layer sizes cannot be NULL\n");
        return NULL;
    }

    if (NULL == activation_types)
    {
        (void)fprintf(stderr, "Activation types cannot be NULL\n");
        return NULL;
    }

    if (learning_rate <= 0)
    {
        (void)fprintf(
            stderr, "Learning rate must be positive (was %f)\n", learning_rate);
        return NULL;
    }

    if (lambda <= 0)
    {
        (void)fprintf(stderr, "Lambda must be positive (was %f)\n", lambda);
        return NULL;
    }

    MLP * p_mlp = calloc(1, sizeof(MLP));

    p_mlp->num_layers    = num_layers - 1;
    p_mlp->layers        = calloc(num_layers, sizeof(layer_t *));
    p_mlp->learning_rate = learning_rate;
    p_mlp->lambda        = lambda;

    for (int index = 0; index < p_mlp->num_layers; index++)
    {
        p_mlp->layers[index]              = calloc(num_layers, sizeof(layer_t));
        p_mlp->layers[index]->input_size  = layer_sizes[index];
        p_mlp->layers[index]->output_size = layer_sizes[index + 1];
        p_mlp->layers[index]->activation_type = activation_types[index];

        p_mlp->layers[index]->weights
            = allocate_2d_array(p_mlp->layers[index]->output_size,
                                p_mlp->layers[index]->input_size);
        p_mlp->layers[index]->biases
            = calloc(p_mlp->layers[index]->output_size, sizeof(double));
        p_mlp->layers[index]->pre_activation
            = calloc(p_mlp->layers[index]->output_size, sizeof(double));
        p_mlp->layers[index]->activations
            = calloc(p_mlp->layers[index]->output_size, sizeof(double));
        p_mlp->layers[index]->delta
            = calloc(p_mlp->layers[index]->output_size, sizeof(double));
    }

    initialize_weights(p_mlp);

    return p_mlp;
}

void initialize_weights (MLP * p_mlp)
{
    errno = 0;

    FILE * p_file = fopen("/dev/urandom", "r");

    if (NULL == p_file)
    {
        perror("Failed to open /dev/urandom");
        return;
    }

    for (int index = 0; index < p_mlp->num_layers; index++)
    {
        layer_t * p_layer = p_mlp->layers[index];
        double ** pp_weights[p_layer->input_size][p_layer->output_size];
        size_t    num_read = fread(
            &pp_weights, p_layer->input_size, p_layer->output_size, p_file);

        if ((size_t)p_layer->output_size != num_read)
        {
            (void)fprintf(stderr, "fread() failed: %zu\n", num_read);
            break;
        }

        for (int row = 0; row < p_layer->output_size; row++)
        {
            memmove(p_layer->weights[index],
                    pp_weights[index * p_layer->input_size],
                    p_layer->input_size);
        }
    }

    (void)fclose(p_file);
}

void free_mlp (MLP * p_mlp)
{
    for (int index = 0; index < p_mlp->num_layers; index++)
    {
        layer_t * p_layer = p_mlp->layers[index];

        for (int row = 0; row < p_layer->output_size; row++)
        {
            free(p_layer->weights[row]);
        }
        free(p_layer->weights);

        free(p_layer->biases);
        free(p_layer->pre_activation);
        free(p_layer->activations);
        free(p_layer->delta);
        free(p_layer);
    }

    free(p_mlp->layers);
    free(p_mlp);
}
