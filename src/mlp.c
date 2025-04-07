#include "mlp.h"
#include "util.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void   initialize_weights (mlp_t * p_mlp);
static void   forward_propagation (mlp_t * p_mlp, double * p_input);
static double compute_loss (mlp_t * p_mlp, double * p_target);
static void   backward_propagation (mlp_t * p_mlp, double * p_target);
static void   update_parameters (mlp_t * p_mlp);

mlp_t * mlp_create (int            num_layers,
                    int *          p_layer_sizes,
                    activation_t * p_activation_types,
                    double         learning_rate,
                    double         lambda)
{
    if (num_layers <= 0)
    {
        (void)fprintf(
            stderr, "Number of layers must be positive (was %i)\n", num_layers);
        return NULL;
    }

    if (NULL == p_layer_sizes)
    {
        (void)fprintf(stderr, "Layer sizes cannot be NULL\n");
        return NULL;
    }

    if (NULL == p_activation_types)
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

    mlp_t * p_mlp = calloc(1, sizeof(mlp_t));

    p_mlp->num_layers    = num_layers - 1;
    p_mlp->pp_layers     = calloc(num_layers, sizeof(layer_t *));
    p_mlp->learning_rate = learning_rate;
    p_mlp->lambda        = lambda;

    for (int index = 0; index < p_mlp->num_layers; index++)
    {
        p_mlp->pp_layers[index] = calloc(num_layers, sizeof(layer_t));
        p_mlp->pp_layers[index]->input_size      = p_layer_sizes[index];
        p_mlp->pp_layers[index]->output_size     = p_layer_sizes[index + 1];
        p_mlp->pp_layers[index]->activation_type = p_activation_types[index];

        p_mlp->pp_layers[index]->pp_weights
            = allocate_2d_array(p_mlp->pp_layers[index]->output_size,
                                p_mlp->pp_layers[index]->input_size);
        p_mlp->pp_layers[index]->p_biases
            = calloc(p_mlp->pp_layers[index]->output_size, sizeof(double));
        p_mlp->pp_layers[index]->p_pre_activation
            = calloc(p_mlp->pp_layers[index]->output_size, sizeof(double));
        p_mlp->pp_layers[index]->p_activations
            = calloc(p_mlp->pp_layers[index]->output_size, sizeof(double));
        p_mlp->pp_layers[index]->p_delta
            = calloc(p_mlp->pp_layers[index]->output_size, sizeof(double));
    }

    initialize_weights(p_mlp);

    p_mlp->p_input_data
        = calloc(p_mlp->pp_layers[0]->input_size, sizeof(double));
    p_mlp->p_output_data = calloc(
        p_mlp->pp_layers[p_mlp->num_layers - 1]->input_size, sizeof(double));

    return p_mlp;
}

void mlp_free (mlp_t * p_mlp)
{
    for (int index = 0; index < p_mlp->num_layers; index++)
    {
        layer_t * p_layer = p_mlp->pp_layers[index];

        for (int row = 0; row < p_layer->output_size; row++)
        {
            free(p_layer->pp_weights[row]);
        }
        free(p_layer->pp_weights);

        free(p_layer->p_biases);
        free(p_layer->p_pre_activation);
        free(p_layer->p_activations);
        free(p_layer->p_delta);
        free(p_layer);
    }

    free(p_mlp->pp_layers);
    free(p_mlp->p_input_data);
    free(p_mlp->p_output_data);
    free(p_mlp);
}

double * mlp_predict (mlp_t * p_mlp, double * p_input)
{
    memmove(p_mlp->p_input_data,
            p_input,
            p_mlp->pp_layers[0]->input_size * sizeof(double));

    // TODO: perform prediction and save to p_mlp->p_output_data

    return p_mlp->p_output_data;
}

static void initialize_weights (mlp_t * p_mlp)
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
        layer_t * p_layer = p_mlp->pp_layers[index];
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
            memmove(p_layer->pp_weights[index],
                    pp_weights[index * p_layer->input_size],
                    p_layer->input_size);
        }
    }

    (void)fclose(p_file);
}
