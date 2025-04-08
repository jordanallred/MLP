#include "mlp.h"
#include "matrix.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static matrix_t * initialize_weights (layer_t * p_layer);
static void       forward_propagation (mlp_t * p_mlp, double * p_input);
static double     compute_loss (mlp_t * p_mlp, double * p_target);
static void       backward_propagation (mlp_t * p_mlp, double * p_target);
static void       update_parameters (mlp_t * p_mlp);

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

    p_mlp->num_layers = num_layers - 1; // input layer is only a layer in name
    p_mlp->pp_layers  = calloc(num_layers, sizeof(layer_t *));
    p_mlp->learning_rate = learning_rate;
    p_mlp->lambda        = lambda;

    for (int index = 0; index < p_mlp->num_layers; index++)
    {
        p_mlp->pp_layers[index] = calloc(num_layers, sizeof(layer_t));
        p_mlp->pp_layers[index]->input_size  = p_layer_sizes[index];
        p_mlp->pp_layers[index]->output_size = p_layer_sizes[index + 1];
        p_mlp->pp_layers[index]->p_weights
            = initialize_weights(p_mlp->pp_layers[index]);
        p_mlp->pp_layers[index]->p_biases
            = matrix_create(1, p_mlp->pp_layers[index]->output_size);
        p_mlp->pp_layers[index]->p_delta
            = matrix_create(1, p_mlp->pp_layers[index]->output_size);
        p_mlp->pp_layers[index]->activation_type = p_activation_types[index];
    }

    return p_mlp;
}

void mlp_free (mlp_t * p_mlp)
{
    for (int index = 0; index < p_mlp->num_layers; index++)
    {
        layer_t * p_layer = p_mlp->pp_layers[index];

        matrix_free(p_layer->p_weights);
        matrix_free(p_layer->p_biases);
        matrix_free(p_layer->p_delta);
        free(p_layer);
    }

    free(p_mlp->pp_layers);
    free(p_mlp);
}

matrix_t * mlp_predict (mlp_t * p_mlp, matrix_t * p_input)
{
    matrix_t * p_matrix_current = p_input;

    for (int index = 0; index < p_mlp->num_layers; index++)
    {
        layer_t * p_layer = p_mlp->pp_layers[index];

        matrix_t * p_matrix_1
            = matrix_multiply(p_matrix_current, p_layer->p_weights);
        matrix_t * p_matrix_2 = matrix_add(p_matrix_1, p_layer->p_biases);

        activation_function_t p_function
            = activation_get_function(p_layer->activation_type);

        matrix_t * p_matrix_3 = p_function(p_matrix_2);

        matrix_free(p_matrix_1);
        matrix_free(p_matrix_2);
        matrix_free(p_matrix_current);

        p_matrix_current = p_matrix_3;
    }

    return p_matrix_current;
}

static matrix_t * initialize_weights (layer_t * p_layer)
{
    errno = 0;

    FILE * p_file = fopen("/dev/urandom", "r");

    if (NULL == p_file)
    {
        perror("Failed to open /dev/urandom");
        return NULL;
    }

    double pp_weights[p_layer->input_size][p_layer->output_size];
    size_t num_read
        = fread(&pp_weights, p_layer->input_size, p_layer->output_size, p_file);

    if ((size_t)p_layer->output_size != num_read)
    {
        (void)fprintf(stderr, "fread() failed: %zu\n", num_read);
        return NULL;
    }

    matrix_t * p_matrix
        = matrix_create(p_layer->input_size, p_layer->output_size);

    for (int row = 0; row < p_matrix->rows; row++)
    {
        for (int column = 0; column < p_matrix->columns; column++)
        {
            p_matrix->pp_data[row][column] = pp_weights[row][column];
        }
    }

    (void)fclose(p_file);

    return p_matrix;
}
