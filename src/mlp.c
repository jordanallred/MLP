#include "mlp.h"
#include "matrix.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

static matrix_t * initialize_weights (layer_t * p_layer);

static matrix_t * forward_pass (mlp_t * p_mlp, matrix_t * p_input);
static void       backward_pass (mlp_t *    p_mlp,
                                 matrix_t * p_output,
                                 matrix_t * p_target);
static double     compute_loss (mlp_t *    p_mlp,
                                matrix_t * p_output,
                                matrix_t * p_target);
static void       back_propagation (mlp_t * p_mlp, double loss);

static print_progress_bar(int    current,
                          int    total,
                          int    bar_width,
                          char * p_message)
{
    float progress = (float)current / total;
    int   position = (int)(bar_width * progress);

    printf("\r[");
    for (int index = 0; index < bar_width; index++)
    {
        if (index < position)
        {
            printf("=");
        }
        else if (index == position)
        {
            printf(">");
        }
        else
        {
            printf(" ");
        }
    }
    printf("] %3d%% (%s)", (int)(progress * 100), p_message);
    fflush(stdout);
}

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
        free(p_layer);
    }

    free(p_mlp->pp_layers);
    free(p_mlp);
}

void mlp_train (mlp_t *     p_mlp,
                matrix_t ** pp_input,
                matrix_t ** pp_target,
                int         num_samples,
                int         num_epochs)
{
    for (int epoch = 0; epoch < num_epochs; epoch++)
    {

        for (int index = 0; index < num_samples; index++)
        {
            matrix_t * p_output = forward_pass(p_mlp, pp_input[index]);
            backward_pass(p_mlp, p_output, pp_target[index]);
            matrix_free(p_output);
        }
        print_progress_bar(epoch + 1, num_epochs, 100, "training");
    }
}

matrix_t * mlp_predict (mlp_t * p_mlp, matrix_t * p_input)
{
    matrix_t * p_prediction = forward_pass(p_mlp, p_input);
    return p_prediction;
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

static matrix_t * forward_pass (mlp_t * p_mlp, matrix_t * p_input)
{
    matrix_t * p_matrix_current = p_input;

    for (int index = 0; index < p_mlp->num_layers; index++)
    {
        layer_t * p_layer = p_mlp->pp_layers[index];

        matrix_t * p_matrix_1
            = matrix_multiply(p_matrix_current, p_layer->p_weights);
        p_layer->p_z = matrix_add(p_matrix_1, p_layer->p_biases);

        activation_function_t p_function
            = activation_get_function(p_layer->activation_type);

        matrix_t * p_matrix_3 = p_function(p_layer->p_z);

        matrix_free(p_matrix_1);

        if (p_matrix_current != p_input)
        {
            matrix_free(p_matrix_current);
        }

        p_matrix_current = p_matrix_3;
    }

    return p_matrix_current;
}

static void backward_pass (mlp_t *    p_mlp,
                           matrix_t * p_output,
                           matrix_t * p_target)
{
    double loss = compute_loss(p_mlp, p_output, p_target);

    // TODO: update weights for output layer from loss

    back_propagation(p_mlp, loss);
}

static double compute_loss (mlp_t *    p_mlp,
                            matrix_t * p_output,
                            matrix_t * p_target)
{
    return 0.0;
}

static void back_propagation (mlp_t * p_mlp)
{
    // TODO: implement backpropagation

    // dA = W_next^T · dZ_next
    // dZ = dA ⊙ activation'(Z)

    // dW = dZ · A_prev^T
    // db = dZ

    // W -= lr * dW
    // b -= lr * db
}