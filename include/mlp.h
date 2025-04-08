#ifndef MLP_H
#define MLP_H

#include "activation.h"
#include "matrix.h"

typedef struct
{
    int          input_size;
    int          output_size;
    matrix_t *   p_weights;
    matrix_t *   p_biases;
    matrix_t *   p_delta;
    activation_t activation_type;
} layer_t;

typedef struct
{
    int        num_layers;
    layer_t ** pp_layers;
    double     learning_rate;
    double     lambda;
} mlp_t;

// Function prototypes
mlp_t *    mlp_create (int            num_layers,
                       int *          p_layer_sizes,
                       activation_t * p_activation_types,
                       double         learning_rate,
                       double         lambda);
void       mlp_free (mlp_t * p_mlp);
void       mlp_train (mlp_t *   p_mlp,
                      double ** pp_inputs,
                      double ** pp_targets,
                      int       num_samples,
                      int       num_epochs,
                      int       batch_size);
matrix_t * mlp_predict (mlp_t * p_mlp, matrix_t * p_input);
void       mlp_save (mlp_t * p_mlp, const char * p_filename);
mlp_t *    mlp_load (const char * p_filename);

#endif // MLP_H
