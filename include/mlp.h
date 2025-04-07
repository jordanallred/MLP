#ifndef MLP_H
#define MLP_H

typedef enum
{
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX,
    LINEAR
} activation_t;

typedef struct
{
    int          input_size;
    int          output_size;
    double **    pp_weights;
    double *     p_biases;
    double *     p_pre_activation;
    double *     p_activations;
    double *     p_delta;
    activation_t activation_type;
} layer_t;

typedef struct
{
    int        num_layers;
    layer_t ** pp_layers;
    double     learning_rate;
    double     lambda;
    double *   p_input_data;
    double *   p_output_data;
} mlp_t;

// Function prototypes
mlp_t *  mlp_create (int            num_layers,
                     int *          p_layer_sizes,
                     activation_t * p_activation_types,
                     double         learning_rate,
                     double         lambda);
void     mlp_free (mlp_t * p_mlp);
void     mlp_train (mlp_t *   p_mlp,
                    double ** pp_inputs,
                    double ** pp_targets,
                    int       num_samples,
                    int       num_epochs,
                    int       batch_size);
double * mlp_predict (mlp_t * p_mlp, double * p_input);
void     mlp_save (mlp_t * p_mlp, const char * p_filename);
mlp_t *  mlp_load (const char * p_filename);

#endif // MLP_H
