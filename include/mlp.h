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
    double **    weights;
    double *     biases;
    double *     pre_activation;
    double *     activations;
    double *     delta;
    activation_t activation_type;
} layer_t;

typedef struct
{
    int        num_layers;
    layer_t ** layers;
    double     learning_rate;
    double     lambda;
    double *   input_data;
    double *   output_data;
} MLP;

// Function prototypes
MLP *    create_mlp (int            num_layers,
                     int *          layer_sizes,
                     activation_t * activation_types,
                     double         learning_rate,
                     double         lambda);
void     free_mlp (MLP * p_mlp);
void     initialize_weights (MLP * p_mlp);
void     forward_propagation (MLP * p_mlp, double * input);
double   compute_loss (MLP * p_mlp, double * target);
void     backward_propagation (MLP * p_mlp, double * target);
void     update_parameters (MLP * p_mlp);
void     train_mlp (MLP *     mlp,
                    double ** inputs,
                    double ** targets,
                    int       num_samples,
                    int       num_epochs,
                    int       batch_size);
double * predict (MLP * p_mlp, double * input);
void     save_mlp (MLP * p_mlp, const char * filename);
MLP *    load_mlp (const char * filename);

#endif // MLP_H
