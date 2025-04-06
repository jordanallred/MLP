#ifndef MLP_H
#define MLP_H

// Activation function types
typedef enum
{
    SIGMOID,
    RELU,
    TANH,
    SOFTMAX,
    LINEAR
} ActivationType;

// Layer structure
typedef struct
{
    int            input_size;  // Number of inputs to this layer
    int            output_size; // Number of neurons in this layer
    double **      weights;     // Weight matrix [output_size][input_size]
    double *       biases;      // Bias vector [output_size]
    double *       z;           // Pre-activation values [output_size]
    double *       activations; // Activation values [output_size]
    double *       delta;       // Error terms for backpropagation [output_size]
    ActivationType activation_type; // Type of activation function
} Layer;

// MLP Network structure
typedef struct
{
    int      num_layers; // Total number of layers (including input and output)
    Layer ** layers;     // Array of layer pointers
    double   learning_rate; // Learning rate for gradient descent
    double   lambda;        // L2 regularization parameter
    double * input_data;    // Pointer to current input data
    double * output_data;   // Pointer to current output data
} MLP;

// Function prototypes
MLP *    create_mlp (int              num_layers,
                     int *            layer_sizes,
                     ActivationType * activation_types,
                     double           learning_rate,
                     double           lambda);
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
