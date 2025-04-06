#ifndef ACTIVATION_H
#define ACTIVATION_H

double sigmoid (double x);
double sigmoid_derivative (double x);
double relu (double x);
double relu_derivative (double x);
double tanh_activation (double x);
double tanh_derivative (double x);
void   softmax (double * input, double * output, int size);
void   softmax_derivative (double * output,
                           double * target,
                           double * delta,
                           int      size);

#endif //ACTIVATION_H
