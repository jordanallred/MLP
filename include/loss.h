//
// Created by jo on 4/6/25.
//

#ifndef LOSS_H
#define LOSS_H

double mse_loss (double * predictions, double * targets, int size);
double binary_cross_entropy (double * predictions, double * targets, int size);
double categorical_cross_entropy (double * predictions,
                                  double * targets,
                                  int      size);

#endif //LOSS_H
