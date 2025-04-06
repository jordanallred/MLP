//
// Created by jo on 4/6/25.
//

#ifndef UTIL_H
#define UTIL_H

double ** allocate_2d_array (int rows, int cols);
void      free_2d_array (double ** array, int rows);
double    random_normal (double mean, double stddev);

#endif //UTIL_H
