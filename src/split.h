#ifndef SPLIT_H
#define SPLIT_H

#include <stdio.h>
#include <stdlib.h>

void split_v0(const double *A, void *A16, void *dA16, int M, int N); 
void split_Ootomo_v0(const double *A, void *A16, void *dA16, int M, int N);

#endif // SPLIT_H
