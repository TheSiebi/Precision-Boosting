#ifndef SPLIT_H
#define SPLIT_H

#include <stdio.h>
#include <stdlib.h>

// Ootomo Eq. 8 & 9
void split_v0(const double *A, void *A16, void *dA16, int M, int N); 
void splitf_v0(const float *A, void *A16, void *dA16, int M, int N); 

// Ootomo Eq. 19 & 20
void split_Ootomo_v0(const double *A, void *A16, void *dA16, int M, int N);
void splitf_Ootomo_v0(const float *A, void *A16, void *dA16, int M, int N);

#endif // SPLIT_H
