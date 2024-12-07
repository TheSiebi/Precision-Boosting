#ifndef SPLIT_H
#define SPLIT_H

#include <stdio.h>
#include <stdlib.h>

// Ootomo Eq. 8 & 9
void split_v0(const double *A, void *A16, void *dA16, size_t M, size_t N); 
void splitf_v0(const float *A, void *A16, void *dA16, size_t M, size_t N); 

// Ootomo Eq. 19 & 20
void split_Ootomo_v0(const double *A, void *A16, void *dA16, size_t M, size_t N);
void splitf_Ootomo_v0(const float *A, void *A16, void *dA16, size_t M, size_t N);

#endif // SPLIT_H
