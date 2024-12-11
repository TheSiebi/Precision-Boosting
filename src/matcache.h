#ifndef MATCACHE_H
#define MATCACHE_H

#include <string>
#include <tuple>

// Store a matrix in a binary file
template<class T>
int storeMatrix(T* matrix, size_t M, size_t K, size_t N, char operand, const std::string& schema, int index);

// Loads matrices if they exist, else computes and stores them
template<class T>
std::tuple<T*, T*, T*> getMatrices(size_t M, size_t K, size_t N, int input_type, int index, struct LCG *rng);

#endif // MATCACHE_H
