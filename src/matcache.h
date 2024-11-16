#ifndef MATCACHE_H
#define MATCACHE_H

#include <string>
#include <tuple>

// Store a matrix in a binary file
template<class T>
int storeMatrix(T* matrix, int M, int K, int N, char operand, const std::string& schema);

// Loads matrices if they exist, else computes and stores them
template<class T>
std::tuple<T*, T*, T*> getMatrices(int M, int K, int N, const std::string& schema, struct LCG *rng);

#endif // MATCACHE_H
