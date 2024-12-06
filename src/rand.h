#ifndef RAND_H
#define RAND_H

#include <algorithm>
#include <assert.h>
#include <bit>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <time.h>

float bits_to_float(uint32_t bits);
uint32_t float_to_bits(float f);
float construct_float(bool sign, int exponent, uint32_t mantissa);

double bits_to_double(uint64_t bits);
uint64_t double_to_bits(double d);
double construct_double(bool sign, int exponent, uint64_t mantissa);

// Linear congruential generator + functions
struct LCG {
    uint64_t state;
    uint64_t a;
    uint64_t b;
};

// Create a new LCG with a random seed
struct LCG new_rng();
// Create a new LCG with a starting seed value
struct LCG rng_seeded(uint64_t seed);

/// Get next random 32-bit value from the RNG
uint32_t next(struct LCG *rng);
/// Get next random 64-bit value from the RNG
uint64_t next_u64(struct LCG *rng);
uint32_t next_below(struct LCG *rng, uint32_t max);
int next_int(struct LCG *rng, int min, int max);
/// Get next random bool from the RNG, which is true with probability p
bool next_bool(struct LCG *rng, float p);

/// Return a 32-bit float value uniformly distributed between 0 and 1
float next_float(struct LCG *rng);
/// Return a 32-bit float value uniformly distributed between -1 and 1
float next_signed_float(struct LCG *rng);
/// Return a 32-bit float value uniformly distributed between 2^min_exp and 2^max_exp (positive or negative)
float next_signed_float_range(struct LCG *rng, int8_t min_exp, int8_t max_exp);
/// Corresponds to `exp_rand` in the Ootomo paper
float next_float_exp_rand(struct LCG *rng, int8_t min_exp, int8_t max_exp);
float next_float_with_exp(struct LCG *rng, int exponent);

/// Return a 64-bit double value uniformly distributed between 0 and 1
double next_double(struct LCG *rng);
/// Return a 64-bit double value uniformly distributed between -1 and 1
double next_signed_double(struct LCG *rng);
/// Return a 64-bit double value uniformly distributed between 2^min_exp and 2^max_exp (positive or negative)
double next_signed_double_range(struct LCG *rng, int16_t min_exp, int16_t max_exp);
/// Corresponds to `exp_rand` in the Ootomo paper
double next_double_exp_rand(struct LCG *rng, int16_t min_exp, int16_t max_exp);
double next_double_with_exp(struct LCG *rng, int exponent);

/// Generate an exponent between min_exp and max_exp following a geometric distribution
int next_exponent_geometric(struct LCG *rng, int min_exp, int max_exp);
int next_int_geometric(struct LCG *rng);

/// Fill array with floats distributed uniformly between -1 and 1
void gen_floats_urand(struct LCG *rng, float *fs, int size);
/// Fill array with doubles distributed uniformly between -1 and 1
void gen_doubles_urand(struct LCG *rng, double *ds, int size);
/**
 * Fill array with values distributed uniformly between -1 and 1
 *
 * @param  rng  the random number generator to use
 * @param  ts   the array to be filled
 * @param  size the number of entries to generate
 */
template<class T>
void gen_urand(struct LCG *rng, T *ts, int size)
{
    if (std::is_same<T, float>::value) {
        return gen_floats_urand(rng, (float*) ts, size);
    } else if (std::is_same<T, double>::value) {
        return gen_doubles_urand(rng, (double*) ts, size);
    } else {
        assert(false);
    }
}

/// Fill array with floats with uniformly distributed exponents between min_exp and max_exp
void gen_floats_exp_rand(struct LCG *rng, float *fs, int size, int min_exp, int max_exp);
/// Fill array with doubles with uniformly distributed exponents between min_exp and max_exp
void gen_doubles_exp_rand(struct LCG *rng, double *ds, int size, int min_exp, int max_exp);
/**
 * Fill array with values with uniformly distributed exponents between min_exp and max_exp
 *
 * @param  rng     the random number generator to use
 * @param  ts      the array to be filled
 * @param  size    the number of entries to generate
 * @param  min_exp the minimum exponent to generate (inclusive)
 * @param  max_exp the maximum exponent to generate (inclusive)
 */
template<class T>
void gen_exp_rand(struct LCG *rng, T *ts, int size, int min_exp, int max_exp)
{
    if (std::is_same<T, float>::value) {
        return gen_floats_exp_rand(rng, (float*) ts, size, min_exp, max_exp);
    } else if (std::is_same<T, double>::value) {
        return gen_doubles_exp_rand(rng, (double*) ts, size, min_exp, max_exp);
    } else {
        assert(false);
    }
}

void fill_matrices_ootomo_type1(struct LCG *rng, float *A, float *B, int size_a, int size_b);
void fill_matrices_ootomo_type2(struct LCG *rng, float *A, float *B, int size_a, int size_b);
void fill_matrices_ootomo_type3(struct LCG *rng, float *A, float *B, int size_a, int size_b);
void fill_matrices_ootomo_type4(struct LCG *rng, float *A, float *B, int size_a, int size_b);

template<class T>
void fill_matrices(struct LCG *rng, int input_type, T *A, T *B, int size_a, int size_b);

extern template void fill_matrices<float> (struct LCG *rng, int input_type, float *A, float *B, int size_a, int size_b);
extern template void fill_matrices<double>(struct LCG *rng, int input_type, double *A, double *B, int size_a, int size_b);

#endif // RAND_H
