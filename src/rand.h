#ifndef RAND_H
#define RAND_H

#include <assert.h>
#include <bit>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <time.h>

float bits_to_float(uint32_t bits);
uint32_t float_to_bits(float f);
double bits_to_double(uint64_t bits);
uint64_t double_to_bits(double d);

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

/// Return a 32-bit float value uniformly distributed between 0 and 1
float next_float(struct LCG *rng);
/// Return a 32-bit float value uniformly distributed between -1 and 1
float next_signed_float(struct LCG *rng);
/// Corresponds to `exp_rand` in the Ootomo paper
float next_float_exp_rand(struct LCG *rng, int16_t min_exp, int16_t max_exp);

/// Return a 64-bit double value uniformly distributed between 0 and 1
double next_double(struct LCG *rng);
/// Return a 64-bit double value uniformly distributed between -1 and 1
double next_signed_double(struct LCG *rng);
/// Corresponds to `exp_rand` in the Ootomo paper
double next_double_exp_rand(struct LCG *rng, int16_t min_exp, int16_t max_exp);

/// Generate an exponent between min_exp and max_exp following a geometric distribution
int16_t next_exponent_geometric(struct LCG *rng, int16_t min_exp, int16_t max_exp);

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

#endif // RAND_H
