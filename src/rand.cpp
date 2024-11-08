#include "rand.h"

float bits_to_float(uint32_t bits) {
    float *f_ptr = (float*) (&bits);
    return *f_ptr;
}
uint32_t float_to_bits(float f) {
    uint32_t *bits_ptr = (uint32_t*) (&f);
    return *bits_ptr;
}
double bits_to_double(uint64_t bits) {
    double *d_ptr = (double*) (&bits);
    return *d_ptr;
}
uint64_t double_to_bits(double d) {
    uint64_t *bits_ptr = (uint64_t*) (&d);
    return *bits_ptr;
}

struct LCG new_rng() {
    // Compute number of bits in RAND_MAX (rounded down)
    uint64_t rmax = RAND_MAX;
    int rmax_log2 = 0;
    while (rmax > 1) {
        rmax_log2++;
        rmax >>= 1;
    }
    // We need 64 bits total for a full seed, so how many rand calls?
    int num_calls = ((64 - 1) / rmax_log2) + 1;
    uint64_t accumulator = 0;
    for (int i = 0; i < num_calls; i++) {
        accumulator |= rand();
        accumulator <<= rmax_log2;
    }
    return rng_seeded(accumulator);
}
struct LCG rng_seeded(uint64_t seed) {
    struct LCG result;
    result.state = seed;
    // These values should have a period of 2^64
    result.a = 6364136223846793005ULL;
    result.b = 1442695040888963407ULL;
    return result;
}

// We only return the 32 most significant bits, because they have longer periods
uint32_t next(struct LCG *rng) {
    uint64_t result = rng->a * rng->state + rng->b;
    rng->state = result;
    return (uint32_t) (result >> 32);
}
uint64_t next_u64(struct LCG *rng) {
    uint64_t r1 = rng->a * rng->state + rng->b;
    uint64_t r2 = rng->a * r1 + rng->b;
    rng->state = r2;

    r1 = r1 >> 32;
    r2 = (r2 >> 32) << 32;
    return r1 | r2;
}

float next_float(struct LCG *rng) {
    // Generate a float uniformly distributed between 0 and 1
    return fabsf(next_signed_float(rng));
}
float next_signed_float(struct LCG *rng) {
    // Generate a float uniformly distributed between -1 and 1
    return next_float_exp_rand(rng, -128, -1);
}
float next_float_exp_rand(struct LCG *rng, int16_t min_exp, int16_t max_exp) {
    assert(max_exp >= min_exp);
    // Float: 1 sign | 8 exp | 23 mantissa
    int16_t exponent = next_exponent_geometric(rng, min_exp, max_exp);
    if (exponent <= -127) {
        // We could generate sub-normal numbers here, but it'll never happen anyway
        return 0.0;
    } else {
        uint32_t random_bits = next(rng);
        uint32_t exponent_bits = ((uint32_t) (exponent + 127)) << 23;
        uint32_t sign = (random_bits & 1) << 31;
        uint32_t mantissa = random_bits >> (32 - 23); // We just need 23 bits

        float result = bits_to_float(sign | exponent_bits | mantissa);
        // Sanity check
        assert(isnormal(result));
        return result;
    }
}

double next_double(struct LCG *rng) {
    // Generate a double uniformly distributed between 0 and 1
    return fabs(next_signed_double(rng));
}
double next_signed_double(struct LCG *rng) {
    // Generate a double uniformly distributed between -1 and 1
    return next_double_exp_rand(rng, -1024, -1);
}
double next_double_exp_rand(struct LCG *rng, int16_t min_exp, int16_t max_exp) {
    assert(max_exp >= min_exp);
    // Double: 1 sign | 11 exp | 52 mantissa
    int16_t exponent = next_exponent_geometric(rng, min_exp, max_exp);
    if (exponent <= -1023) {
        // We could generate sub-normal numbers here, but it'll never happen anyway
        return 0.0;
    } else {
        uint64_t random_bits = next_u64(rng);
        uint64_t exponent_bits = ((uint64_t) (exponent + 1023)) << 52;
        uint64_t sign = (random_bits & 1) << 63;
        uint64_t mantissa = random_bits >> (64 - 52); // We just need 52 bits
        float result = bits_to_double(sign | exponent_bits | mantissa);
        // Sanity check
        assert(isnormal(result));
        return result;
    }
}

// Generates a geometric distribution for a floating-point exponent value
int16_t next_exponent_geometric(struct LCG *rng, int16_t min_exp, int16_t max_exp) {
    assert(max_exp >= min_exp);
    
    // Create a geometric distribution for the exponent
    int16_t exp = 0;
    uint32_t bits = 0;
    while (exp < max_exp - min_exp) {
        if (exp % 32 == 0) {
            // Generate 32 random bits
            bits = next(rng);
        } else {
            // Advance to next bit
            bits = bits >> 1;
        }
        // 50% odds to exit, 50% odds of next smaller exponent
        if (bits & 1) {
            break;
        }
        exp++;
    }
    return max_exp - exp;
}

void gen_floats_urand(struct LCG *rng, float *fs, int size) {
    for (int i = 0; i < size; i++) {
        fs[i] = next_signed_float(rng);
    }
}
void gen_doubles_urand(struct LCG *rng, double *fs, int size) {
    for (int i = 0; i < size; i++) {
        fs[i] = next_signed_double(rng);
    }
}