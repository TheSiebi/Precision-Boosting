#include "rand.h"

float bits_to_float(uint32_t bits) {
    float *f_ptr = (float*) (&bits);
    return *f_ptr;
}
uint32_t float_to_bits(float f) {
    uint32_t *bits_ptr = (uint32_t*) (&f);
    return *bits_ptr;
}
float construct_float(bool sign, int exponent, uint32_t mantissa) {
    // Float: 1 sign | 8 exp | 23 mantissa
    assert(-127 <= exponent && exponent <= 128);
    assert(mantissa <= (1ull << 23));
    uint32_t sign_bit = ((uint32_t) sign) << 31;
    uint32_t exp_bits = ((uint32_t) (exponent + 127)) << 23;
    uint64_t res = sign_bit | exp_bits | mantissa;
    float* res_ptr = (float*) &res;
    return *res_ptr;
}
double bits_to_double(uint64_t bits) {
    double *d_ptr = (double*) (&bits);
    return *d_ptr;
}
uint64_t double_to_bits(double d) {
    uint64_t *bits_ptr = (uint64_t*) (&d);
    return *bits_ptr;
}
double construct_double(bool sign, int exponent, uint64_t mantissa) {
    // Double: 1 sign | 11 exp | 52 mantissa
    assert(-1023 <= exponent && exponent <= 1024);
    assert(mantissa <= (1ull << 52));
    uint64_t sign_bit = ((uint64_t) (sign ? 1 : 0)) << 63;
    uint64_t exp_bits = ((uint64_t) (exponent + 1023)) << 52;
    uint64_t res = sign_bit | exp_bits | mantissa;
    double* res_ptr = (double*) &res;
    return *res_ptr;
}

struct LCG new_rng() {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC_RAW, &time);
    // Randomize seed using nanoseconds of current time
    return rng_seeded((uint64_t) time.tv_nsec);
}
struct LCG rng_seeded(uint64_t seed) {
    struct LCG result;
    result.state = seed;
    // These values should have a period of 2^64
    result.a = 6364136223846793005ULL;
    result.b = 1442695040888963407ULL;
    return result;
}

uint32_t next(struct LCG *rng) {
    // We only use the 32 most significant bits, because they have longer periods
    uint64_t result = rng->a * rng->state + rng->b;
    rng->state = result;
    return (uint32_t) (result >> 32);
}
uint64_t next_u64(struct LCG *rng) {
    uint64_t r1 = (uint64_t) next(rng);
    uint64_t r2 = ((uint64_t) next(rng)) << 32;
    return r1 | r2;
}
uint32_t next_below(struct LCG *rng, uint32_t max) {
    uint64_t limit = UINT32_MAX + 1;
    uint64_t rand = next(rng);
    uint64_t mod = limit % max;
    while (rand < mod) {
        // Debias the result
        limit = mod << 32;
        rand = (rand << 32) + next(rng);
        mod = limit % max;
    }
    return (rand - mod) % max;
}
int next_int(struct LCG *rng, int min, int max) {
    uint32_t delta = (uint32_t) (max - min);
    uint32_t rand = next_below(rng, delta);
    int32_t result = min + (int32_t) rand;
    return (int) result;
}

float next_float(struct LCG *rng) {
    // Generate a float uniformly distributed between 0 and 1
    return fabsf(next_signed_float(rng));
}
float next_signed_float(struct LCG *rng) {
    // Generate a float uniformly distributed between -1 and 1
    return next_signed_float_range(rng, -128, -1);
}
float next_signed_float_range(struct LCG *rng, int8_t min_exp, int8_t max_exp) {
    // It's fine for min_exp to be small (we can generate sub-normal values)
    // But max_exp above 1023 will generate non-uniform values
    assert(min_exp <= max_exp && max_exp <= 127);
    // Geometrically distributed exponent -> uniformly distributed values
    int exponent = next_exponent_geometric(rng, min_exp, max_exp);
    return next_float_with_exp(rng, exponent);
}
float next_float_exp_rand(struct LCG *rng, int8_t min_exp, int8_t max_exp) {
    // It's fine for min_exp to be small (we can generate sub-normal values)
    // But max_exp above 127 will generate non-uniform values
    assert(min_exp <= max_exp && max_exp <= 127);
    // Uniformly random exponent
    int exponent = next_int(rng, (int) min_exp, (int) max_exp);
    return next_float_with_exp(rng, exponent);
}
float next_float_with_exp(struct LCG *rng, int exponent) {
    // Don't allow inf / NaN values (exp=128)
    exponent = std::clamp(exponent, -127, 127);

    uint32_t random_bits = next(rng);
    bool sign = (bool) (random_bits & 1);
    uint32_t mantissa = random_bits >> (32 - 23); // We just need 23 bits

    float result = construct_float(sign, exponent, mantissa);
    // Sanity check
    assert(isnormal(result));
    return result;
}

double next_double(struct LCG *rng) {
    // Generate a double uniformly distributed between 0 and 1
    return fabs(next_signed_double(rng));
}
double next_signed_double(struct LCG *rng) {
    // Generate a double uniformly distributed between -1 and 1
    return next_signed_double_range(rng, -1024, -1);
}
double next_signed_double_range(struct LCG *rng, int16_t min_exp, int16_t max_exp) {
    // It's fine for min_exp to be small (we can generate sub-normal values)
    // But max_exp above 1023 will generate non-uniform values
    assert(min_exp <= max_exp && max_exp <= 1023);
    // Geometrically distributed exponent -> uniformly distributed values
    int exponent = next_exponent_geometric(rng, min_exp, max_exp);
    return next_double_with_exp(rng, exponent);
}
double next_double_exp_rand(struct LCG *rng, int16_t min_exp, int16_t max_exp) {
    // It's fine for min_exp to be small (we can generate sub-normal values)
    // But max_exp above 1023 will generate non-uniform values
    assert(min_exp <= max_exp && max_exp <= 1023);
    // Uniformly random exponent
    int exponent = next_int(rng, (int) min_exp, (int) max_exp);
    return next_double_with_exp(rng, exponent);
}
double next_double_with_exp(struct LCG *rng, int exponent) {
    // Don't allow inf / NaN values (exp=1024)
    exponent = std::clamp(exponent, -1023, 1023);

    uint64_t random_bits = next_u64(rng);
    bool sign = (bool) (random_bits & 1);
    uint64_t mantissa = random_bits >> (64 - 52); // We just need 52 bits

    double result = construct_double(sign, exponent, mantissa);
    // Sanity check
    assert(isnormal(result));
    return result;
}

// Generates a geometric distribution for a floating-point exponent value
int next_exponent_geometric(struct LCG *rng, int min_exp, int max_exp) {
    // Early exit
    if (max_exp <= min_exp) {
        assert(max_exp >= min_exp);
        return min_exp;
    }
    
    // Create a geometric distribution for the exponent
    int exp = next_int_geometric(rng);
    if (exp > max_exp - min_exp) {
        // Debias the result in this case
        return next_exponent_geometric(rng, min_exp, max_exp);
    } else {
        return max_exp - exp;
    }
}
int next_int_geometric(struct LCG *rng) {
    int result = 0;
    uint32_t random_bits = next(rng);
    while (random_bits == 0) {
        // Shortcut all zeroes
        random_bits = next(rng);
        result += 32;
    }
    while ((random_bits & 1) == 0) {
        random_bits >>= 1;
        result++;
    }
    return result;
}

void gen_floats_urand(struct LCG *rng, float *fs, int size) {
    for (int i = 0; i < size; i++) {
        fs[i] = next_signed_float(rng);
    }
}
void gen_doubles_urand(struct LCG *rng, double *ds, int size) {
    for (int i = 0; i < size; i++) {
        ds[i] = next_signed_double(rng);
    }
}

void gen_floats_exp_rand(struct LCG *rng, float *fs, int size, int min_exp, int max_exp) {
    assert(-128 <= min_exp && 127 <= max_exp);
    for (int i = 0; i < size; i++) {
        fs[i] = next_float_exp_rand(rng, (int8_t) min_exp, (int8_t) max_exp);
    }
}
void gen_doubles_exp_rand(struct LCG *rng, double *ds, int size, int min_exp, int max_exp) {
    assert(-1024 <= min_exp && 1023 <= max_exp);
    for (int i = 0; i < size; i++) {
        ds[i] = next_double_exp_rand(rng, (int16_t) min_exp, (int16_t) max_exp);
    }
}