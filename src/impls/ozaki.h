#pragma once

#include <cuda_fp16.h>
#include <vector>
#include <string>

// DEBUG
std::string fp64_binary_representation(const double d_);
void bitwise_comparison(const double expected, const double actual);

std::vector<std::vector<half>> ozaki_split_to_half(const size_t m, const size_t n, double* a, const int l);
std::vector<std::vector<float>> ozaki_split_to_float(const size_t m, const size_t n, double* a, const int l);
