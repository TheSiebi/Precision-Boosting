#pragma once

#include <cuda_fp16.h>
#include <vector>

std::vector<std::vector<half>> ozaki_split_to_half(const size_t m, const size_t n, double* a, const int l);
std::vector<std::vector<float>> ozaki_split_to_float(const size_t m, const size_t n, double* a, const int l);
