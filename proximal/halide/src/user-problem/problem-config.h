#pragma once
#include <cstdint>

/** User-defined problem configurations. This is supposed to be generated automatically by
 * ProxImal-Gen */
namespace problem_config {

/** Number of functions in the set "psi", for (L-)ADMM solvers. */
constexpr auto psi_size = 2;

constexpr auto upsample = 8;
constexpr auto blur_size = 7;

/** input data size of the user-provided (distorted and noisy) image. */
constexpr int32_t input_width = 128;
constexpr auto input_height = input_width;
constexpr auto input_layers = 8 * 8;
constexpr auto input_size = size_t(input_width) * input_height * input_layers;

/** output data size of the restored image. */
constexpr auto output_width = input_width * upsample;
constexpr auto output_height = output_width;
constexpr auto output_size = size_t(output_width) * output_height;

constexpr float alpha = 0.7e-4f;
constexpr float beta = 0.7f;
}  // namespace problem_config
