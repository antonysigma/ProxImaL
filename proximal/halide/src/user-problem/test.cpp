#include <HalideBuffer.h>

#include <iostream>
#include <chrono>

#include <armadillo>
#include "halide_image_io.h"
#include "ladmm-runtime.h"
#include "problem-config.h"
#include <highfive/H5File.hpp>

using Halide::Runtime::Buffer;
using Halide::Tools::load_and_convert_image;
using proximal::runtime::ladmmSolver;

namespace {

constexpr auto W = problem_config::input_width;
constexpr auto H = problem_config::input_height;
constexpr auto M = problem_config::input_layers;

constexpr auto W2 = problem_config::output_width;
constexpr auto H2 = problem_config::output_height;

#ifndef RAW_IMAGE_PATH
#error Path to the raw image must be defined with -DRAW_IMAGE_PATH="..." in the compile command.
#endif

constexpr char raw_image_path[]{RAW_IMAGE_PATH};

constexpr bool verbose = true;

void
saveFirstSlice(const Buffer<float>& slices) {
    using namespace arma;
    Mat<float> copied(slices.begin(), W, H, true);
    copied /= max(max(copied));

    Buffer<float> first_slice(copied.memptr(), W, H);
    Halide::Tools::convert_and_save_image(first_slice, "raw.png");
}

Buffer<const float>
readDataset(size_t left, size_t top) {
    using namespace HighFive;

    File file{RAW_IMAGE_PATH, File::ReadOnly};
    auto dataset = file.getDataSet("background_suppressed");

    Buffer<float> buffer(W, H, M);
    dataset.select({0, top, left}, {M, H, W}).read(buffer.begin());

    //saveFirstSlice(buffer);

    return buffer;
}

void
normalize(Buffer<float>& image) {
    using namespace arma;
    Mat<float> mapped(image.begin(), W2, H2, false, true);
    const float vmin = min(min(mapped)); 
    const float vmax = max(max(mapped)); 
    mapped = (mapped - vmin) / (vmax - vmin);

    std::cout << "Vmax = " << vmax << '\n';
}

}  // namespace

int
main() {
    using namespace arma;
    auto raw = readDataset(2592/2, 1944/2);
    std::cout << "Top-left pixel (raw) = " << raw(0, 0, 0) << '\n';
    constexpr auto shift_step = 0.175f;

    constexpr auto max_n_iter = 6;

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;

    auto start{high_resolution_clock::now()};
    const auto [error_code, denoised, r, s, eps_pri, eps_dual] =
        ladmmSolver(raw, shift_step, max_n_iter);
    auto diff=duration_cast<milliseconds>(high_resolution_clock::now()-start);
    std::cout << "Time elapsed: " << diff.count() << "ms\n";

    // TODO(Antony): use std::ranges::zip_view
    for (size_t i = 0; i < r.size(); i++) {
        const bool converged = (r[i] < eps_pri[i]) && (s[i] < eps_dual[i]);

        std::cout << "{r, eps_pri, s, eps_dual}[" << i << "] = " << r[i] << '\t' << eps_pri[i]
                  << '\t' << s[i] << '\t' << eps_dual[i] << (converged ? "\tconverged" : "")
                  << '\n';
    }

    std::cout << "Top-left pixel = " << denoised(0, 0) << '\n';

    Buffer<float> output = std::move(denoised);
    normalize(output);

    Halide::Tools::convert_and_save_image(output, "denoised.png");

    return 0;
}
