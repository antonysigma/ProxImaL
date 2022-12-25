#pragma once

#include <HalideBuffer.h>

namespace proximal {
namespace runtime {

using Halide::Runtime::Buffer;

struct signals_t {
    int error_code;
    Buffer<float> v_new;
    std::vector<float> r;
    std::vector<float> s;
    std::vector<float> eps_pri;
    std::vector<float> eps_dual;
};

/** Runtime function to call (L-)ADMM, with early termination.
 *
 * Halide being a non-Turing complete language, is unable to dynamically
 * terminate a while-loop. Therefore, we generate the Halide-optimized AOT
 * pipeline with a fixed (e.g. 10) iterations, returning the convergence
 * metrics at iteration #10.
 *
 * Then, we check the convergence criteria, and terminate the for-loop when the
 * criteria are met. Otherwise, repeat for another (10) iterations.
 *
 * Reference: https://stackoverflow.com/a/33472074
 */
signals_t ladmmSolver(Buffer<const float>& input, float shift_step, const size_t iter_max = 100);
}  // namespace runtime

}  // namespace proximal