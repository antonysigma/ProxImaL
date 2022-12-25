#include "ladmm-runtime.h"

#include <HalideBuffer.h>

#include "ladmm_iter.h"
#include "problem-config.h"

using Halide::Runtime::Buffer;

namespace proximal {
namespace runtime {

constexpr auto W = problem_config::input_width;
constexpr auto H = problem_config::input_height;
constexpr auto M = problem_config::input_layers;

constexpr auto W2 = problem_config::output_width;
constexpr auto H2 = problem_config::output_height;

signals_t
ladmmSolver(Buffer<const float>& input, float shift_step, const size_t iter_max) {
    Buffer<float> v(W2, H2);
    Buffer<float> z0(W2, H2, 2);
    Buffer<float> u0(W2, H2, 2);

    Buffer<float> z1(W2, H2, M);
    Buffer<float> u1(W2, H2, M);

    // Set zeros
    for (auto* buf : {&v, &z0, &z1, &u0, &u1}) {
        buf->fill(0.0f);
        buf->set_host_dirty();
    }

    input.set_host_dirty();

    Buffer<float> z0_new(W2, H2, 2);
    Buffer<float> u0_new(W2, H2, 2);

    Buffer<float> z1_new(W2, H2, M);
    Buffer<float> u1_new(W2, H2, M);

    Buffer<float> v_new(W2, H2);

    std::vector<float> r(iter_max);
    std::vector<float> s(iter_max);
    std::vector<float> eps_pri(iter_max);
    std::vector<float> eps_dual(iter_max);

    for (size_t i = 0; i < iter_max; i++) {
        auto _r = Buffer<float>::make_scalar(r.data() + i);
        auto _s = Buffer<float>::make_scalar(s.data() + i);
        auto _eps_pri = Buffer<float>::make_scalar(eps_pri.data() + i);
        auto _eps_dual = Buffer<float>::make_scalar(eps_dual.data() + i);

        const auto error = ladmm_iter(input, shift_step, v, z0, z1, u0, u1, v_new, z0_new, z1_new, u0_new,
                                      u1_new, _r, _s, _eps_pri, _eps_dual);

        if (error) {
            return {error, {}, {}, {}, {}, {}};
        }

        // Terminate the algorithm early, if optimal solution is reached.
        _r.copy_to_host();
        _s.copy_to_host();
        _eps_pri.copy_to_host();
        _eps_dual.copy_to_host();
        
        const bool converged = (r[i] < eps_pri[i]) && (s[i] < eps_dual[i]);
        if (converged) {
            for (auto* v : {&r, &s, &eps_pri, &eps_dual}) {
                v->resize(i + 1);
            }
            break;
        }

        if (i != iter_max - 1) {
            // This iteration's v_new becomes current v in the next iteration.
            std::swap(v, v_new);
            std::swap(u0, u0_new);
            std::swap(u1, u1_new);
            std::swap(z0, z0_new);
            std::swap(z1, z1_new);
        }
    }

    constexpr int success = 0;
    v_new.copy_to_host();
    return {success, v_new, r, s, eps_pri, eps_dual};
}

}  // namespace runtime

}  // namespace proximal
