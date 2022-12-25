#pragma once
#include <Halide.h>

#include <utility>

using namespace Halide;

#include "prior_transforms.h"
#include "problem-config.h"
#include "problem-interface.h"
#include "prox_operators.h"
#include "prox_parameterized.h"
#include "subsample-prox.h"
#include "user-linear-operator.h"

namespace problem_definition {

using proximal::prox::ParameterizedProx;

/** Transform between variables z and u.
 *
 * This is a demonstration of how ProxImaL should generate Halide code for the
 * (L-)ADMM solver. In this example, the end-user defines the TV-regularizaed
 * denoising problem with non-negative constraints in Proximal language. Then,
 * the ProxImaL-Gen module exports the following functions automatically.
 *
 * Currently (Sept, 2022), ProxImaL has not yet implemented Halide-codegen
 * feature in ProxImal. Here, we manualy craft the transform in Halide code as
 * an example.
 */
struct Transform {
    constexpr static auto width = problem_config::output_width;
    constexpr static auto height = problem_config::output_height;
    constexpr static auto N = problem_config::psi_size;

    constexpr static auto blur_size = problem_config::blur_size;
    const RDom blur_window{0, blur_size};

    /** Compute dx, dy of a two dimensional image with c number of channels. */
    FuncTuple<N> forward(const Func& u, const Func& shifts) {
        assert(u.dimensions() == 2);

        const Func image_gradient = K_grad_mat(u, width, height);

        const auto blurred = image_formation::boxBlur(u, width, height, blur_window, true, "blurred_x_fwd");
        Func scaled{"scaled"};
        scaled(x, y) = blurred(x, y) / (float(blur_size) * blur_size);

        const auto [shifted, _] = image_formation::A_warp(scaled, width, height, 1, shifts);

        /* Begin code-generation */
        return {image_gradient, shifted};
        /* End code-generation */
    }

    Func adjoint(const FuncTuple<N>& z, const Func& shifts) {
        using problem_config::input_layers;

        /* Begin code-generation */
        assert(z[0].dimensions() == 3);
        const Func image_gradient_inv = KT_grad_mat(z[0], width, height);

        assert(z[1].dimensions() == 3);
        const auto [shifted, _, __] = image_formation::At_warp(
            z[1], width, height, 1, shifts, input_layers);

        using problem_config::blur_size;
        const auto blurred = image_formation::boxBlur(shifted, width, height, blur_window, false, "blurred_x_adj");
        Func scaled{"scaled"};
        scaled(x, y) = blurred(x, y) / (float(blur_size) * blur_size * input_layers);

        Func aggregated;
        aggregated(x, y) = image_gradient_inv(x, y) + scaled(x, y);
        return aggregated;
        /* End code-generation */
    }
} K;

/** List of functions in set Omega, after problem splitting by ProxImaL. */
const ParameterizedProx omega_fn{
    /* Begin code-generation */
    /* . prox = */ nullptr
    /* .alpha = 1.0f, */
    /* .beta = 1.0f, */
    /* .gamma = 1.0f, */
    /* ._c = 0.0f, */
    /* .d = 0.0f, */
    /* .n_dim = 3, */
    /* End code-generation */
};

#if __cplusplus >= 202002L

// If C++20 is supported, validate the transform structure.
static_assert(Prox<ParameterizedProx>);
#endif

/** List of functions in set Psi, after problem splitting by ProxImaL.
 *
 * Note(Antony): these proximal functions can be parameterized with
 * proximal::prox::ParameterizedProx . But first the class needs to be generalized to 4D signal.
 *
 * TODO(Antony): Use C++20 reference designator to initialize fields. This requires Halide >= 14.0.
 */

using problem_config::alpha;
using problem_config::beta;

const std::array<ParameterizedProx, problem_config::psi_size> psi_fns{
    /* Begin code generation */
    ParameterizedProx{.prox = [](const Func& u, const Expr& theta) -> Func {
                          using problem_config::output_width;
                          using problem_config::output_height;

                          return proxIsoL1(u, output_width, output_height, theta);
                      },
                      .alpha = alpha * beta,
                      .beta = 1.0f,
                      .gamma = alpha * (1.0f - beta), 
                      ._c = 0.0f,
                      .d = 0.0f,
                      .n_dim = 3},
    ParameterizedProx{
        .prox_direct = [](const Func& u, const Expr& rho, const Func& b) -> Func {
            assert(u.dimensions() == 3);
            assert(b.dimensions() == 3);
            return prox::proxSubampleSumsq<problem_config::upsample>(u, rho, b);
        },
        .n_dim = 3,
        .need_b = true
    }
    /* End code-generation */
};

}  // namespace problem_definition
