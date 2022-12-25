#include "user-linear-operator.h"

using namespace Halide;

namespace {
Var x{"x"}, y{"y"}, k{"k"};

Var g{"g"}, l{"l"}; //<! bi-linear interp2 axes

Var i{"i"}; //<! Axis label

enum axis_t {
    X = 0,
    Y = 1
};

/** Interpolation kernel (linear) */
Expr
kernel_linear(Expr x) {
    Expr xx = abs(x);
    return select(xx < 1.0f, 1.0f - xx, 0.0f);
}

}

namespace image_formation {

std::pair<Func, Func>
A_warp(const Func& input, const Expr width, const Expr height, const Expr upsample,
       const Func& shift) {
    // Local vars
    assert(input.dimensions() == 2);

    Func input_f32;
    input_f32(x, y) = cast<float>(input(x, y));

    // Repeatable boundary
    Func clamped{"clampedInput"};
    clamped(x, y) = BoundaryConditions::constant_exterior(input_f32, 0.0f, {{0, width}, {0, height}})(x, y);

    // coords from warp matrix
    const Expr sourcex = (x + shift(X, k)) * upsample;
    const Expr sourcey = (y + shift(Y, k)) * upsample;

    // Initialize interpolation kernels.
    Func kernel{"kernel"};
    Expr beginx = cast<int>(sourcex);
    Expr beginy = cast<int>(sourcey);
    RDom dom(0, 2, 0, 2, "dom");
    kernel(x, y, k, g, l) =
        kernel_linear(g + beginx - sourcex) * kernel_linear(l + beginy - sourcey);

    // Perform resampling
    Func resampled("resampled");

    constexpr bool clamp_input = true;
    if constexpr (clamp_input) {
        // Auto-scheduler Li2018 for GPU requires a doubple clamped input.
        // Delete this code when we manually implement the GPU code.
        resampled(x, y, k) +=
            kernel(x, y, k, dom.x, dom.y) *
            clamped(clamp(dom.x + beginx, 0, width - 1), clamp(dom.y + beginy, 0, height - 1));
    } else {
        resampled(x, y, k) +=
            kernel(x, y, k, dom.x, dom.y) * clamped(dom.x + beginx, dom.y + beginy);
    }

    return {resampled, clamped};
}

std::tuple<Func, Func, Func>
At_warp(const Func& input, const Expr width, const Expr height, const Expr upsample,
       const Func& shift, const Expr n_shifts) {
    assert(input.dimensions() == 3);

    // Repeatable boundary
    Func clamped{"clampedInput"};
    clamped(x, y, k) = BoundaryConditions::constant_exterior(input, 0.0f, {{0, width}, {0, height}, {0, n_shifts}})(x, y, k);

    // Given the destination coordinate in the high-resolution frame, find the
    // corresponding coordinate in the low-resolution frame.
    const Expr sourcex = x / upsample - shift(X, k);
    const Expr sourcey = y / upsample - shift(Y, k);

    // Initialize interpolation kernels.
    Func kernel{"kernel"};
    Expr beginx = cast<int>(sourcex);
    Expr beginy = cast<int>(sourcey);
    RDom dom(0, 2, 0, 2, "dom");
    kernel(x, y, k, g, l) =
        kernel_linear(g + beginx - sourcex) * kernel_linear(l + beginy - sourcey);

    // Perform resampling
    Func resampled_at("resampled_at");

    constexpr bool clamp_input = true;
    if constexpr (clamp_input) {
        // Auto-scheduler Li2018 for GPU requires a doubple clamped input.
        // Delete this code when we manually implement the GPU code.
        resampled_at(x, y, k) +=
            kernel(x, y, k, dom.x, dom.y) *
            clamped(clamp(dom.x + beginx, 0, width - 1), clamp(dom.y + beginy, 0, height - 1), k);
    } else {
        resampled_at(x, y, k) +=
            kernel(x, y, k, dom.x, dom.y) * clamped(dom.x + beginx, dom.y + beginy, k);
    }

    const RDom all_shifts{0, n_shifts, "all_shifts"};
    Func resampled_sum{"resampled_sum"};
    resampled_sum(x, y) += resampled_at(x, y, all_shifts);

    return {resampled_sum, resampled_at, clamped};
}

Func
boxBlur(const Func& input, const Expr width, const Expr height, const RDom win, const bool needs_clamping, const std::string& name) {
    // Local vars
    assert(input.dimensions() == 2);

    // Repeatable boundary
    Func clamped{"clamped"};

    if (needs_clamping) {
        clamped = BoundaryConditions::repeat_edge(input, {{0, width}, {0, height}});
    } else {
        clamped(x, y) = input(x, y);
    }

    Func blurred_x{name};
    blurred_x(x, y) += clamped(x + win, y);

    Func blurred{"blurred"};
    blurred(x, y) += blurred_x(x, y + win);
   
    return blurred;
}

Func
getSubpixelShift(const Expr n_steps, const Expr upsample_factor, const Expr step_size) {
    Func shift_px{"shift_px"};
    shift_px(i, k) = mux(i, {k / n_steps,  // X coordinate
                             k % n_steps}  // Y coordinate
    );

    const auto offset = 0;  // n_steps * 0.5f;

    Func shift_scaled{"shift_scaled"};
    shift_scaled(i, k) = (offset - shift_px(i, k)) * (step_size * upsample_factor);

    return shift_scaled;
}

}  // namespace image_formation
