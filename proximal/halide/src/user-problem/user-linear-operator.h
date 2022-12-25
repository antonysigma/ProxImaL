#pragma once

#include <utility>
#include <Halide.h>

namespace image_formation {
using namespace Halide;

/** Applies an Affine transformation to an image. The function A_warp transforms
 * the source using a specified warp matrix. For pixel super-resolution image
 * reconstruction, the function is only used for pure translation.
 *
 * @param[in] input input image
 * @param[in] width image width
 * @param[in] height image height
 * @param[in] upsample upsample ratio
 * @param[in] shift lateral shift in x, y axes, of the g-th low-res image.
 * @return transformed image slices, and the clamped input
 */
std::pair<Func, Func>
A_warp(const Func& input, const Expr width, const Expr height, const Expr upsample,
       const Func& shift);

/** Applies an Affine transformation to an image. The function At_warp
 * transforms the source using a specified scale and shift value.
 *
 * @param[in] input input image
 * @param[in] width image width
 * @param[in] height image height
 * @param[in] upsample upsample ratio
 * @param[in] shift lateral shift in x and y axes, of the g-th Low-res image.
 * @param[in] n_shifts number of low-res image slices.
 * @return fused image, transformed image slices, and the clamped input.
 */
std::tuple<Func, Func, Func>
At_warp(const Func& input, const Expr width, const Expr height, const Expr upsample,
       const Func& shift, const Expr n_shifts);

/** 2D convolution with repeated edge condition.
 *
 * @param[in] input input image
 * @param[in] width image width
 * @param[in] height image height
 * @param[in] box_size convolution kernel
 * @param[in] needs_clamping whether to resolve boundary conditions by repeating the edge pixels.
 * @return convoluted image
 */
Func
boxBlur(const Func& input, const Expr width, const Expr height, const RDom window, bool needs_clamping=true, const std::string& name = "blurred_x");

/** Generate the map of pixel (sub)-pixel shifts.
 *
 * Given the number of LEDs used in the image capture, compute the relative
 * pixel shifts of the k-th frame.
 *
 * @param[in] n_steps Number of LEDs in one row.
 * @param[in] upsample_factor Image 2D interpolation factor; decoupled from the shift step size.
 * @param[in] step_size lateral image shifts as a fraction of the sensor pixel size.
 * @return An array of pixel shifts.
 */
Func getSubpixelShift(const Expr n_steps, const Expr upsample_factor, const Expr step_size);

} // namespace image_formation