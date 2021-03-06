#include "util.hpp"
#include "warpImgT.h"

namespace proximal {

int At_warp_glue(const array_float_t input, const array_float_t H,
    array_float_t output) {

        auto input_buf = getHalideBuffer<4>(input);
        auto H_buf = getHalideBuffer<3>(H);
        auto output_buf = getHalideBuffer<3>(output);

        return warpImgT(input_buf, H_buf, output_buf);
    }

} // proximal

PYBIND11_MODULE(libAt_warp, m) {
    m.def("run", &proximal::At_warp_glue, "Apply inverse affine transform");
}
