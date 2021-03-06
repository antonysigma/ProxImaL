#include "util.hpp"
#include "proxL1.h"

namespace proximal {

int prox_L1_glue(const array_float_t input, const float theta, array_float_t output) {

        auto input_buf = getHalideBuffer<4>(input);
        auto output_buf = getHalideBuffer<4>(output);

        return proxL1(input_buf, theta, output_buf);
    }

} // proximal

PYBIND11_MODULE(libprox_L1, m) {
    m.def("run", &proximal::prox_L1_glue, "Apply soft thresholding");
}