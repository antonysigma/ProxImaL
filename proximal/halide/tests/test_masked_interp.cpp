#include <cstdint>

#include <armadillo>
#include <HalideBuffer.h>
#include "halide_image_io.h"

#include "masked_interp.h"

using Halide::Runtime::Buffer;
using Halide::Tools::load_and_convert_image;
using Halide::Tools::convert_and_save_image;

int main() {
    Buffer<uint16_t, 2> image = load_and_convert_image("../spots.png");

    {
        using namespace arma;
        Mat<uint16_t> mapped(image.data(), image.width(), image.height(), false, true);
        // Set non-zero black level.
        constexpr uint16_t black_level = 10;
        mapped += black_level;

        mapped.elem(find(mapped <= black_level)).fill(65535);

    }

    Buffer<uint16_t, 2> interpolated(image.width(), image.height());
    masked_interp(image, interpolated);

    convert_and_save_image(image, "masked.png");
    convert_and_save_image(interpolated, "interpolated.png");

    return 0;
}