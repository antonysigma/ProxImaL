////////////////////////////////////////////////////////////////////////////////
//Convolution as part of image formation.
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
#include <vector>
using namespace Halide;

class masked_interp_gen : public Generator<masked_interp_gen> {
    Func input_clamped{"input_clamped"};
    std::vector<Func> downscaled;
    std::vector<Func> upscaled;

    Var x{"x"}, y{"y"};
public:
    static constexpr uint16_t INVALID = 65535;

    Input<Buffer<uint16_t, 2>> input{"input"};
    Output<Buffer<uint16_t, 2>> output{"output"};

    GeneratorParam<int> downsample_level{"downsample_level", 5, 1, 8};

    Func do_downscale(const Func& in) const {
        Func ds{"downscaled"};

        Func masked{"masked"};
        masked(x, y) = select(in(x, y) == INVALID, uint16_t(0), in(x, y));

        Func sum2by2{"sum2by2"};
        sum2by2(x, y) = {
            cast<uint32_t>(masked(x, y)) + masked(x + 1, y) +
            masked(x, y + 1) + masked(x + 1, y + 1),
            min(masked(x, y), 1) + min(masked(x + 1, y), 1) +
            min(masked(x, y + 1), 1) + min(masked(x + 1, y + 1), 1)};

        const Expr weighted_sum = sum2by2(x * 2, y * 2)[0];
        const Expr pixel_count = sum2by2(x * 2, y * 2)[1];

        // Bug: don't paint over zero valued pixels.
        ds(x, y) = select(pixel_count > 0, weighted_sum / pixel_count, INVALID);

        return ds;
    }

    Func do_upscale(const Func& current_level, const Func& next_level) const {
        Func upscaled{"upscaled"};

        upscaled(x, y) = select(next_level(x, y) != INVALID, next_level(x, y),
            current_level(x / 2, y / 2)
        );

        return upscaled;
    }

    void generate () {
        const auto width = input.width();
        const auto height = input.height();
        
        input_clamped = BoundaryConditions::repeat_edge(input, {{0, width}, {0, height}});
        Func input_f32{"input_f32"};
        input_f32(x, y) = cast<float>(input_clamped(x, y));

        downscaled.emplace_back(input_f32);

        for(int i = 0; i < downsample_level + 1; i++) {
            downscaled.emplace_back(do_downscale(downscaled[i]));
        }

        for(int i = 0; i < downsample_level + 1; i++) {
            const Func& current_level = *(downscaled.rbegin() + i);
            const Func& next_level = *(downscaled.rbegin() + i + 1);

            upscaled.emplace_back(do_upscale(current_level, next_level));
        }

        const Func& interpolated = upscaled.back();
        output(x, y) = cast<uint16_t>(interpolated(x, y));
    }

    void schedule() {
        input.dim(0).set_bounds(0, 128).set_stride(1);
        input.dim(1).set_bounds(0, 128).set_stride(128);

        output.dim(0).set_bounds(0, 128).set_stride(1);
        output.dim(1).set_bounds(0, 128).set_stride(128);

        if (using_autoscheduler()) {
            input.set_estimates({{0, 128}, {0, 128}});
            output.set_estimates({{0, 128}, {0, 128}});
            return;
        }

        for(auto& f : downscaled) {
            f.compute_root();
        }

        for(auto& f : upscaled) {
            f.compute_root();
        }
    }
};

HALIDE_REGISTER_GENERATOR(masked_interp_gen, masked_interp);