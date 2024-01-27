////////////////////////////////////////////////////////////////////////////////
//Convolution as part of image formation.
////////////////////////////////////////////////////////////////////////////////

#include <Halide.h>
using namespace Halide;

class loco_gen : public Generator<loco_gen> {
    Func input_clamped{"input_clamped"};
    Func predicted{"predicted"};
    Func residual{"residual"};

    Func buffer{"buffer"};
    Func symbol{"symbol"};
    RDom r;

    Var x{"x"}, y{"y"};
public:

    Input<Buffer<uint16_t, 2>> input{"input"};
    Output<Buffer<uint16_t, 2>> output{"output"};
    GeneratorParam<int> quantization{"quantization", 2, 1, 8};

    Expr predict(const Expr a, const Expr b, const Expr c) {
        return select( c > max(a, b), cast<int32_t>(min(a, b)), //
        c < min(a, b),
            cast<int32_t>(max(a, b)), //
        cast<int32_t>(a) + b - c);
    }

    void generate () {
        const auto width = input.width();
        const auto height = input.height();
        
        input_clamped = BoundaryConditions::constant_exterior(input, 0, {{0, width}, {0, height}});

        if (quantization == 1) {
            predicted(x, y) = predict(input_clamped(x - 1, y), input_clamped(x, y - 1), input_clamped(x - 1, y - 1));
            residual(x, y) = input(x, y) - predicted(x, y);

            symbol(x, y) = abs(residual(x, y)) * 2 + select(residual(x, y) < 0, -1, 0);

            output(x, y) = cast<uint16_t>(symbol(x, y));
            return;
        }

        // else if quantization >= 2

        r = RDom(0, width, 0, height, "r");
        buffer(x, y) = {undef<int32_t>(), cast<int32_t>(input_clamped(x, y))};

        const Expr predicted_value = predict(buffer(r.x - 1, r.y)[1], buffer(r.x, r.y - 1)[1],buffer(r.x - 1, r.y - 1)[1]);
        const Expr residual_value = (buffer(r.x, r.y)[1] - predicted_value) / float(int(quantization));
        const Expr quantized_value = cast<int32_t>(residual_value + select(residual_value < 0, -0.5f, 0.5f));
        buffer(r.x, r.y) = {quantized_value, quantized_value * quantization + predicted_value};

        const Expr new_quantized_value = buffer(x, y)[0];
        symbol(x, y) = abs(new_quantized_value) * 2 + select(new_quantized_value < 0, -1, 0);

        output(x, y) = cast<uint16_t>(symbol(x, y));
    }

    void schedule() {
        input.dim(0).set_bounds(0, 512).set_stride(1);
        input.dim(1).set_bounds(0, 512).set_stride(512);

        output.dim(0).set_bounds(0, 512).set_stride(1);
        output.dim(1).set_bounds(0, 512).set_stride(512);

        if (using_autoscheduler()) {
            input.set_estimates({{0, 512}, {0, 512}});
            output.set_estimates({{0, 512}, {0, 512}});
            return;
        }

        if (quantization == 1) {
            residual.compute_root();
            predicted.compute_root();
            input_clamped.compute_root();
            return;
        } 

        buffer.compute_root();
        input_clamped.compute_root();
    }
};

HALIDE_REGISTER_GENERATOR(loco_gen, loco);