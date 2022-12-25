#include <Halide.h>
using namespace Halide;

#include "linearized-admm.h"
#include "problem-definition.h"
#include "ladmm_iter.schedule.h"

class LinearizedADMMIter : public Generator<LinearizedADMMIter> {
    static constexpr auto W = problem_config::input_width;
    static constexpr auto H = problem_config::input_height;
    static constexpr auto M = problem_config::input_layers;

    static constexpr auto W2 = problem_config::output_width;
    static constexpr auto H2 = problem_config::output_height;

   public:
    /** User-provided distorted, and noisy image. */
    Input<Buffer<float, 3>> input{"input"};

    /** User calibrated subpixel alignment */
    Input<float> shift_step{"shift_step"};

    /** Initial estimate of the restored image. */
    Input<Buffer<float, 2>> v{"v"};

    // TODO(Antony): How do we determine the number of inputs z_i at run time? Generator::configure() ?
    // Does Buffer<Func[2]> results in terse code? How to set dimensions?
    Input<Buffer<float, 3>> z0{"z0"};
    Input<Buffer<float, 3>> z1{"z1"};

    Input<Buffer<float, 3>> u0{"u0"};
    Input<Buffer<float, 3>> u1{"u1"};

    /** Problem scaling factor.
     *
     * This influences the convergence rate of the (L-)ADMM algorithm.
     */
    GeneratorParam<float> lmb{"lmb", 1.0f, 0.0f, 1e3f};
    GeneratorParam<float> mu{"mu", 1.0f, 0.0f, 1e3f};

    /** Number of (L-)ADMM iterations before computing convergence metrics.
     *
     * This reduces the overhead of convengence check by a factor of n_iter
     * times. The end-users can decide whether to re-run this pipeline for
     * another n_iter times in their own runtime.
     */
    GeneratorParam<uint32_t> n_iter{"n_iter", 1ul, 1ul, 500ul};

    /** Optimal solution, after a hard termination after iterating for n_iter
     * times. */
    Output<Buffer<float, 2>> v_new{"v_new"};

    // TODO(Antony): How do we figure out the number of outputs z_i at run time? configure() ?
    Output<Buffer<float, 3>> z0_new{"z0_new"};
    Output<Buffer<float, 3>> z1_new{"z1_new"};

    Output<Buffer<float, 3>> u0_new{"u0_new"};
    Output<Buffer<float, 3>> u1_new{"u1_new"};

    // Convergence metrics
    Output<float> r{"r"};  //!< Primal residual
    Output<float> s{"s"};  //!< Dual residual
    Output<float> eps_pri{"eps_pri"};
    Output<float> eps_dual{"eps_dual"};

    void generate() {
        using problem_config::psi_size;
        using problem_definition::K;
        using problem_definition::omega_fn;
        using problem_definition::psi_fns;

        // How to generalize z_i as inputs?
        static_assert(psi_size == 2);

        std::vector<Func> v_list(n_iter);

        std::vector<FuncTuple<psi_size>> z_list(n_iter);
        std::vector<FuncTuple<psi_size>> u_list(n_iter);

        using problem_config::input_height;
        using problem_config::input_size;
        using problem_config::input_width;
        using problem_config::input_layers;
        using problem_config::output_height;
        using problem_config::output_size;
        using problem_config::output_width;

        const std::array<RDom, psi_size> all_dimensions{
            RDom{0, output_width, 0, output_height, 0, 2},
            RDom{0, input_width, 0, input_height, 0, input_layers}
        };

	const RDom output_dimensions{0, output_width, 0, output_height};

        const auto n_steps = 8;
        assert(n_steps * n_steps == M && "Expects 64 low-res images.");
        const Func subpixel_shift = image_formation::getSubpixelShift(8, problem_config::upsample, shift_step);

        for (size_t i = 0; i < n_iter; i++) {
            const Func& v_prev =
                (i == 0) ? v : v_list[i - 1];
            const FuncTuple<psi_size>& z_prev =
                (i == 0) ? FuncTuple<psi_size>{z0, z1} : z_list[i - 1];
            const FuncTuple<psi_size>& u_prev =
                (i == 0) ? FuncTuple<psi_size>{u0, u1} : u_list[i - 1];

            std::tie(v_list[i], z_list[i], u_list[i]) = algorithm::linearized_admm::iterate(
                v_prev, z_prev, u_prev, K, omega_fn, psi_fns, lmb, mu, input, subpixel_shift);
        }

        const auto& z_prev = (n_iter > 1u) ? *(z_list.rbegin() + 1) : FuncTuple<psi_size>{z0, z1};
        const auto [_r, _s, _eps_pri, _eps_dual] = algorithm::linearized_admm::computeConvergence(
            v_list.back(), z_list.back(), u_list.back(), z_prev, K, subpixel_shift, lmb, input_size,
            all_dimensions, output_size,
            output_dimensions);

        // Export data
        v_new = v_list.back();
        std::tie(z0_new, z1_new) = std::make_pair(z_list.back()[0], z_list.back()[1]);
        std::tie(u0_new, u1_new) = std::make_pair(u_list.back()[0], u_list.back()[1]);
        r() = _r;
        s() = _s;
        eps_pri() = _eps_pri;
        eps_dual() = _eps_dual;
    }

    /** Inform Halide of the fixed input and output image sizes. */
    void setBounds() {
        const auto W = input.dim(0).extent();
        const auto H = input.dim(1).extent();

        constexpr auto factor = 128;
	// Raw low resolution images
        input.dim(0).set_bounds(0, W).set_stride(1);
        input.dim(1).set_bounds(0, H / factor * factor).set_stride(W);
        input.dim(2).set_bounds(0, M).set_stride(W*H);

        const auto W2 = W * problem_config::upsample;
        const auto H2 = H * problem_config::upsample;
        constexpr auto factor2 = factor * problem_config::upsample;
	    v.dim(0).set_bounds(0, W2 / factor2 * factor2).set_stride(1);
	    v.dim(1).set_bounds(0, H2 / factor2 * factor2).set_stride(W2);

	const auto setBound0 = [=](auto& z) {
            z.dim(0).set_bounds(0, W2 / factor2 * factor2).set_stride(1);
            z.dim(1).set_bounds(0, H2 / factor2 * factor2).set_stride(W2);
            z.dim(2).set_bounds(0, 2).set_stride(W2*H2);
	};

	// Reconstructed images and the gradient
	setBound0(u0);
	setBound0(z0);
	setBound0(u0_new);
	setBound0(z0_new);

	const auto setBound1_src = [=](auto& u) {
            u.dim(0).set_bounds(0, W2 / factor2 * factor2).set_stride(1);
            u.dim(1).set_bounds(0, H2 / factor2 * factor2).set_stride(W2);
	};

	// HR images
	setBound1_src(v);
	setBound1_src(v_new);

	const auto setBound1_dst = [=](auto& z) {
            z.dim(0).set_bounds(0, W2 / factor2 * factor2).set_stride(1);
            z.dim(1).set_bounds(0, H2 / factor2 * factor2).set_stride(W2);
            z.dim(2).set_bounds(0, M).set_stride(W2*H2);
	};

	// Simulated LR images
	setBound1_dst(u1);
	setBound1_dst(u1_new);
	setBound1_dst(z1);
	setBound1_dst(z1_new);

	// Subpixel shifts
        //subpixel_shift.dim(0).set_bounds(0, 2).set_stride(1);
        //subpixel_shift.dim(1).set_bounds(0, M).set_stride(2);
    }

    void scheduleForCPU() {
        apply_schedule_ladmm_iter(get_pipeline(), get_target());
    }

    void schedule() {
        setBounds();

        if (using_autoscheduler()) {

	    // Reconstructed HR image and the gradient
        v.set_estimates({{0, W2/2}, {0, H2/2}});
	    v_new.set_estimates({{0, W2/2}, {0, H2/2}});

        //subpixel_shift.set_estimates({{0, 2}, {0, M}});
        shift_step.set_estimate(0.125f);

	    const auto setEst0 = [](auto& z) {
		    z.set_estimates({{0, W2/2}, {0, H2/2}, {0, 2}});
        };
	    // Gradient HR images
	    setEst0(z0);
	    setEst0(u0);
	    setEst0(z0_new);
	    setEst0(u0_new);

	    const auto setEst1_dst = [](auto& z) {
		    z.set_estimates({{0, W/2}, {0, H/2}, {0, M}});
		  };

	    // Simulated LR images
	    setEst1_dst(u1);
	    setEst1_dst(z1);
	    setEst1_dst(u1_new);
	    setEst1_dst(z1_new);
	    setEst1_dst(input);

            return;
        }

        // Schedule for CPU
        return scheduleForCPU();
    }
};

HALIDE_REGISTER_GENERATOR(LinearizedADMMIter, ladmm_iter);
