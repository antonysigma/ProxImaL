#pragma once

#include <Halide.h>

namespace prox {

using namespace Halide;

Var x{"x"}, y{"y"}, c{"c"}, g{"g"}, k{"k"}, l{"l"};

template<int subsample_factor>
Func
proxSubampleSumsq(const Func& input, const Expr theta, const Func& reference, const std::string&& name = "xhat") {
    Func output{name};

    constexpr auto S = subsample_factor;

    const Expr alpha = sqrt(2 / theta);
    const Expr beta = theta / 2 + 1;
    output(x, y, c) = select(
        x % S == 0 && y % S == 0,
        (reference(x / S, y / S, c) + input(x, y, c) / alpha ) / beta,
        input(x, y, c) * theta / 2);

    return output;
}


} // namespace prox