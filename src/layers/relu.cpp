#include "layers/relu.h"

Tensor ReLU::forward(const Tensor& x) {
    input = x;  // cache for backprop later

    Tensor out(x.rows, x.cols);

    for (int i = 0; i < x.rows; ++i) {
        for (int j = 0; j < x.cols; ++j) {
            out(i, j) = x(i, j) > 0.0f ? x(i, j) : 0.0f;
        }
    }

    return out;
}

Tensor ReLU::backward(const Tensor& grad_out) {
    Tensor grad_input(input.rows, input.cols);

    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            if (input(i, j) > 0.0f)
                grad_input(i, j) = grad_out(i, j);
            else
                grad_input(i, j) = 0.0f;
        }
    }

    return grad_input;
}