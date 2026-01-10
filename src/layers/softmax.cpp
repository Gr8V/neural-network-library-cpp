#include "layers/softmax.h"
#include <cmath>
#include <algorithm>

Tensor SoftMax::forward(const Tensor& x) {
    // x is (N x 1)
    Tensor out(x.rows, x.cols);

    // Finx Max Value
    float max_val = x.max_value();

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < x.rows; ++i) {
        out(i, 0) = std::exp(x(i, 0) - max_val);
        sum += out(i, 0);
    }

    // Normalize
    for (int i = 0; i < x.rows; ++i) {
        out(i, 0) /= sum;
    }

    output = out;
    return out;
}