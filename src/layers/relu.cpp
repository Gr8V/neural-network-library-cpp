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
