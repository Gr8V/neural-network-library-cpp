#include "layers/dense.h";
#include <random>

Dense::Dense(int in, int out)
    : W(out, in), b(out, 1)
{
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    for (int i = 0; i < W.rows; ++i) {
        for (int j = 0; j < W.cols; ++j) {
            W(i, j) = dist(gen);
        }
    }

    for (int i = 0; i < b.rows; ++i) {
        b(i, 0) = 0.0f;
    }
}

Tensor Dense::forward(const Tensor& x) {
    // Cache input for backprop later
    input = x;

    // Output: (out, 1)
    Tensor out(W.rows, 1);

    for (int i = 0; i < W.rows; ++i) {
        float sum = 0.0f;

        for (int j = 0; j < W.cols; ++j) {
            sum += W(i, j) * x(j, 0);
        }

        out(i, 0) = sum + b(i, 0);
    }

    return out;
}
