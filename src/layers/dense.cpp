#include "layers/dense.h"
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

    // y = W*x + b
    return out;
}

Tensor Dense::backward(const Tensor& grad_out) {
    // grad_out shape: (out × 1)
    // input shape:    (in × 1)
    // W shape:        (out × in)

    int out = W.rows; // 10
    int in  = W.cols; // 64

    // Initialize gradients
    dW = Tensor(out, in);
    db = Tensor(out, 1);

    // Compute dW = grad_out * input^T
    for (int i = 0; i < out; ++i) {
        for (int j = 0; j < in; ++j) {
            dW(i, j) = grad_out(i, 0) * input(j, 0);
        }
    }

    // Compute db = grad_out
    for (int i = 0; i < out; ++i) {
        db(i, 0) = grad_out(i, 0);
    }

    // Compute grad_input = W^T * grad_out
    Tensor grad_input(in, 1);

    for (int j = 0; j < in; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < out; ++i) {
            sum += W(i, j) * grad_out(i, 0);
        }
        grad_input(j, 0) = sum;
    }

    return grad_input;
}
