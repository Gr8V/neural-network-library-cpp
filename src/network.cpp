#include "network.h"

Network::Network()
    : d1(784, 256),
    d2(256, 128),
    d3(128, 10)
{}

Tensor Network::forward(const Tensor& x) {
    Tensor out = d1.forward(x);
    out = r1.forward(out);

    out = d2.forward(out);
    out = r2.forward(out);

    out = d3.forward(out);

    return out; // logits (10 × 1)
}

void Network::backward(const Tensor& grad_logits) {
    // grad_logits: (10 × 1)

    Tensor grad = grad_logits;

    // Last Dense (64 → 10)
    grad = d3.backward(grad);

    // ReLU (after d2)
    grad = r2.backward(grad);

    // Dense (128 → 64)
    grad = d2.backward(grad);

    // ReLU (after d1)
    grad = r1.backward(grad);

    // Dense (784 → 128)
    grad = d1.backward(grad);

    // grad now is ∂Loss / ∂input (unused)
}

Tensor Network::image_to_tensor(const std::vector<float>& image) {
    Tensor x(784, 1);
    for (int i = 0; i < 784; ++i)
        x(i, 0) = image[i];
    return x;
}