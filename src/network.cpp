#include "network.h"

Network::Network()
    : d1(784, 128),
    d2(128, 64),
    d3(64, 10)
{
}

Tensor Network::forward(const Tensor& x) {
    Tensor out = d1.forward(x);
    out = r1.forward(out);

    out = d2.forward(out);
    out = r2.forward(out);

    out = d3.forward(out);

    return out; // logits (10 × 1)
}

Tensor Network::image_to_tensor(const std::vector<float>& image) {
    Tensor x(784, 1);
    for (int i = 0; i < 784; ++i)
        x(i, 0) = image[i];
    return x;
}