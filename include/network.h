#pragma once
#include "maths/tensor.h"
#include "layers/dense.h"
#include "layers/relu.h"

class Network {
public:
    Dense d1;
    ReLU r1;
    Dense d2;
    ReLU r2;
    Dense d3;

    Network();
    Tensor forward(const Tensor& x);
    Tensor image_to_tensor(const std::vector<float>& image);
};