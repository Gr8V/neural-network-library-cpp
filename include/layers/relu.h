#pragma once
#include "maths/tensor.h"

class ReLU {
public:
    Tensor input;

    Tensor forward(const Tensor& x);
};
