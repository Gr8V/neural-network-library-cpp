#pragma once
#include "maths/tensor.h"

class SoftMax {
public:
    Tensor output;

    Tensor forward(const Tensor& x);
};