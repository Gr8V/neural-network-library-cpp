#pragma once
#include "maths/tensor.h"

class CrossEntropyLoss {
public:
    float forward(const Tensor& probs, int label);
};