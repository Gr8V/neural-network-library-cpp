#pragma once
#include "maths/tensor.h"

class SoftmaxCrossEntropyLoss {
public:
    Tensor probs;
    int label;

    float forward(const Tensor& logits, int label);
    Tensor backward();
};