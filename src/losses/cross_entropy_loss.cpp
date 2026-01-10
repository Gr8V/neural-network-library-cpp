#include "losses/cross_entropy_loss.h"
#include <cmath>

float CrossEntropyLoss::forward(const Tensor& probs, int label) {
    const float epsilon = 1e-9f;
    float p = probs(label, 0);
    
    return -std::log(p + epsilon);
}