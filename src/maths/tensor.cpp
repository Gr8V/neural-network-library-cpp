#include "maths/tensor.h"
#include <random>

// Default constructor
Tensor::Tensor()
    : rows(0), cols(0) {}

// Zero-initialized tensor
Tensor::Tensor(int r, int c)
    : rows(r), cols(c), data(r * c, 0.0f) {}

// Constant-initialized tensor
Tensor::Tensor(int r, int c, float initVal)
    : rows(r), cols(c), data(r * c, initVal) {}

// Mutable access
float& Tensor::operator()(int r, int c) {
    assert(r >= 0 && r < rows);
    assert(c >= 0 && c < cols);
    return data[r * cols + c];
}

// Const access
const float& Tensor::operator()(int r, int c) const {
    assert(r >= 0 && r < rows);
    assert(c >= 0 && c < cols);
    return data[r * cols + c];
}

// Fill with constant
void Tensor::fill(float value) {
    std::fill(data.begin(), data.end(), value);
}

// Random initialization (Xavier-like basic init)
void Tensor::randomize(float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);

    for (auto& v : data)
        v = dist(gen);
}

// Total number of elements
int Tensor::size() const {
    return rows * cols;
}

float Tensor::max_value() const {
    float max_val = data[0];
    for (float v : data) {
        if (v > max_val)
            max_val = v;
    }
    return max_val;
}