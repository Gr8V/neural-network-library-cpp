#pragma once
#include <vector>
#include <cassert>

class Tensor {
public:
    int rows;
    int cols;
    std::vector<float> data;

    // Constructors
    Tensor();
    Tensor(int r, int c);
    Tensor(int r, int c, float initVal);

    // Element access
    float& operator()(int r, int c);
    const float& operator()(int r, int c) const;

    // Utilities
    void fill(float value);
    void randomize(float min = -0.1f, float max = 0.1f);

    // Shape helpers
    int size() const;
};
