#include <vector>

class Tensor {
public:
    int rows, cols;
    std::vector<float> data;

    Tensor(int r, int c);
    float& operator()(int r, int c);
};
