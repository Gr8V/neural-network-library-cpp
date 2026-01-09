#include "network.h";


static Tensor image_to_tensor(const std::vector<float>& image) {
    Tensor x(784, 1);
    for (int i = 0; i < 784; ++i)
        x(i, 0) = image[i];
    return x;
}