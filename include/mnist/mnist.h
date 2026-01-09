#ifndef MNIST_H
#define MNIST_H

#include <vector>
#include <string>
#include <cstdint>

struct MNISTDataset {
    std::vector<std::vector<float>> images; // normalized [0,1]
    std::vector<int> labels;
    uint32_t rows = 0;
    uint32_t cols = 0;
};

MNISTDataset loadMNIST(
    const std::string& imagePath,
    const std::string& labelPath
);

void printMNISTImage(const std::vector<float>& img, int rows, int cols);

#endif // MNIST_H
