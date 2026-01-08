#include "mnist.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <vector>

static uint32_t readBigEndianUInt32(std::ifstream& file) {
    uint32_t value = 0;
    file.read(reinterpret_cast<char*>(&value), 4);
    return
        ((value & 0x000000FF) << 24) |
        ((value & 0x0000FF00) << 8)  |
        ((value & 0x00FF0000) >> 8)  |
        ((value & 0xFF000000) >> 24);
}

MNISTDataset loadMNIST(
    const std::string& imagePath,
    const std::string& labelPath
) {
    std::ifstream images(imagePath, std::ios::binary);
    std::ifstream labels(labelPath, std::ios::binary);

    if (!images || !labels)
        throw std::runtime_error("Failed to open MNIST files");

    uint32_t imageMagic = readBigEndianUInt32(images);
    uint32_t numImages = readBigEndianUInt32(images);
    uint32_t rows       = readBigEndianUInt32(images);
    uint32_t cols       = readBigEndianUInt32(images);

    uint32_t labelMagic = readBigEndianUInt32(labels);
    uint32_t numLabels = readBigEndianUInt32(labels);

    if (numImages != numLabels)
        throw std::runtime_error("Image-label count mismatch");

    MNISTDataset dataset;
    dataset.rows = rows;
    dataset.cols = cols;

    dataset.images.resize(numImages, std::vector<float>(rows * cols));
    dataset.labels.resize(numLabels);

    for (uint32_t i = 0; i < numImages; ++i) {
        for (uint32_t j = 0; j < rows * cols; ++j) {
            unsigned char pixel = 0;
            images.read(reinterpret_cast<char*>(&pixel), 1);
            dataset.images[i][j] = pixel / 255.0f;
        }

        unsigned char label = 0;
        labels.read(reinterpret_cast<char*>(&label), 1);
        dataset.labels[i] = static_cast<int>(label);
    }

    return dataset;
}


void printMNISTImage(const std::vector<float>& img, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float pixel = img[i * cols + j];

            char c;
            if (pixel > 0.8f)      c = '#';
            else if (pixel > 0.6f) c = 'O';
            else if (pixel > 0.4f) c = 'o';
            else if (pixel > 0.2f) c = '.';
            else                   c = ' ';

            std::cout << c << ' ';
        }
        std::cout << '\n';
    }
}