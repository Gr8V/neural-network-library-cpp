#include <iostream>

#include "maths/tensor.h"
#include "mnist.h"

int main() {
    try {
        auto train = loadMNIST(
            "data/train-images.idx3-ubyte",
            "data/train-labels.idx1-ubyte"
        );

        printMNISTImage(
            train.images[3],
            train.rows,
            train.cols
        );

    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
    }
}