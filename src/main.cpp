#include <iostream>

#include "maths/tensor.h"
#include "mnist/mnist.h"
#include "layers/dense.h"
#include "network.h"

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

        Dense d(784, 128);
        Tensor x = image_to_tensor(train.images[0]);
        Tensor y = d.forward(x);
        std::cout << y.rows << " " << y.cols << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
    }

}