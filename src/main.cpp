#include <iostream>

#include "maths/tensor.h"
#include "mnist/mnist.h"
#include "layers/dense.h"
#include "layers/relu.h"
#include "network.h"

int main() {
    Network net;
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

        // Convert MNIST image to tensor (784 × 1)
        Tensor image = net.image_to_tensor(train.images[0]);

        Tensor logits = net.forward(image);

        // Debug
        std::cout << logits.rows << " " << logits.cols << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
    }

}