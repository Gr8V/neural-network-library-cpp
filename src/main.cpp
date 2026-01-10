#include <iostream>

#include "maths/tensor.h"
#include "mnist/mnist.h"
#include "layers/dense.h"
#include "layers/relu.h"
#include "layers/softmax.h"
#include "network.h"

int main() {
    Network net;
    SoftMax sm;
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
        Tensor probs = sm.forward(logits);

        // Debug
        std::cout << probs.rows << " " << probs.cols << std::endl;
        float sum = 0.f;
        for (size_t i = 0; i < probs.rows; i++)
        {
            std::cout << probs(i,0) << std::endl;
            sum += probs(i,0);
        }
        std::cout << "SUM = " << sum;

    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
    }

}