#include <iostream>

#include "maths/tensor.h"
#include "mnist/mnist.h"
#include "layers/dense.h"
#include "layers/relu.h"
#include "layers/softmax.h"
#include "losses/cross_entropy_loss.h"
#include "network.h"

int main() {
    Network net;
    SoftMax sm;
    CrossEntropyLoss loss_fn;
    try {
        auto train = loadMNIST(
            "data/train-images.idx3-ubyte",
            "data/train-labels.idx1-ubyte"
        );
        printMNISTImage(
            train.images[0],
            train.rows,
            train.cols
        );

        // Convert MNIST image to tensor (784 × 1)
        Tensor image = net.image_to_tensor(train.images[0]);
        int label = train.labels[0];

        Tensor logits = net.forward(image);
        Tensor probs = sm.forward(logits);
        float loss = loss_fn.forward(probs, label);

        // Debug
        std::cout << probs.rows << " " << probs.cols << std::endl;
        for (size_t i = 0; i < probs.rows; i++)
        {
            std::cout << probs(i,0) << std::endl;
        }
        std::cout << "LOSS = " << loss << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
    }

}