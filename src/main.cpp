#include <iostream>

#include "maths/tensor.h"
#include "mnist/mnist.h"
#include "layers/dense.h"
#include "layers/relu.h"
#include "losses/softmax_cross_entropy_loss.h"
#include "network.h"

int main() {
    Network net;
    SoftmaxCrossEntropyLoss loss_fn;
    try {
        auto train = loadMNIST(
            "data/train-images.idx3-ubyte",
            "data/train-labels.idx1-ubyte"
        );
        // Convert MNIST image to tensor (784 × 1)
        Tensor image = net.image_to_tensor(train.images[0]);
        int label = train.labels[0];

        Tensor logits = net.forward(image);
        float loss = loss_fn.forward(logits, label);
        Tensor grad_logists = loss_fn.backward();
        net.backward(grad_logists);

        // Debug
        std::cout << "LOSS = " << loss << std::endl;
        std::cout << net.d1.dW(0,0) << std::endl;

    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
    }

}