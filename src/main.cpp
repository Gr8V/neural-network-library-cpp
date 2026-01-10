#include <iostream>

#include "maths/tensor.h"
#include "mnist/mnist.h"
#include "layers/dense.h"
#include "layers/relu.h"
#include "losses/softmax_cross_entropy_loss.h"
#include "network.h"
#include "optimizers/SGD.h"

int main() {
    Network net;
    SoftmaxCrossEntropyLoss loss_fn;
    SGD optimizer(0.01f);
    try {
        auto train = loadMNIST(
            "data/train-images.idx3-ubyte",
            "data/train-labels.idx1-ubyte"
        );
        // Convert MNIST image to tensor (784 × 1)
        Tensor image = net.image_to_tensor(train.images[0]);
        int label = train.labels[0];

        // Forward
        Tensor logits = net.forward(image);
        float loss = loss_fn.forward(logits, label);

        // Backward
        Tensor grad_logists = loss_fn.backward();
        net.backward(grad_logists);

        // Update

        // Debug
        for (int i = 0; i < 100; ++i) {
            Tensor logits = net.forward(image);
            float loss = loss_fn.forward(logits, label);

            Tensor grad = loss_fn.backward();
            net.backward(grad);
            optimizer.step(net);

            std::cout << loss << std::endl;
        }


    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
    }

}