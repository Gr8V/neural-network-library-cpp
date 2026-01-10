#include <iostream>

#include "maths/tensor.h"
#include "mnist/mnist.h"
#include "layers/dense.h"
#include "layers/relu.h"
#include "losses/softmax_cross_entropy_loss.h"
#include "network.h"
#include "optimizers/SGD.h"
#include "training/trainer.h"

int main() {
    try {
        auto train = loadMNIST(
            "data/train-images.idx3-ubyte",
            "data/train-labels.idx1-ubyte"
        );
        auto test = loadMNIST(
            "data/t10k-images.idx3-ubyte",
            "data/t10k-labels.idx1-ubyte"
        );

        Network net;
        SoftmaxCrossEntropyLoss loss_fn;
        SGD optimizer(0.01f);

        Trainer trainer(net, loss_fn, optimizer);

        trainer.train(train, 8);

        trainer.evaluate(test);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
    }
}