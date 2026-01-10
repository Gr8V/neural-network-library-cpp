#pragma once
#include "network.h"
#include "losses/softmax_cross_entropy_loss.h"
#include "optimizers/SGD.h"
#include "mnist/mnist.h"

class Trainer {
public:
    Trainer(Network& net,
            SoftmaxCrossEntropyLoss& loss_fn,
            SGD& optimizer);

    void train(const MNISTDataset& train_data, int epochs);
    float evaluate(const MNISTDataset& test_data);

private:
    Network& net;
    SoftmaxCrossEntropyLoss& loss_fn;
    SGD& optimizer;
};
