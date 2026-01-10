#include <iostream>
#include "training/trainer.h"
#include "utils/metrics.h"

Trainer::Trainer(Network& n,
                SoftmaxCrossEntropyLoss& lf,
                SGD& opt)
    : net(n), loss_fn(lf), optimizer(opt) {}

void Trainer::train(const MNISTDataset& train_data, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int correct = 0;

        for (size_t i = 0; i < train_data.images.size(); ++i) {
            Tensor image = net.image_to_tensor(train_data.images[i]);
            int label = train_data.labels[i];

            // Forward
            Tensor logits = net.forward(image);
            float loss = loss_fn.forward(logits, label);
            total_loss += loss;

            // Accuracy
            int pred = argmax(logits);
            if (pred == label)
                correct++;

            // Backward
            Tensor grad = loss_fn.backward();
            net.backward(grad);

            // Update
            optimizer.step(net);
        }

        float avg_loss = total_loss / train_data.images.size();
        float accuracy = (float)correct / train_data.images.size();

        std::cout << "Epoch " << epoch + 1
                << " | Loss: " << avg_loss
                << " | Accuracy: " << accuracy * 100.0f << "%\n";
    }
}

float Trainer::evaluate(const MNISTDataset& test_data) {
    int correct = 0;

    for (size_t i = 0; i < test_data.images.size(); ++i) {
        Tensor image = net.image_to_tensor(test_data.images[i]);
        int label = test_data.labels[i];

        Tensor logits = net.forward(image);
        int pred = argmax(logits);

        if (pred == label)
            correct++;
    }

    return (float)correct / test_data.images.size();
}
