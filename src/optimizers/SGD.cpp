#include "optimizers/SGD.h"

SGD::SGD(float learning_rate)
    : lr(learning_rate) {}

void SGD::step(Network& net) {
    // Update d1
    for (int i = 0; i < net.d1.W.rows; ++i) {
        for (int j = 0; j < net.d1.W.cols; ++j) {
            net.d1.W(i, j) -= lr * net.d1.dW(i, j);
        }
        net.d1.b(i, 0) -= lr * net.d1.db(i, 0);
    }

    // Update d2
    for (int i = 0; i < net.d2.W.rows; ++i) {
        for (int j = 0; j < net.d2.W.cols; ++j) {
            net.d2.W(i, j) -= lr * net.d2.dW(i, j);
        }
        net.d2.b(i, 0) -= lr * net.d2.db(i, 0);
    }

    // Update d3
    for (int i = 0; i < net.d3.W.rows; ++i) {
        for (int j = 0; j < net.d3.W.cols; ++j) {
            net.d3.W(i, j) -= lr * net.d3.dW(i, j);
        }
        net.d3.b(i, 0) -= lr * net.d3.db(i, 0);
    }
}
