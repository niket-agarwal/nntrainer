/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	layers.cpp
 * @date	04 December 2019
 * @brief	This is Layers Classes for Neural Network
 * @see		https://github.sec.samsung.net/jijoong-moon/Transfer-Learning.git
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "include/layers.h"
#include <assert.h>

/**
 * @brief     random function
 * @param[in] x float
 */
float random(float x) { return (float)(rand() % 10000 + 1) / 10000 - 0.5; }

/**
 * @brief     sqrt function for float type
 * @param[in] x float
 */
float sqrt_float(float x) { return (float)(sqrt(x)); }

/**
 * @brief     log function for float type
 * @param[in] x float
 */
float log_float(float x) { return (float)(log(x)); }

/**
 * @brief     sigmoid activation function
 * @param[in] x input
 */
float sigmoid(float x) { return 1 / (1 + exp(-x)); }

/**
 * @brief     derivative sigmoid function
 * @param[in] x input
 */
float sigmoidePrime(float x) { return (float)(1.0 / ((1 + exp(-x)) * (1.0 + 1.0 / (exp(-x) + 0.0000001)))); }

/**
 * @brief     tanh function for float type
 * @param[in] x input
 */
float tanh_float(float x) { return (float)tanh(x); }

/**
 * @brief     derivative tanh function
 * @param[in] x input
 */
float tanhPrime(float x) {
  float th = (float)tanh(x);
  return 1.0 - th * th;
}

namespace Layers {

void InputLayer::setOptimizer(Optimizer opt) {
  this->opt = opt;
  switch (opt.activation) {
    case ACT_TANH:
      activation = tanh_float;
      activationPrime = tanhPrime;
      break;
    case ACT_SIGMOID:
      activation = sigmoid;
      activationPrime = sigmoidePrime;
      break;
    default:
      break;
  }
}

void InputLayer::copy(Layer *l) {
  InputLayer *from = static_cast<InputLayer *>(l);
  this->opt = from->opt;
  this->index = from->index;
  this->height = from->height;
  this->width = from->width;
  this->Input.copy(from->Input);
  this->hidden.copy(from->hidden);
}

Tensor InputLayer::forwarding(Tensor input) {
  Input = input;
  return Input;
}

void InputLayer::initialize(int b, int h, int w, int id, bool init_zero) {
  batch = b;
  width = w;
  height = h;
  index = 0;
}

void FullyConnectedLayer::initialize(int b, int h, int w, int id, bool init_zero) {
  this->batch = b;
  this->width = w;
  this->height = h;
  this->index = id;
  this->init_zero = init_zero;

  Weight = Tensor(h, w);
  Bias = Tensor(1, w);

  Weight = Weight.applyFunction(random);
  if (init_zero) {
    Bias.setZero();
  } else {
    Bias = Bias.applyFunction(random);
  }
}

void FullyConnectedLayer::setOptimizer(Optimizer opt) {
  this->opt = opt;
  switch (opt.activation) {
    case ACT_TANH:
      activation = tanh_float;
      activationPrime = tanhPrime;
      break;
    case ACT_SIGMOID:
      activation = sigmoid;
      activationPrime = sigmoidePrime;
      break;
    default:
      break;
  }
  if (opt.type == OPT_ADAM) {
    M = Tensor(height, width);
    V = Tensor(height, width);
    M.setZero();
    V.setZero();
  }
}

Tensor FullyConnectedLayer::forwarding(Tensor input) {
  Input = input;
  hidden = Input.dot(Weight).add(Bias).applyFunction(activation);
  return hidden;
}

void FullyConnectedLayer::read(std::ifstream &file) {
  Weight.read(file);
  Bias.read(file);
}

void FullyConnectedLayer::save(std::ofstream &file) {
  Weight.save(file);
  Bias.save(file);
}

void FullyConnectedLayer::copy(Layer *l) {
  FullyConnectedLayer *from = static_cast<FullyConnectedLayer *>(l);
  this->opt = from->opt;
  this->index = from->index;
  this->height = from->height;
  this->width = from->width;
  this->Input.copy(from->Input);
  this->hidden.copy(from->hidden);
  this->Weight.copy(from->Weight);
  this->Bias.copy(from->Bias);
}

Tensor FullyConnectedLayer::backwarding(Tensor derivative, int iteration) {
  Tensor dJdB = derivative.multiply(Input.dot(Weight).add(Bias).applyFunction(activationPrime));
  Tensor dJdW = Input.transpose().dot(dJdB);
  Tensor ret = dJdB.dot(Weight.transpose());

  float ll = opt.learning_rate;
  if (opt.decay_steps != -1) {
    ll = opt.learning_rate * pow(opt.decay_rate, (iteration / opt.decay_steps));
  }

  switch (opt.type) {
    case OPT_SGD:
      Weight = Weight.subtract(dJdW.average().multiply(ll));
      break;
    case OPT_ADAM:
      M = M.multiply(opt.beta1).add(dJdW.average().multiply(1 - opt.beta1));
      V = V.multiply(opt.beta2).add((dJdW.average().multiply(dJdW.average())).multiply(1 - opt.beta2));
      M.divide(1 - pow(opt.beta1, iteration + 1));
      V.divide(1 - pow(opt.beta2, iteration + 1));
      Weight = Weight.subtract((M.divide(V.applyFunction(sqrt_float).add(opt.epsilon))).multiply(ll));
      break;
    default:
      break;
  }

  if (!this->init_zero) {
    Bias = Bias.subtract(dJdB.average().multiply(ll));
  }

  return ret;
}

void OutputLayer::initialize(int b, int h, int w, int id, bool init_zero) {
  this->batch = b;
  this->width = w;
  this->height = h;
  this->index = id;
  this->init_zero = init_zero;
  Weight = Tensor(h, w);
  Bias = Tensor(1, w);
  this->cost = cost;

  Weight = Weight.applyFunction(random);
  if (init_zero) {
    Bias.setZero();
  } else {
    Bias = Bias.applyFunction(random);
  }
}

Tensor OutputLayer::forwarding(Tensor input) {
  Input = input;
  if (cost == COST_ENTROPY)
    hidden = input.dot(Weight).applyFunction(activation);
  else
    hidden = input.dot(Weight).add(Bias).applyFunction(activation);
  return hidden;
}

void OutputLayer::read(std::ifstream &file) {
  Weight.read(file);
  Bias.read(file);
}

void OutputLayer::save(std::ofstream &file) {
  Weight.save(file);
  Bias.save(file);
}

void OutputLayer::copy(Layer *l) {
  OutputLayer *from = static_cast<OutputLayer *>(l);
  this->opt = from->opt;
  this->index = from->index;
  this->height = from->height;
  this->width = from->width;
  this->Input.copy(from->Input);
  this->hidden.copy(from->hidden);
  this->Weight.copy(from->Weight);
  this->Bias.copy(from->Bias);
  this->loss = from->loss;
}

void OutputLayer::setOptimizer(Optimizer opt) {
  this->opt = opt;
  switch (opt.activation) {
    case ACT_TANH:
      activation = tanh_float;
      activationPrime = tanhPrime;
      break;
    case ACT_SIGMOID:
      activation = sigmoid;
      activationPrime = sigmoidePrime;
      break;
    default:
      break;
  }

  if (opt.type == OPT_ADAM) {
    M = Tensor(height, width);
    V = Tensor(height, width);
    M.setZero();
    V.setZero();
  }
}

Tensor OutputLayer::backwarding(Tensor label, int iteration) {
  float lossSum = 0.0;
  Tensor Y2 = label;
  Tensor Y = hidden;
  Tensor ret;
  Tensor dJdB;

  float ll = opt.learning_rate;
  if (opt.decay_steps != -1) {
    ll = opt.learning_rate * pow(opt.decay_rate, (iteration / opt.decay_steps));
  }

  if (cost == COST_ENTROPY) {
    dJdB = Y.subtract(Y2);
    Tensor temp = ((Y2.multiply(-1.0).transpose().dot(Y.add(opt.epsilon).applyFunction(log_float)))
                       .subtract(Y2.multiply(-1.0).add(1.0).transpose().dot(
                           Y.multiply(-1.0).add(1.0).add(opt.epsilon).applyFunction(log_float))));
    loss = (1.0 / Y.Mat2Vec().size()) * temp.Mat2Vec()[0];
  } else {
    Tensor sub = Y2.subtract(Y);
    Tensor l = (sub.multiply(sub)).sum().multiply(0.5);
    std::vector<float> t = l.Mat2Vec();
    for (int i = 0; i < l.getBatch(); i++) {
      lossSum += t[i];
    }

    loss = lossSum / (float)l.getBatch();

    dJdB = Y.subtract(Y2).multiply(Input.dot(Weight).add(Bias).applyFunction(activationPrime));
  }

  Tensor dJdW = Input.transpose().dot(dJdB);
  ret = dJdB.dot(Weight.transpose());

  switch (opt.type) {
    case Layers::OPT_SGD:
      Weight = Weight.subtract(dJdW.average().multiply(ll));
      break;
    case Layers::OPT_ADAM:
      M = M.multiply(opt.beta1).add(dJdW.average().multiply(1 - opt.beta1));
      V = V.multiply(opt.beta2).add((dJdW.average().multiply(dJdW.average())).multiply(1 - opt.beta2));
      M.divide(1 - pow(opt.beta1, iteration + 1));
      V.divide(1 - pow(opt.beta2, iteration + 1));
      Weight = Weight.subtract((M.divide(V.applyFunction(sqrt_float).add(opt.epsilon))).multiply(ll));
      break;
    default:
      break;
  }

  if (!this->init_zero) {
    Bias = Bias.subtract(dJdB.average().multiply(ll));
  }

  return ret;
}
}
