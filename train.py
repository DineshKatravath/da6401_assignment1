import argparse
import wandb
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

# Activation Functions

def sigmoid(X):
  X = np.clip(X, -1000, 1000)  # Clipping x to avoid overflow
  return 1./(1+np.exp(-X))

def tanh(X):
  return np.tanh(X)

def relu(X):
  return np.maximum(0,X)

def identity(X):
  return X

def softMax(X):
  # subtracting maxVal for stability
  maxVal = np.max(X,axis=0,keepdims=True)
  X = X - maxVal
  return np.exp(X)/np.sum(np.exp(X),axis=0,keepdims=True)


# Gradient Activation Functions

def sigmoidGrad(X):
  S = sigmoid(X)
  return S * (1 - S)

def tanhGrad(X):
  return 1-tanh(X)**2

def reluGrad(X):
  return np.where(X>0,1,0)

def identityGrad(X):
  return np.ones_like(X)

def softMaxGrad(X):
  S = softMax(X)
  return np.diagflat(S) - np.outer(S, S) # for multiple classes

# Weight Initializations

def randomInit(shape):
  return np.random.randn(*shape)*0.01

def xavierInit(shape):
  inputCount,outputCount = shape
  std = np.sqrt(2/(inputCount+outputCount))
  return np.random.normal(0,std,shape)

class feedForwardNeuralNetwork:

  def __init__(self,inputSize,hiddenSize,hiddenLayerCount,outputSize,epochs,batchSize,optimiser,weightDecay,beta=0.9,lossFunc="crossEntropyLoss",activationFunc="tanh",learningRate=0.001,beta1=0.9,beta2=0.999,dataset="fashionMNIST",isWandb=False,initMode="xavier"):
    # declaring all parameters
    self.inputSize = inputSize
    self.hiddenSize = hiddenSize
    self.hiddenLayerCount = hiddenLayerCount
    self.outputSize = outputSize
    self.epochs = epochs
    self.batchSize = batchSize
    self.weightDecay = weightDecay
    self.beta = beta
    self.beta1 = beta1
    self.beta2 = beta2
    self.lossFunc = lossFunc
    self.activationFunc = activationFunc
    self.learningRate = learningRate
    self.optimiser = optimiser
    self.isWandb = isWandb
    self.initMode = initMode
    self.dataset = dataset


    self.weights = []
    self.biases = []
    self.activationLayer = []
    self.preActivationLayer = []

    # loading the data from dataset into X and y into train,test and val data
    if self.dataset == "fashionMNIST":
      (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
      X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
    elif self.dataset == "MNIST":
      (X_train, y_train), (X_test, y_test) = mnist.load_data()
      X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

    # normalizing and resizing images
    X_train = X_train.flatten().reshape(X_train.shape[0], -1) / 255.0
    X_val = X_val.flatten().reshape(X_val.shape[0], -1) / 255.0
    X_test = X_test.flatten().reshape(X_test.shape[0], -1) / 255.0

    X_train = X_train.T
    X_val = X_val.T
    X_test = X_test.T

    self.X_train = X_train
    self.y_train = y_train
    self.X_val = X_val
    self.y_val = y_val
    self.X_test = X_test
    self.y_test = y_test

  def initialize(self):

    weights = []
    biases = []
    activationLayer = []
    preActivationLayer = []

    #initializing inputWeights
    if self.initMode == 'xavier':
      weights.append(xavierInit((self.hiddenSize,self.inputSize)))
    else:
      weights.append(randomInit((self.hiddenSize,self.inputSize)))
    biases.append(np.zeros((self.hiddenSize,1))) # initializing biases to 0.

    for i in range(1,self.hiddenLayerCount):
      if self.initMode == 'xavier': # initializing random weights
        weights.append(xavierInit((self.hiddenSize,self.hiddenSize)))
      else:
        weights.append(randomInit((self.hiddenSize,self.hiddenSize)))
      biases.append(np.zeros((self.hiddenSize,1))) # initializing biases to 0.

    # Initializing weights from lastHidden layer to output layer
    if self.initMode == 'xavier': # initializing random weights
      weights.append(xavierInit((self.outputSize,self.hiddenSize)))
    else:
      weights.append(randomInit((self.outputSize,self.hiddenSize)))
    biases.append(np.zeros((self.outputSize,1))) # initializing biases to 0.

    # updating parameters
    self.weights = weights
    self.biases = biases
    self.activationLayer = activationLayer
    self.preActivationLayer = preActivationLayer

  # takes a vector and applies activation function on it.
  def activateLayer(self,X):

    type = self.activationFunc
    if type == 'sigmoid':
      activatedLayer = sigmoid(X)
    elif type == 'tanh':
      activatedLayer = tanh(X)
    elif type == 'relu':
      activatedLayer = relu(X)
    elif type == 'softmax':
      activatedLayer = softMax(X)
    elif type == 'identity':
      activatedLayer = identity(X)

    return activatedLayer

  # calculation g'(ai(x))
  def gradActivationLayer(self,X):

    type = self.activationFunc
    if type == 'sigmoid':
      gradActivatedLayer = sigmoidGrad(X)
    elif type == 'tanh':
      gradActivatedLayer = tanhGrad(X)
    elif type == 'relu':
      gradActivatedLayer = reluGrad(X)
    elif type == 'softmax':
      gradActivatedLayer = softMaxGrad(X)
    elif type == 'identity':
      gradActivatedLayer = identityGrad(X)

    return gradActivatedLayer

  # applying softmax on output layer
  def outputActivationLayer(self,X):
    return softMax(X)

  # getting a0 = W0X+B
  def inputLayer(self,X):
    return np.dot(self.weights[0],X)+self.biases[0]

  # getting preactivation layers of next layer from activated layer of curr level and weights
  def hiddenLayer(self,X,layerNo):
    activatedLayer = self.activateLayer(X)
    self.activationLayer.append(activatedLayer)
    return np.dot(self.weights[layerNo],activatedLayer)+self.biases[layerNo]

  # forward propogation
  def forward(self,X):
    self.activationLayer=[]
    self.preActivationLayer=[]
    self.activationLayer.append(X) # h0 is input
    self.preActivationLayer.append(self.inputLayer(X))

    #calculating preActivation layers and activation layers
    for num in range(1,self.hiddenLayerCount+1):
      self.preActivationLayer.append(self.hiddenLayer(self.preActivationLayer[num-1],num))

    # adding softmax to activation layer
    self.output = self.outputActivationLayer(self.preActivationLayer[self.hiddenLayerCount])

    return self.output

  #backpropogation
  def backPropagation(self,X,ypred,y,weights,ai_x,hi_x,loss="crossEntropyLoss",activationType="softmax"):

    m = X.shape[1]
    dw = []
    db = []
    grad_ai = 0
    grad_al = 0
    grad_Wi = 0
    grad_bi = 0

    # gradient of loss w.r.t pre-activation of output layer

    y_one_hot = np.eye(self.outputSize)[y].T

    if loss == "crossEntropyLoss":
      grad_al = -(y_one_hot-ypred)
    elif loss == "squaredLoss":
      grad_al = (ypred-y_one_hot)*ypred*(1-ypred)
      grad_al = grad_al/m

    grad_ai = grad_al

    for k in range(self.hiddenLayerCount,-1,-1):
      # Gradient of loss w.r.t weights of kth layer
      grad_Wi = np.dot(grad_ai, hi_x[k].T)
      dw.append(grad_Wi)

      # Gradient of loss w.r.t biases of kth layer
      grad_bi = np.sum(grad_ai, axis=1, keepdims=True)
      db.append(grad_bi)

      if(k>0):
        grad_h = np.dot(weights[k].T,grad_ai)
        grad_ai = grad_h * self.gradActivationLayer(ai_x[k-1])

    # reversing arrays as we stored from last layer to first
    dw.reverse()
    db.reverse()

    # weight decay (L2 regularization)
    for i in range(len(dw)):
      dw[i] += self.weightDecay*weights[i]

    return dw,db

  # calculating accuracy and loss
  def calcAccuracyandLoss(self,weights,biases,input,labels,activationFunc="softmax"):
    n = input.shape[1]
    loss = 0
    accuracy = 0
    labels_one_hot = np.eye(10)[labels].T

    # calculating output from given input (similar to forward above)
    yPred = input
    for i in range(self.hiddenLayerCount):
      yPred = np.dot(weights[i],yPred)+biases[i]
      yPred = self.activateLayer(yPred)

    yPred = np.dot(weights[self.hiddenLayerCount],yPred)+biases[self.hiddenLayerCount]
    yPred = self.outputActivationLayer(yPred)

    for i in range(self.hiddenLayerCount+1):
      loss+= self.weightDecay*np.linalg.norm(weights[i])

    # calculating accuracy using correct predictions
    # also loss with given lossType
    for i in range(n):
      y_pred = np.argmax(yPred[:,i])
      if labels[i]==y_pred:
        accuracy+=1

      if self.lossFunc == "crossEntropyLoss":
        loss += -np.log(yPred[:,i][labels[i]]+1e-15)
      else:
        loss += np.sum((yPred[:,i]-labels_one_hot[:,i])**2)

    return ((accuracy*100)/n),(loss/n)

  # predicting output probabilities for given input
  def predict(self,weights,biases,input,labels):
    n = input.shape[1]
    loss = 0
    accuracy = 0

    yPred = input
    for i in range(self.hiddenLayerCount):
      yPred = np.dot(weights[i],yPred)+biases[i]
      yPred = self.activateLayer(yPred)

    yPred = np.dot(weights[self.hiddenLayerCount],yPred)+biases[self.hiddenLayerCount]
    yPred = self.outputActivationLayer(yPred)

    predictions = []
    for i in range(n):
      y_pred = np.argmax(yPred[:,i])
      predictions.append(y_pred)
    return predictions

  # Stochastic Gradient Descent (sgd) optimiser
  def sgd(self):
    # initializing weights,biases,activation and preactivation lists
    self.initialize()

    it = 0
    while it < self.epochs:
      it += 1

      i = 0
      while i<self.X_train.shape[1]:
        batch_end = min(i + self.batchSize, self.X_train.shape[1])

        # forward and backward passes
        ypred = self.forward(self.X_train[:,i:batch_end])
        dw,db = self.backPropagation(self.X_train[:,i:batch_end],ypred,self.y_train[i:batch_end],self.weights,
                                     self.preActivationLayer,self.activationLayer,self.lossFunc,self.activationFunc)

        # updating parameters
        for k in range(len(dw)):
          self.weights[k] -= self.learningRate*dw[k]
          self.biases[k] -= self.learningRate*db[k]

        i += self.batchSize
      accuracy, loss = self.calcAccuracyandLoss(self.weights,self.biases,self.X_train,self.y_train,self.activationFunc)
      validationAccuracy , validationLoss = self.calcAccuracyandLoss(self.weights,self.biases,self.X_val,self.y_val,self.activationFunc)

      print("Validation Accuracy: ",validationAccuracy)
      print("Validation Loss: ",validationLoss)
      print("Training Accuracy: ",accuracy)
      print("Training Loss: ",loss)

      # logging specs to wandb
      if self.isWandb:
        wandb.log({"Training Accuracy": accuracy})
        wandb.log({"Training Loss": loss})
        wandb.log({"Validation Accuracy": validationAccuracy})
        wandb.log({"Validation Loss": validationLoss})
        wandb.log({"Epoch": it})
        wandb.log({"Learning Rate": self.learningRate})
        wandb.log({"Batch Size": self.batchSize})
        wandb.log({"Optimiser": self.optimiser})

  def momentumBasedGradientDescent(self):
    # initializing weights,biases,activation and preactivation lists
    self.initialize()

    # Properly initialize momentum
    uw = [np.zeros_like(w) for w in self.weights]
    ub = [np.zeros_like(b) for b in self.biases]

    it = 0
    while it < self.epochs:
      it += 1

      i = 0
      while i < self.X_train.shape[1]:
        batch_end = min(i + self.batchSize, self.X_train.shape[1])

        # forward and backward passes
        ypred = self.forward(self.X_train[:,i:batch_end])
        dw,db = self.backPropagation(self.X_train[:,i:batch_end],ypred,self.y_train[i:batch_end],self.weights,
                                     self.preActivationLayer,self.activationLayer,self.lossFunc,self.activationFunc)

        # Update momentum and weights for each layer
        for k in range(len(dw)):
          uw[k] = self.beta * uw[k] + dw[k]
          ub[k] = self.beta * ub[k] + db[k]

        # upadting weights and biases
        for k in range(len(dw)):
          self.weights[k] -= self.learningRate * uw[k]
          self.biases[k] -= self.learningRate * ub[k]

        i += self.batchSize
      accuracy, loss = self.calcAccuracyandLoss(self.weights,self.biases,self.X_train,self.y_train,self.activationFunc)
      validationAccuracy , validationLoss = self.calcAccuracyandLoss(self.weights,self.biases,self.X_val,self.y_val,self.activationFunc)

      print("Validation Accuracy: ",validationAccuracy)
      print("Validation Loss: ",validationLoss)
      print("Training Accuracy: ",accuracy)
      print("Training Loss: ",loss)

      # logging specs to wandb
      if self.isWandb:
        wandb.log({"Training Accuracy": accuracy})
        wandb.log({"Training Loss": loss})
        wandb.log({"Validation Accuracy": validationAccuracy})
        wandb.log({"Validation Loss": validationLoss})
        wandb.log({"Epoch": it})
        wandb.log({"Learning Rate": self.learningRate})
        wandb.log({"Batch Size": self.batchSize})
        wandb.log({"Optimiser": self.optimiser})

  def nesterovAcceleratedBasedGradientDescent(self):
    # initializing weights,biases,activation and preactivation lists
    self.initialize()

    # Properly initialize momentum
    uw = [np.zeros_like(w) for w in self.weights]
    ub = [np.zeros_like(b) for b in self.biases]

    it=0
    while it<self.epochs:
      it+=1

      i=0
      while i<self.X_train.shape[1]:
        batch_end = min(i + self.batchSize, self.X_train.shape[1])

        # forward and backward passes
        ypred = self.forward(self.X_train[:,i:batch_end])
        dw,db = self.backPropagation(self.X_train[:,i:batch_end],ypred,self.y_train[i:batch_end],self.weights,
                                     self.preActivationLayer,self.activationLayer,self.lossFunc,self.activationFunc)

        # updating momentum
        for k in range(len(dw)):
          uw[k] = self.beta*uw[k] + dw[k]
          ub[k] = self.beta*ub[k] + db[k]

        for k in range(len(dw)):
          self.weights[k] -= self.learningRate*(self.beta*uw[k] + dw[k])
          self.biases[k] -= self.learningRate*(self.beta*ub[k] + db[k])

        i += self.batchSize
      accuracy, loss = self.calcAccuracyandLoss(self.weights,self.biases,self.X_train,self.y_train,self.activationFunc)
      validationAccuracy , validationLoss = self.calcAccuracyandLoss(self.weights,self.biases,self.X_val,self.y_val,self.activationFunc)

      print("Validation Accuracy: ",validationAccuracy)
      print("Validation Loss: ",validationLoss)
      print("Training Accuracy: ",accuracy)
      print("Training Loss: ",loss)

      # logging specs to wandb
      if self.isWandb:
        wandb.log({"Training Accuracy": accuracy})
        wandb.log({"Training Loss": loss})
        wandb.log({"Validation Accuracy": validationAccuracy})
        wandb.log({"Validation Loss": validationLoss})
        wandb.log({"Epoch": it})
        wandb.log({"Learning Rate": self.learningRate})
        wandb.log({"Batch Size": self.batchSize})
        wandb.log({"Optimiser": self.optimiser})

  def rmsProp(self):
    # initializing weights,biases,activation and preactivation lists
    self.initialize()

    v_w = [np.zeros_like(w) for w in self.weights]
    v_b = [np.zeros_like(b) for b in self.biases]

    eps = 1e-8
    it = 0

    while it < self.epochs:
      it += 1

      i = 0
      while i < self.X_train.shape[1]:
          batch_end = min(i + self.batchSize, self.X_train.shape[1])
          X_batch = self.X_train[:, i:batch_end]
          y_batch = self.y_train[i:batch_end]

          # Forward pass
          ypred = self.forward(X_batch)

          # Backward pass
          dw, db = self.backPropagation(X_batch, ypred, y_batch, self.weights,
                                      self.preActivationLayer, self.activationLayer,self.lossFunc, self.activationFunc)

          # Update velocity and weights for each layer
          for k in range(len(dw)):
              v_w[k] = self.beta * v_w[k] + (1 - self.beta) * (dw[k]**2)
              v_b[k] = self.beta * v_b[k] + (1 - self.beta) * (db[k]**2)

              self.weights[k] -= (self.learningRate / (np.sqrt(v_w[k]) + eps)) * dw[k]
              self.biases[k] -= (self.learningRate / (np.sqrt(v_b[k]) + eps)) * db[k]
          i += self.batchSize
      accuracy, loss = self.calcAccuracyandLoss(self.weights,self.biases,self.X_train,self.y_train,self.activationFunc)
      validationAccuracy , validationLoss = self.calcAccuracyandLoss(self.weights,self.biases,self.X_val,self.y_val,self.activationFunc)

      print("Validation Accuracy: ",validationAccuracy)
      print("Validation Loss: ",validationLoss)
      print("Training Accuracy: ",accuracy)
      print("Training Loss: ",loss)

      # logging specs to wandb
      if self.isWandb:
        wandb.log({"Training Accuracy": accuracy})
        wandb.log({"Training Loss": loss})
        wandb.log({"Validation Accuracy": validationAccuracy})
        wandb.log({"Validation Loss": validationLoss})
        wandb.log({"Epoch": it})
        wandb.log({"Learning Rate": self.learningRate})
        wandb.log({"Batch Size": self.batchSize})
        wandb.log({"Optimiser": self.optimiser})

  def adam(self):
    # initializing weights,biases,activation and preactivation lists
    self.initialize()

    v_w = [np.zeros_like(w) for w in self.weights]
    v_b = [np.zeros_like(b) for b in self.biases]
    m_w = [np.zeros_like(w) for w in self.weights]
    m_b = [np.zeros_like(b) for b in self.biases]

    eps = 1e-8
    it = 0
    t = 0

    while it < self.epochs:
      it += 1
      i = 0

      while i < self.X_train.shape[1]:
        batch_end = min(i + self.batchSize, self.X_train.shape[1])
        X_batch = self.X_train[:, i:batch_end]
        y_batch = self.y_train[i:batch_end]

        # Forward pass
        ypred = self.forward(X_batch)

        # Backward pass
        dw, db = self.backPropagation(X_batch, ypred, y_batch, self.weights,
                                    self.preActivationLayer, self.activationLayer, self.lossFunc, self.activationFunc)

        t += 1

        # Update momentum for each layer
        for k in range(len(dw)):
          m_w[k] = self.beta1 * m_w[k] + (1 - self.beta1) * dw[k]
          m_b[k] = self.beta1 * m_b[k] + (1 - self.beta1) * db[k]

          v_w[k] = self.beta2 * v_w[k] + (1 - self.beta2) * (dw[k]**2)
          v_b[k] = self.beta2 * v_b[k] + (1 - self.beta2) * (db[k]**2)

          # Bias correction
          mhat_w = m_w[k] / (1 - self.beta1**t)
          mhat_b = m_b[k] / (1 - self.beta1**t)

          vhat_w = v_w[k] / (1 - self.beta2**t)
          vhat_b = v_b[k] / (1 - self.beta2**t)

          self.weights[k] -= (self.learningRate / (np.sqrt(vhat_w) + eps)) * mhat_w
          self.biases[k] -= (self.learningRate / (np.sqrt(vhat_b) + eps)) * mhat_b
        i += self.batchSize
      accuracy, loss = self.calcAccuracyandLoss(self.weights,self.biases,self.X_train,self.y_train,self.activationFunc)
      validationAccuracy , validationLoss = self.calcAccuracyandLoss(self.weights,self.biases,self.X_val,self.y_val,self.activationFunc)

      print("Validation Accuracy: ",validationAccuracy)
      print("Validation Loss: ",validationLoss)
      print("Training Accuracy: ",accuracy)
      print("Training Loss: ",loss)

      # logging specs to wandb
      if self.isWandb:
        wandb.log({"Training Accuracy": accuracy})
        wandb.log({"Training Loss": loss})
        wandb.log({"Validation Accuracy": validationAccuracy})
        wandb.log({"Validation Loss": validationLoss})
        wandb.log({"Epoch": it})
        wandb.log({"Learning Rate": self.learningRate})
        wandb.log({"Batch Size": self.batchSize})
        wandb.log({"Optimiser": self.optimiser})

  def nadam(self):
    # initializing weights,biases,activation and preactivation lists
    self.initialize()

    v_w = [np.zeros_like(w) for w in self.weights]
    v_b = [np.zeros_like(b) for b in self.biases]
    m_w = [np.zeros_like(w) for w in self.weights]
    m_b = [np.zeros_like(b) for b in self.biases]

    eps = 1e-8
    it = 0
    t = 0

    while it < self.epochs:
      it += 1
      i = 0

      while i < self.X_train.shape[1]:
        batch_end = min(i + self.batchSize, self.X_train.shape[1])
        X_batch = self.X_train[:, i:batch_end]
        y_batch = self.y_train[i:batch_end]

        # Forward pass
        ypred = self.forward(X_batch)

        # Backward pass
        dw, db = self.backPropagation(X_batch, ypred, y_batch, self.weights,
                                    self.preActivationLayer, self.activationLayer, self.lossFunc, self.activationFunc)

        t += 1

        # Update momentum for each layer
        for k in range(len(dw)):
          m_w[k] = self.beta1 * m_w[k] + (1 - self.beta1) * dw[k]
          m_b[k] = self.beta1 * m_b[k] + (1 - self.beta1) * db[k]

          v_w[k] = self.beta2 * v_w[k] + (1 - self.beta2) * (dw[k]**2)
          v_b[k] = self.beta2 * v_b[k] + (1 - self.beta2) * (db[k]**2)

          # Bias correction
          mhat_w = m_w[k] / (1 - self.beta1**t)
          mhat_b = m_b[k] / (1 - self.beta1**t)

          vhat_w = v_w[k] / (1 - self.beta2**t)
          vhat_b = v_b[k] / (1 - self.beta2**t)

          self.weights[k] -= (self.learningRate / (np.sqrt(vhat_w) + eps)) * (self.beta1 * mhat_w + (1 - self.beta1) * dw[k] / (1 - self.beta1**t))
          self.biases[k] -= (self.learningRate / (np.sqrt(vhat_b) + eps)) * (self.beta1 * mhat_b + (1 - self.beta1) * db[k] / (1 - self.beta1**t))
        i += self.batchSize
      accuracy, loss = self.calcAccuracyandLoss(self.weights,self.biases,self.X_train,self.y_train,self.activationFunc)
      validationAccuracy , validationLoss = self.calcAccuracyandLoss(self.weights,self.biases,self.X_val,self.y_val,self.activationFunc)

      print("Validation Accuracy: ",validationAccuracy)
      print("Validation Loss: ",validationLoss)
      print("Training Accuracy: ",accuracy)
      print("Training Loss: ",loss)

      # logging specs to wandb
      if self.isWandb:
        wandb.log({"Training Accuracy": accuracy})
        wandb.log({"Training Loss": loss})
        wandb.log({"Validation Accuracy": validationAccuracy})
        wandb.log({"Validation Loss": validationLoss})
        wandb.log({"Epoch": it})
        wandb.log({"Learning Rate": self.learningRate})
        wandb.log({"Batch Size": self.batchSize})
        wandb.log({"Optimiser": self.optimiser})

  def fit(self):
    if self.optimiser == "sgd":
      self.sgd()
    elif self.optimiser == "momentum":
      self.momentumBasedGradientDescent()
    elif self.optimiser == "nag":
      self.nesterovAcceleratedBasedGradientDescent()
    elif self.optimiser == "rmsprop":
      self.rmsProp()
    elif self.optimiser == "adam":
      self.adam()
    elif self.optimiser == "nadam":
      self.nadam()

  def confusionMatrix(self):
    predictions = self.predict(self.weights,self.biases,self.X_test,self.y_test)

    testAccuracy , testLoss = self.calcAccuracyandLoss(self.weights,self.biases,self.X_test,self.y_test,self.activationFunc)
    print("\n")
    print("Test Accuracy: ",testAccuracy)
    print("Test Loss: ",testLoss)

    if self.dataset == "fashion_mnist":
      classLabels = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    else:
      classLabels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
    confusionMatrix = confusion_matrix(self.y_test, predictions)
    plt.figure(figsize=(15,10))
    # plotting confusion matrix
    sns.heatmap(confusionMatrix, annot=True, fmt='d', cmap='Blues', xticklabels= classLabels, yticklabels= classLabels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    # Save the plot to an image file
    heatmapFile = "confusionMatrix_heatmap.png"
    plt.savefig(heatmapFile,bbox_inches='tight')

    plt.show()
    
    if self.isWandb == True:
      # Logging image to Wandb
      wandb.log({"confusionMatrix": wandb.Image(heatmapFile)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_entity", "-we",help = "Wandb Entity used to track experiments in the Weights & Biases dashboard.", default="cs24m018")
    parser.add_argument("--wandb_project", "-wp",help="Project name used to track experiments in Weights & Biases dashboard", default="DA6401 Assignment1")
    parser.add_argument("--dataset", "-d", help = "dataset", choices=["MNIST","fashionMNIST"], default="fashionMNIST")
    parser.add_argument("--epochs","-e", help= "Number of epochs to train neural network", type= int, default=10)
    parser.add_argument("--batch_size","-b",help="Batch size used to train neural network", type =int, default=32)
    parser.add_argument("--optimizer","-o",help="batch size is used to train neural network", default= "adam", choices=["sgd","momentum","nag","rmsprop","adam","nadam"])
    parser.add_argument("--loss","-l", default= "crossEntropyLoss", choices=["squaredLoss", "crossEntropyLoss"])
    parser.add_argument("--learning_rate","-lr", default=0.0001, type=float)
    parser.add_argument("--momentum","-m", default=0.5,type=float)
    parser.add_argument("--beta","-beta", default=0.9, type=float)
    parser.add_argument("--beta1","-beta1", default=0.9,type=float)
    parser.add_argument("--beta2","-beta2", default=0.999,type=float)
    parser.add_argument("--epsilon","-eps",type=float, default = 1e-8)
    parser.add_argument("--weight_decay","-w_d", default=0.0005,type=float)
    parser.add_argument("-w","--weight_init", default="xavier",choices=["random","xavier"])
    parser.add_argument("--num_layers","-nhl",type=int, default=3)
    parser.add_argument("--hidden_size","-sz",type=int, default=128)
    parser.add_argument("-a","--activation",choices=["identity","sigmoid","tanh","relu"], default="relu")
    
    
    args = parser.parse_args()

    print(args.dataset)
    print(args.epochs)
    print(args.batch_size)
    print(args.optimizer)
    print(args.loss)
    print(args.learning_rate)
    print(args.momentum)
    print(args.beta)
    print(args.beta1)
    print(args.beta2)
    print(args.epsilon)
    print(args.weight_decay)
    print(args.weight_init)
    print(args.num_layers)
    print(args.hidden_size)
    print(args.activation)

    wandb.login()
    wandb.init(project=args.wandb_project,entity=args.wandb_entity)
    model = feedForwardNeuralNetwork(inputSize=784,hiddenSize=args.hidden_size,hiddenLayerCount=args.num_layers,outputSize=10,
                                      epochs=10,batchSize=args.batch_size,optimiser=args.optimizer,weightDecay=args.weight_decay,
                                      beta=args.beta,lossFunc=args.loss,activationFunc=args.activation,learningRate=args.learning_rate,beta1=args.beta1,beta2=args.beta2,
                                      dataset=args.dataset,isWandb=True,initMode=args.weight_init)
    model.fit()
    wandb.finish()


