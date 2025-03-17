# DA6401_assignment1

wandb report link :- https://wandb.ai/cs24m018-iitmaana/DA6401%20Assignment1/reports/DA6401-Assignment-1-Report-CS24M018--VmlldzoxMTY5NDQ5Mw?accessToken=1005wqn5z03377uqtbo1j327vo3e2ov4bbox37r9upz8ezeanqk69jdygqtw6qb4

## General Instructions :

if the libraries are not present just run the command:
  pip install <library_name>

## Running the program:
Run the command(Runs in default settings mentioned in table below): 
``` python train.py ```

How to pass arguments:
``` python train.py -e 10 -lr 0.001 -cm 1```

**Arguments supported** :

| Name        | Default Value   | Description  |
| --------------------- |-------------| -----|
| -wp --wandb_project | myprojectname	| Project name used to track experiments in Weights & Biases dashboard |
| -we	--wandb_entity| myname | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| -d --dataset | fashionMNIST  |choices: ["MNIST", "fashionMNIST"]|
|-e, --epochs|10|Number of epochs to train neural network.|
|-b, --batch_size|32|Batch size used to train neural network.|
|-l, --loss|crossEntropyLoss|choices: ["squaredLoss", "crossEntropyLoss"]|
|-o, --optimizer	|adam|choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]|
|-lr, --learning_rate|0.0001|Learning rate used to optimize model parameters|
|-m, --momentum	|0.5|Momentum used by momentum and nag optimizers.|
|-beta, --beta	|0.9|Beta used by rmsprop optimizer|
|-beta1, --beta1|0.9|Beta1 used by adam and nadam optimizers.|
|-beta2, --beta2|0.999|Beta2 used by adam and nadam optimizers.|
|-eps, --epsilon|1e-8|Epsilon used by optimizers.|
|-w_d, --weight_decay|0.0005|	Weight decay/Regularization coefficient used by optimizers.|
|-w_i, --weight_init|xavier|	choices: ["random", "xavier"]|
|-nhl, --num_layers|3|Number of hidden layers used in feedforward neural network.|
|-sz, --hidden_size	|32|	Number of hidden neurons in a feedforward layer.|
|-a, --activation|relu|	choices: ["identity", "sigmoid", "tanh", "relu"]|


## possible optimisation functions are 
- sgd
- momentum based gradient descent
- nesterov accelerated gradient descent
- rmsprop
- adam
- nadam
 
## Possible Activation functions are
- Tanh
- Sigmoid
- Relu
- 
## Possible weight inistilaisers are
- Random
- Xavior

One can define their own no.of hidden layers, batch size, and no.of neaurons per layer, L2 regularisation coefficient(weight decay), epochs, and dataset from "mnist" or "fashion_mnist". 


## How to use:

First create a model object with the class
```
model = feedForwardNeuralNetwork(inputSize=784,hiddenSize=128,hiddenLayerCount=3,outputSize=10,batchSize=32,learningRate=1e-4,initMode = "xavier", optimiser = "adam", activationFunc="relu",weightDecay = 0.0005,lossFunc = "crossEntropyLoss", epochs = 10,dataset="fashionMNIST")
```

and we can call the fit method to run the model
```
model.fit()
```

to get the confusionMatrix:
```
model.confusionMatrix()
```
