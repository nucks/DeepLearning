# Week 4 - Deep Neural Networks

## Deep L-layer neural network
* You can always start out with shallow neural networks and try getting deeper to see if that helps solve the problem.
* **L** - used to denote the number of layers used within a network
* **n<sup>[l]</sup>** - used to denote the number of nodes in layer l
* **a<sup>[l]</sup>** - used to denote the number of activations in layer l
* a<sup>[L]</sup> = y&#770;

## Forward Propagation in a Deep Network
* The vectorized forward propagation equations look like this:<br/>
<code>Z<sup>[l]</sup> = W<sup>[l]</sup>A<sup>l-1</sup> + b<sup>[l]</sup></code><br/>
<code>A<sup>[l]</sup> = g<sup>[l]</sup>(Z<sup>l</sup>)</code><br/>
* The explicit for loop that goes outside these equations has not been able to be replaced. So that outer for loop will not be vectorized.

## Getting your matrix dimensions right
* The way to check if your dimensions are correct is to write it down on paper.
* <code>w<sup>[l]</sup> = (n<sup>[l]</sup>, n<sup>[l-1]</sup>)</code>
* <code>b<sup>[l]</sup> = (n<sup>[l]</sup>, 1)</code>
    * For the bias (b) to be added to the weights (w)* the activations (a), it will need to have the same parameters as the w vector.
* dw and db should have the same dimensions as w and b, respectively.
* <code>z<sup>[l]</sup>, a<sup>[l]</sup> = (n<sup>[l]</sup>, 1)</code>
* <code>Z<sup>[l]</sup>, A<sup>[l]</sup> = (n<sup>[l]</sup>, m)</code>
    * dZ and dA will also have the same dimensions as Z and A.

## Why deep representations?
* The more layers that you have, the more complex the functions that the computer can learn.
* You can think of the first layers as computing simple pieces like the edges of a picture, while the later layers will begin to compose the pieces and can learn more complex functions.
    * The larger layers also may be looking at larger pieces in comparison to the small blocks that would be analyzed in the first or second layer.
* An example with audio
    * Layer 1 - low level, waves
    * Layer 2 - Phonemes
    * Layer 3 - words
    * Layer 4 - sentences, phrases
* Some scientists believe that the human brain works in a similar pattern. They think that we first begin to detect edges and then later on, we are able to see beyond the first things we noticed.
    * Some analogies can be dangerous, but it is possible that we think this way.
* **Circuit theory** - There are functions you can compute with a “small” L-layer deep neural network that shallower networks require exponentially more hidden units to compute.
    * In other words, there are mathematical functions that are much easier to compute with deep networks instead of shallow networks with a large amount of hidden units.
* History lesson
    * Neural networks were rebranded to be called “deep learning” to catch the attention of public eye. This wasn’t necessarily the only reason, but it did play a part in it.
* Generally, it’s best to start shallow and try adding on more layers as needed to find the correct amount of depth.

## Building blocks of deep neural networks
* The entire flow of neural networks is illustrated in the picture below. It first shows the forward propagation step and the parameters that it requires, as well as the backward propagation step.
* You will also notice that we cache a value between the two steps so that we can access the variables from the forward propagation step in the backward step too.<br/>
￼
<img src=“../images/week4/neuralNetworkFlow.png” alt=“Entire neural network flow” width=“50%”></img>

## Parameters vs. Hyperparameters
* **Hyperparameters** - parameters that control the parameters that are being passed into the function (w, b)
    * Examples of hyperparameters: learning rate, iterations, hidden layers, hidden units, etc.
* Hyperparameters are usually just found by trying different things out and see what works.

## What does this have to do with the brain?
* It doesn’t have a whole lot to do with the brain, in reality. The analogy is being used much less in the field nowadays.
* There is a simplistic analogy that says the brains neurons send data to other neurons, but we know so little about even single neurons in the brain that it’s hard to compare. We don’t know how neurons in the human brain learn.

### Additional Resources
* [Implementing a neural network from scratch](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/)
* [StackExchange: Why do we normalize images?](https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c)
