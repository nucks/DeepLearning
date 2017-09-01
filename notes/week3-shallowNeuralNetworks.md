# Week 3 - Shallow Neural Networks

## Neural Network Overview
* In logistic regression, we found z and then used that to find the activation. Then, we used the activation to calculate the loss.
￼<br/>
<img src="../images/week3/logisticRegression.png" alt="Logistic Regression" width="50%"></img>
* In a neural network, we do the same thing but we do it multiple times. 
    * **Layer** - Each iteration of finding z and a. Layers are denoted by square brackets like so:  a<sup>[1]</sup>. 
    * After finding a and z a certain amount of times, we will then calculate the loss like in logistic regression.
    * Backward propagation is still used to find the derivatives for the network.

## Neural Network Representation
* What does a neural network look like?
￼￼<br/>
<img src="../images/week3/neuralNetwork.png" alt="Neural Network" width="50%"></img>

* **Hidden layer** - These are values that are not seen within the training set. You see both the input and the output values, but the hidden layer values are not shown.
* Previously the input layer was referenced as x, but it can also be referenced as a<sup>[0]</sup>. Along those lines, the hidden layer becomes a<sup>[1]</sup>, and the output layer becomes a<sup>[2]</sup>.
* The neural network shown above is counted as 2-layer NN because the input layer is not counted as an actual layer.
* Both the hidden layer and outer layer have parameters associated with them.
    * Hidden layer parameters
        * w<sup>[1]</sup>
            * It’s in the shape of a (4, 3) vector.
            * The 4 comes from the fact that there are 4 nodes associated with the layer. The 3 comes from the 3 input layers.
        * b<sup>[1]</sup>
            * It’s in the shape of a (4, 1) matrix.
    * Output layer parameters
        * w<sup>[2]</sup>
            * It’s in the shape of a (1, 4) matrix.
            * The 4 here comes from the hidden layer’s 4 hidden units. The 1 comes from the single output unit.
        * b<sup>[2]</sup>
            * It’s in the shape of a (1, 1) matrix.

## Computing a Neural Network’s Output
* Now, let’s dive into what each node would look like in the hidden layer. As I wrote above, each node is used to calculate z and a.
￼￼<br/>
<img src="../images/week3/singleNode.png" alt="Single Node with Calculations" width="30%"></img>
* Example notation that illustrates what is happening looks like this: a<sub>2</sub><sup>[1]</sup>
    * This annotates that we are looking at the second node in the first layer, or the hidden layer.
    * Subscript - the node number in the layer
    * Superscript - the layer number that you are working with
* Based on this, the z and a equations would both look a little different.
    * z<sub>2</sub><sup>[1]</sup> = w<sub>2</sub><sup>[1]T</sup>x + b<sub>2</sub><sup>[1]</sup>
* So, given x, you can solve this with four lines of code:<br/>
<code>z<sup>[1]</sup> = W<sup>[1]</sup>x + b<sup>[1]</sup></code><br/>
<code>a<sup>[1]</sup> = sigma(z<sup>[1]</sup>)</code><br/>
<code>z<sup>[2]</sup> = W<sup>[2]</sup>a<sup>[1]</sup> + b<sup>[2]</sup></code><br/>
<code>a<sup>[2]</sup> = sigma(z<sup>[2]</sup>)</code><br/>

## Vectorizing Across Multiple Examples
* To vectorize across multiple training sets, we need to do the same thing for all the different sets.
* So if we have a set that is x<sup>1</sup>-x<sup>m</sup>, then we are going to generate a<sup>[2](i)</sup> as a solution. The first exponent tells us that we are in layer 2, and the second refers to which training set we are dealing with.
* Those four lines of code change a little.<br/>
<code>for i = 1 to m:</code><br/>
	<code>  z<sup>[1](i)</sup> = W<sup>[I]</sup>x<sup>(i)</sup> + b<sup>[1]</sup></code><br/>
	<code>  a<sup>[1](i)</sup> = sigmoid(z<sup>[1](i)</sup>)</code><br/>
	<code>  z<sup>[2](i)</sup> = W<sup>[2]</sup>a<sup>[1](i)</sup> + b<sup>[2]</sup></code><br/>
	<code>  a<sup>[2](i)</sup> = sigmoid(z<sup>[2](i)</sup>)</code><br/>

* The solution above is not complete, however, because we still want to completely vectorize it. The way to do that is by using the capital version of the equation. ie. X = [x<sup>1</sup>, x<sup>2</sup>, x<sup>m</sup>]. Then you will have the full implementation.<br/>
<code>Z<sup>[1]</sup> = W<sup>[1]</sup>X + b<sup>[1]</sup></code><br/>
<code>A<sup>[1]</sup> = sigmoid(Z<sup>[1]</sup>)</code><br/>
<code>Z<sup>[2]</sup> = W<sup>[2]</sup>A<sup>[1]</sup> + b<sup>[2]</sup></code><br/>
<code>A<sup>[2]</sup> = sigmoid(Z<sup>[2]</sup>)</code><br/>

## Activation Functions
* Up to this point we have only used the sigmoid function as an activation function but there are others that work better.
* The tanh function is an example of an activation function that almost always performs better than the sigmoid function. 
    * `a = tanh(z)`
        * It goes between 1 and -1 on the graph. The sigmoid function went between 1 and 0.
        * The equation: e<sup>z</sup> - e<sup>-z</sup> / e<sup>z</sup> + e<sup>-z</sup>.
        * This almost always works better than the sigmoid function because it “centers” your data around 0. This mean around 0 makes it easier for the computer to learn.
        * The one exception to this rule is when you are using binary classification because you actually want data that is between 0 and 1. So you would use the sigmoid activation function at the output layer.
* One other function that is popular is the ReLU function.
    * `a = max(0, z)`
        * When it’s positive, the derivative is 1. When it’s negative, the derivative is 0.
* **Leaky ReLU** - slight angle where the line is normally straight in the ReLU function. It’s usually more accurate than the ReLU function but it isn’t used in practice as much.
    * `a = max(0.01z, z)`
* Summary
    * If your output is a 0 or 1, the sigmoid function is good for the activation layer of the output.
    * The ReLU function is the favorite default as an activation function so if you aren’t sure which you should use, use it.
    * ReLU will be learned faster because the slope is much further away from zero, unlike other activation functions.

## Why do you need non-linear activation functions?
* **Linear activation function** or **identity activation function** - `g(z) = z`
    * What would happen if we just got rid of the g in the activation functions?
        * If you do this, it will just calculate the linear activation function of the inputs.
        * A linear hidden layer is more or less useless because the computation is basically the same as logistic regression.
    * The only time you may want to use linear activation functions is if you are trying to predict real numbers (say the price of homes) and your output needs to be a real number. You would use that activation function on the output layer.
        * The hidden layer will still need to use a different activation function.
        * You could still use a ReLU function to find the output in the real estate case.

## Derivatives of Activation Functions
* When doing back propagation, you need to be able to compute the derivative of the activation function.
* Sigmoid activation function
    * `d/dz*g(z)` which is otherwise known as g’<sup>z</sup> = a(1-a)
    * The plus for the g’ function is that if you have already calculated the a then you can very easily find out what the derivative of the function is.
* Tanh activation function
    * g’<sup>z</sup> = 1 - (tanh(z))<sup>2<sup>
* ReLU
    * g’<sup>z</sup> = 0 if z<0, 1 if z>=0
* Leaky ReLU
    * g’<sup>z</sup> = 0.01 if z<0, 1 if z>=0

## Gradient Descent for Neural Networks
* Parameters: w<sup>[1]</sup>, b<sup>[1]</sup>, w<sup>[2]</sup>, b<sup>[2]</sup>
* Cost function: J(W<sup>[1]</sup>, b<sup>[1]</sup>, W<sup>[2]</sup>, b<sup>[2]</sup>) = 1/m * np.sum(y&#770;, y). 
    * y&#770; also means a<sup>[2]</sup>

* Forward Propagation Step<br/>
<code>Z<sup>[1]</sup> = W<sup>[1]</sup>X + b<sup>[1]</sup></code><br/>
<code>A<sup>[1]</sup> = g<sup>[1]</sup>(Z<sup>[1]</sup>)</code><br/>
<code>Z<sup>[2]</sup> = W<sup>[2]</sup>A<sup>[1]</sup> + b<sup>[2]</sup></code><br/>
<code>A<sup>[2]</sup> = g<sup>[2]</sup>(Z<sup>[2]</sup>) = sigmoid(Z<sup>[2]</sup>)</code><br/>
* Backward Propagation Step<br/>
<code>dZ<sup>[2]</sup> = A<sup>[2]</sup> - Y</code><br/>
<code>dW<sup>[2]</sup> = 1/m * dZ<sup>[2]</sup>A<sup>[2]T</sup></code><br/>
<code>db<sup>[2]</sup> = 1/m * np.sum(dZ<sup>[2]</sup>, axis = 1, keepdims = True)</code><br/>
<code>dZ<sup>[1]</sup> = W<sup>[2]T</sup>dZ<sup>[2]</sup> * g<sup>[1]’</sup>(Z<sup>[1]</sup>)</code><br/>
<code>dW<sup>[1]</sup> = 1/m * dZ<sup>[1]</sup>X<sup>T</sup></code><br/>
<code>db<sup>[1]</sup> = a/m * np.sum(dZ<sup>[1]</sup>, axis = 1, keepdims = True)</code><br/>

## Random Initialization
* In neural networks, it is important that you initialize weights as random numbers.
* If all the hidden layer nodes are initialized to zero then they will be symmetrical and they will never differ from one another. In other words, it would become pointless to have more than one node.<br/>
<code>w<sup>[1]</sup> = np.random.randn(2, 2) * 0.01</code><br/>
<code>b<sup>[1]</sup> = np.zeros((2, 1))</code><br/>
<code>w<sup>[2]</sup> = np.random.randn(2, 2) * 0.01</code><br/>
<code>b<sup>[2]</sup> = np.zeros((2, 1))</code><br/>
* When training with more hidden layers, and having more of a deep neural network, you oftentimes won’t want to use the constant 0.01. For one hidden layer, it is fine.

### Additional Resources
* http://scs.ryerson.ca/~aharley/neural-networks/
* https://cs231n.github.io/neural-networks-case-study/