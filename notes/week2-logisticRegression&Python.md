# Week 2 - Logistic Regression as a Neural Network

* Quick Tips
    * You always want to process training sets without explicit for loops because they are too expensive.
    * In logistic regression, there is a forward propagation step followed by a backward propagation step.

##  Binary Classification
### Images
* Let's say that you have an image that is 64px x 64px. To represent that image in a computer, it will be represented as 3 separate matrices (red matrix, blue matrix, green matrix) and each one will be 64 x 64.
* This picture represents y. You will either get a 0 or a 1 as a result that will tell you whether or not the image is what you expect.

￼￼![Binary Classification](/images/week2/binaryClassification.png)
* To find x, we need to use those values to create a **feature vector**.

<img src="/images/week2/imageVector.png" alt="Image Vector" width="50%"></img>
￼
### Notation
* (x,y) = x<sup>nx</sup>, y<sup>0 or 1</sup>
* m<sub>training</sub> examples = {(x<sup>1</sup>, y<sup>1</sup>), (x<sup>2</sup>, y<sup>2</sup>), (x<sup>3</sup>, y<sup>3</sup>), (x<sup>4</sup>, y<sup>4</sup>), (x<sup>5</sup>, y<sup>5</sup>)}
* **m<sub>training</sub> examples** - the number of training examples
* **m<sub>test</sub>** - the number of test sets
X = [x<sup>1</sup>, x<sup>2</sup>, x<sup>3</sup>…x<sup>m</sup>] 
* This matrix will have m columns and nx rows
* X = <sup>nx x m</sup> matrix
* `X.shape` = (nx, m). This will give you the shape of a matrix in python.
* Y = [y<sup>1</sup>, y<sup>2</sup>, y<sup>3</sup>…y<sup>m</sup>]
* Y = <sup>1x x m</sup> matrix
* `Y.shape` = (1, m). This is a 1 x m matrix.

##  Logistic Regression Model
Given x, we want y&#770; to equl the probability that y = 1, given x.
* X<sup>nx</sup> with parameters w<sup>nx</sup>, b (which is a real number)
* y&#770; = w transpose x + b (linear function of the input x) 
    * This is good for linear regression.
    * The problem with this is that it is hard to enforce that 0 <= y <= 1. It can be negative, or much bigger.
* So, the solution is y&#770; = sigmoid(w transpose x + b)
<img src="/images/week2/logisticRegressionModel.png" alt="Logistic Regression Model" width="50%" />
* We use z to replace the w transpose x + b
* So, sigmoid(z) = 1/1+e<sup>-z</sup>
    * If z is large then it will equal something very close to 1.
    * If z is very small (large negative number) then sigmoid(z) will be close to 0.

##  Logistic Regression Cost Function
* Squared error =  1/2(y&#770; - y)<sup>2</sup> is one way to find the loss, but it doesn’t work well with gradient descent so we don’t want to do that.
* Loss(y&#770;, y) = -(y log y&#770; + (1-y)log(1-y&#770;)) 
* We want the loss function to be as small as possible
    * If y = 1, want y&#770; to be large
    * If y = 0, want y&#770; to be small 
* The loss function works to see how a single training example is doing.
* The cost function (how you are doing on the entire training set)
    * `J(w, b) = 1/m` of the sum of the loss function (y&#770;<sup>i</sup>, y<sup>i</sup>) = 
￼![Cost Function](/images/week2/costFunction.png)
**Loss function** - works for a single training example
**Cost function** - Applied to parameters of the algorithm and works for entire training set

##  Gradient Descent
* We can use gradient descent to learn the parameters of the training set. We want to find w, b that minimize J(w, b)
* Convex functions look like big bowls (in contrast to functions that have lines that go up and down)
* You can initialize it anywhere, but you will always end up at the same point.
    * You always take a step in the steepest downhill direction.
    * **Global optimum** - The lowest point in the bowl where it is most optimized.
* The derivative term dJ(w)/dw is usually represented in code as “dw” and the equation representing the b is “db”
￼![Gradient Descent](/images/week2/gradientDescent.png)
* The actual equations to update each of the parameters (w and b)
<img src="/images/week2/actualEquations.png" alt="The actual equations to update parameters" width="50%" />
**Partial derivative symbol** - ∂ (lowercase d in fancy font that is used to describe derivative). This symbol will be shown in place of a lowercase d if there is more than one parameter. This is a rule of calculus.

## Derivatives
* **Derivative** - A fancy term for the slope of a line (height/width or rise/run). Slope and derivative can be used interchangeably.
    * Df(a)/da or d/da(f(a))
    * As you move up a line with this formula, the slope will remain the same.
* If you have f(a) = a<sup>2</sup> then the slope will change as you move on the line.
* `D/da(a^2) = 2a`
    * If you nudge up fa at some point, then you can expect the derivate to move up 2a. This will tell you exactly how much you can expect fa (f of a) to go up. 4 times as much.
* D/da(a<sup>3</sup>) = 3a<sup>2</sup>
    * If A = 2, derivative = 8. If you check this 3a<sup>2</sup> = 12, meaning that it will be 12 times as much.
* Log<sub>e</sub><sup>a</sup> = d/da(f(a)) = 1/a
    * If a = 2, derivative = 0.69315. so we’d expect it to go up by 1/2
* Derivative of functions can usually be found in textbooks—so you can always look them up.

## Computation Graph
* The computation graph helps to explain why a backward and forward propagation steps are necessary.
￼![Computation Graph](/images/week2/computationGraph.png)
* This shows how you can take a left-to-right pass to figure out J, and we are going to learn how you can take a right-to-left pass to figure out some of the other variables as well.
* One step of a backward propagation on a computation graph yields derivative of final output variable.

## Derivatives with a Computation Graph
* dJ/dv = What is the derivative of J according to v?
**Chain rule** - The product in the change of `dJ/dv * dv/da = dJ/da` (if a => v => j)
* The picture below shows what we would name a variable that is looking for the derivative based on a certain variable. The derivative of the final output variable that you care about (such as J)
<img src="/images/week2/derivativeInCode.png" alt="Derivative in Code" width="45%" />

## Logistic Regression & Gradient Descent w/ one training example
* We want to modify the parameters to create a lower loss function.
￼￼![Logistic regression & gradient descent example](/images/week2/logisticRegressionFlow.png)
* Remember: to get the derivatives of a training example, you always need to go backwards.
* The first step is you need to find the derivative of the loss function.
    * `da = dL(a,y)/da = -y/a + 1-y/1-a`
￼￼![Computation of Loss](/images/week2/computationOfLoss.png)

## Gradient Descent on m Examples
￼￼￼![Vectorized vs. Non-vectorized](/images/week2/nonVectorizedVsVectorized.png)
* The above code is what gradient descent looks like with two variables. You’ll notice the two for loops—we will get rid of those for loops using vectorization.
* Vectorization techniques allow us to get rid of explicit for loops in our code.

# Python & Vectorization

## Vectorization
* Vectorization is so important because it helps code run faster and since we are working with such big datasets, that can make massive differences.
* `Z = np.dot(w,x) + b`
* `w transpose x = np.dot(w,x)`
￼￼￼![Vectorized vs. Non-vectorized 2](/images/week2/nonVectorizedVsVectorized2.png)

```
import numpy as np

a = np.array([1, 2, 3, 4])
print(a)

Import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()

print(c)
print(“Vectorized version: “ + str(1000 + (toc - tic)) + “ms”)

c = 0

tic = time.time()
for i in range(1000000):
	c += a[I] * b[i]
toc = time.time()

print(c)
print(“For loop: “ + str(1000 + (toc - tic)) + “ms”)
```

* Deep learning algorithms run better on a GPU but can also be run on a CPU.
    * Both GPU’s and CPU’s have SIMD. GPU’s are better at SIMD calculations.
    * **SIMD**- Single instantiation multiple data - takes advantage of parallelism to run your computations much faster if you use methods like np.dot().

## More Vectorization Examples
￼￼￼￼![Vectorize each Element](/images/week2/vectorizeEachElement.png)
* `np.zeros(nx, 1)` = this is the function you can use to create a vector

## Vectorizing Logistic Regression
* This is the code required for a forward propagation step.
* `X = [x<sup>1</sup> x<sup>2</sup> x<sup>3</sup> x<sup>m</sup>]`. Any capital letter that is used denotes this same format.
    * (nx by m matrix = (nx,m))
* `Z = np.dot(w.T, x) + b` (this is a 1xm matrix that calculates all of the z variables)
* Activation logic
    * `A = [a<sup>1</sup> a<sup>2</sup> a<sup>3</sup> a<sup>m</sup>] = sigmoid(Z)`
* **Broadcasting** - python will take b and “broadcast” it out to be [b<sup>1</sup> b<sup>2</sup> b<sup>m</sup>].
￼￼￼￼￼![Explanation of Vectorizing Logistic Regression](/images/week2/bigExplan.png)

## Vectorizing Logistic Regression’s Gradient Output
￼￼￼￼￼![Logistic regression gradient ouput](/images/week2/equationCombo.png)
* You can see how `dZ = A - Y` through the logic included in the picture above.
<img src="/images/week2/logicToCode.png" alt="Converting logic to code" width="75%" />
<img src="/images/week2/finalAns.png" alt="Final Equation" width="75%"/>
* The perfect example on the right compared to the first bad example on the left:
    * In this we compute the forward & backward propagations without using an exclusive for loop.
￼￼￼￼![Full Examples with Vectorization](/images/week2/rightVsWrong.png)
* Using the code on the right, we have just implemented a single iteration of gradient descent for logistic regression.
* If you want to implement multiple iterations of gradient descent then you still need to use a for loop. There isn’t a known better way of doing this.

## Broadcasting in Python
￼￼￼￼￼![Calorie Example of Broadcasting](/images/week2/carbExample.png)
```
import numpy as np

A = np.array([[56.0, 0.0, 4.0, 68.0],
		     [1.2, 104.0, 52.0, 8.0],
		     [1.8, 135.0, 99.0, 0.9]])
print(A)

cal = A.sum(axis=0)                         # this will sum them up vertically
print(cal)

percentage = 100 * A/cal.reshape(1,4)       # Calling reshape on cal is a little redundant because it’s already in that shape, but sometimes it’s good to do that if you aren’t sure what your matrix looks like.
print(percentage)
```

* The reshape command is very cheap to call so you don’t have to worry about it.
* If you have a vector i.e. [1, 2, 3, 4] * 100 and multiply it by a one dimensional vector then it will expand that 100 and multiple it across the board.
￼￼￼￼￼![Broadcast Example](/images/week2/broadcastVector.png)
￼￼￼￼￼![Broadcast Example 2](/images/week2/broadcastVector2.png)

## A note on python/numpy vectors
* Broadcasting is a pro and con in python.
    * Pro: It can allow for great expressivity.
    * Con: It can bring about strange bugs that are difficult to understand.
* **Rank 1 Array** - Data structure in python that doesn’t look act like a column or row vector. It looks like (5,). Avoid these with neural networks. 
    * Instead, make a column vector `(np.random.randn(5,1)` or a row vector `(np.random.randn(1,5)`
* If you are unsure about what type of vector you are working with, you can throw an assertion
    * `assert(a.shape == (5,1))`
    * Assertions are very inexpensive to execute and they help to serve as code documentation.
    * If you do end up with a rank 1 array, you can reshape it: `a.reshape((5,1))`

```
import numpy as np

a = np.random.rand(5)
print(a)
print(a.shape)                              # This will produce a “rank 1 array” i.e. (5,)
print(a.T)                                  # This will look the same as a
print(np.dot(a, a.T))                       # This produces a single number
```

* When coding neural networks, don’t use structures like this: (5,). Don’t use rank 1 arrays.

```
a = np.random.randn(5,1)
print(a)
print(a.T) # This produces a row vector.
print(np.dot(a, a.T)) # This will give you the product of a vector
```

## Quick tour of Jupyter/iPython Notebooks
* Jupyter notebooks are open source notebooks that you can use on any site. They are great for python problems.
* https://jupyter.org/install.html