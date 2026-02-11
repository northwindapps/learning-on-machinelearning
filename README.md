The following is a list of questions I had during my learning of Machine Learning (and the answers from ChatGpt and Gemini). Hope you find it helpful if you're having a hard time in yours.

What is Cross-Entropy?
- It's a loss function primarily used for classification. It measures the difference between predicted probabilities and actual labels.

What is a weight and a bias?
- These are parameters of each neuron. The optimizer updates them to minimize the loss function’s value.

What "Minimize Loss Function" means?
- It means finding the specific weights and biases that produce the lowest possible error. The optimizer achieves this using algorithms like Gradient Descent (GD).

How to analysis and visualize the learning process?
- We typically use a Loss Curve (Loss vs. Epochs) for training. To visualize the "Loss Landscape," we can pick 2 weight directions and the loss value of these 2 directions to create a 3D graph.

Does the 3D graph form a continuous plain of loss values ?
- Actually, the loss landscape is technically continuous (mathematically), but it is extremely rugged, bumpy, and full of "valleys" and "hills." It's not a smooth, simple bowl.

If the loss landscape is rugged/complex, how to find a minimum point?
- We use Optimizers with Momentum (like Adam) or Stochastic methods. These help the model "jump" out of small local pits (Local Minima) or glide over bumpy areas to find a better, deeper minimum point.

What is feedforward and backpropagation?
- Feedforward: Relays and propagates input data through neurons to perform predictions. (The "Guessing" phase)
- Backpropagation: Calculates how much each weight contributed to the error and propagates that information backward to update the weights. (The "Learning" phase)
- In the training process, the model repeats both feedforward and backpropagation in a loop to continuously improve.

Does GD decide the direction in the trying? 
- GD shifts the weights and biases based on the gradient vector's direction.



