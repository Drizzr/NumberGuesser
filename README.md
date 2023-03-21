# Number Guesser

This project was heavily inspired by the following book:
http://neuralnetworksanddeeplearning.com

It implements the stochastic gradient descent using a full matrix-based approach (as suggested in the book)
which dramaticly speeds up the entire process!

For Example:
My Programm needs only 96.9 seconds for training the model for 30 epochs where as the example from the book needs
322.9 seconds (measured on an Macbook Air M1).


The guesser.py file contains the number-guesser-ai logic as mentioned in the book.
The inv.py file contains also uses the neural networks but tries the determine whether a 3x3 int matrix is invertible or not (I know that this could be done by simply calculating the determinant which would most likely be way faster)
