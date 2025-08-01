import numpy as np
import random


def train(X, w):
    # Generate next weights randomly
    w_next = np.copy(w)
    for i in range(len(w)):
        change = random.random()
        if random.randint(0, 1) == 1:
            change *= -1
        w_next[i] += change

    # Calculate loss for each observation using
    # current and new weights
    loss = 0
    next_loss = 0
    for x in X:
        target = x[-1]
        result = np.dot(x[:-1], w)
        loss += abs(target - result)
        result = np.dot(x[:-1], w_next)
        next_loss += abs(target - result) 

    # Keep better performing weights
    if next_loss < loss:
        w[:] = w_next[:]


def main():
    # Data with last column as target
    X = np.array([
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [2, 4, 5, 6],
            [4, 5, 6, 7]
        ])

    # Starting weights
    w = np.array([.1, .1, .1])

    # Print initial predictions 
    print(np.dot(X[:, :-1], w))

    # Train
    for i in range(100000):
        train(X, w)

    # Print final predictions and associated weights
    print(np.dot(X[:, :-1], w))
    print(f"final weights: {w}")


if __name__ == "__main__":
    main()
