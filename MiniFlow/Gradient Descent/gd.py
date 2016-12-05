def gradient_descent_update(x, gradx, learning_rate):
    """
    Performs a gradient descent update.
    """
    # TODO: Implement gradient descent.
    x = x - learning_rate * gradx
    # Return the new value for x
    return x
