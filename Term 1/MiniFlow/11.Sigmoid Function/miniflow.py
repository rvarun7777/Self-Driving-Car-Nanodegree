"""
Fix the Sigmoid class so that it computes the sigmoid function
on the forward pass!

Scroll down to get started.
"""

import numpy as np


class Layer:
    def __init__(self, inbound_layers=[]):
        self.inbound_layers = inbound_layers
        self.value = None
        self.outbound_layers = []
        for layer in inbound_layers:
            layer.outbound_layers.append(self)

    def forward():
        raise NotImplementedError

    def backward():
        raise NotImplementedError


class Input(Layer):
    def __init__(self):
        # An Input layer has no inbound layers,
        # so no need to pass anything to the Layer instantiator
        Layer.__init__(self)

    def forward(self):
        # Do nothing because nothing is calculated.
        pass

    def backward(self):
        # An Input Layer has no inputs so we refer to ourself
        # for the gradient
        self.gradients = {self: 0}
        for n in self.outbound_Layers:
            self.gradients[self] += n.gradients[self]


class Linear(Layer):
    def __init__(self, inbound_layer, weights, bias):
        # Notice the ordering of the input layers passed to the
        # Layer constructor.
        Layer.__init__(self, [inbound_layer, weights, bias])

    def forward(self):
        inputs = self.inbound_layers[0].value
        weights = self.inbound_layers[1].value
        bias = self.inbound_layers[2].value
        self.value = np.dot(inputs, weights) + bias


class Sigmoid(Layer):
    """
    You need to fix the `_sigmoid` and `forward` methods.
    """

    def __init__(self, layer):
        Layer.__init__(self, [layer])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used with `backward` as well.

        `x`: A numpy array-like object.

        Return the result of the sigmoid function.

        Your code here!
        """
        return 1./(1.+np.exp(x*-1))

    def forward(self):
        """
        Set the value of this layer to the result of the
        sigmoid function, `_sigmoid`.

        Your code here!
        """
        # This is a dummy value to prevent numpy errors
        # if you test without changing this method.
        self.value = self._sigmoid(self.inbound_layers[0].value)


def topological_sort(feed_dict):
    """
    Sort the layers in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Layer and the value is the respective value feed to that Layer.

    Returns a list of sorted layers.
    """

    input_layers = [n for n in feed_dict.keys()]

    G = {}
    layers = [n for n in input_layers]
    while len(layers) > 0:
        n = layers.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_layers:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            layers.append(m)

    L = []
    S = set(input_layers)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_layers:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_layer, sorted_layers):
    """
    Performs a forward pass through a list of sorted Layers.

    Arguments:

        `output_layer`: A Layer in the graph, should be the output layer (have no outgoing edges).
        `sorted_layers`: a topologically sorted list of layers.

    Returns the output layer's value
    """

    for n in sorted_layers:
        n.forward()

    return output_layer.value