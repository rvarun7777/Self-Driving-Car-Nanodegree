"""
Bonus Challenge!

Write your code in Add (scroll down).
"""

class Neuron:
    def __init__(self, inbound_neurons=[], label=''):
        # An optional description of the neuron - most useful for outputs.
        self.label = label
        # Neurons from which this Node receives values
        self.inbound_neurons = inbound_neurons
        # Neurons to which this Node passes values
        self.outbound_neurons = []
        # A calculated value
        self.value = None
        # Add this node as an outbound node on its inputs.
        for n in self.inbound_neurons:
            n.outbound_neurons.append(self)


    # These will be implemented in a subclass.
    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_neurons` and
        store the result in self.value.
        """
        raise NotImplemented

    def backward(self):
        """
        Backward propagation.

        Compute the gradient of the current Neuron with respect
        to the input neurons. The gradient of the loss with respect
        to the current Neuron should already be computed in the `gradients`
        attribute of the output neurons.
        """
        raise NotImplemented

class Input(Neuron):
    def __init__(self):
        # An Input Neuron has no inbound neurons,
        # so no need to pass anything to the Neuron instantiator
        Neuron.__init__(self)

    # NOTE: Input Neuron is the only Neuron where the value
    # may be passed as an argument to forward().
    #
    # All other Neuron implementations should get the value
    # of the previous neurons from self.inbound_neurons
    #
    # Example:
    # val0 = self.inbound_neurons[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value:
            self.value = value

    def backward(self):
        # An Input Neuron has no inputs so we refer to ourself
        # for the gradient
        self.gradients = {self: 0}
        for n in self.outbound_neurons:
            self.gradients[self] += n.gradients[self]


"""
Can you augment the Add class so that it accepts
any number of neurons as input?

Hint: this may be useful:
https://docs.python.org/3/tutorial/controlflow.html#unpacking-argument-lists
"""
class Add(Neuron):
    # You may need to change this...
    def __init__(self, inputs):
        Neuron.__init__(self, inputs)

    def forward(self):
        x_value = self.inbound_neurons[0].value
        y_value = self.inbound_neurons[1].value
        self.value = x_value + y_value
        
def topological_sort(feed_dict):
    """
    Sort the neurons in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Neuron and the value is the respective value feed to that Neuron.

    Returns a list of sorted neurons.
    """

    input_neurons = [n for n in feed_dict.keys()]

    G = {}
    neurons = [n for n in input_neurons]
    while len(neurons) > 0:
        n = neurons.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_neurons:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            neurons.append(m)

    L = []
    S = set(input_neurons)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_neurons:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_Neuron, sorted_neurons):
    """
    Performs a forward pass through a list of sorted neurons.

    Arguments:

        `output_Neuron`: A Neuron in the graph, should be the output Neuron (have no outgoing edges).
        `sorted_neurons`: a topologically sorted list of neurons.

    Returns the output Neuron's value
    """

    for n in sorted_neurons:
        n.forward()

    return output_Neuron.value
