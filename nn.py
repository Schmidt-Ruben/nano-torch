from tensor import Parameter, Tensor


class Module:
    def __init__(self):
        self.training = True

    def forward(self, x):
        raise NotImplementedError("Module has no forward Method")

    def __call__(self, *args):
        return self.forward(*args)

    def parameters(self):
        # Collect all Parameters also from nested modules without duplicates
        parameters = []

        def check_attribute(attribute):
            if isinstance(attribute, Module):
                new_params = [param for param in attribute.parameters() if param not in parameters]
                parameters.extend(new_params)
            elif isinstance(attribute, Parameter):
                if attribute not in parameters:
                    parameters.append(attribute)
            elif isinstance(attribute, (list, tuple)):
                for item in attribute:
                    check_attribute(item)

        for attribute in vars(self).values():
            check_attribute(attribute)

        return parameters

    def num_params(self):
        return sum([parameter.data.size for parameter in self.parameters()])


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features
        # He initialisation with variance 2 / fan_in
        self.W = Parameter.normal(
            (in_features, out_features), std=(2 / in_features) ** (1 / 2), requires_grad=True
        )
        self.b = Parameter.zeros((out_features,), requires_grad=True) if bias else None

    def forward(self, x):
        if self.bias:
            return x @ self.W + self.b
        else:
            return x @ self.W


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.layers = list(modules)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = Parameter.normal(num_embeddings, embedding_dim, std=0.02**2)

    def forward(self, ids):
        if ids.dtype != 'int64':
            raise TypeError('ids musst be dtype int64')
        return self.weight[ids]


def cross_entropy(logits, targets):
    return -(1 / logits.shape[0]) * (logits.log_softmax() * targets).sum()
