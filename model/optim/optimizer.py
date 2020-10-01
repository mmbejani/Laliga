from model.networks.auxiliary_network import AuxiliaryNetwork


class SGD:

    def __init__(self, net: AuxiliaryNetwork, lr: float):
        self.params = net.t_param
        self.lr = lr

    def step(self):
        for p in self.params:
            dp = p.grad.data
            p.data.add_(-self.lr, dp)
