from main.cifar10.dataset import get_data_loader
from model.networks.VGG import VGG
# from model.networks.ShakeoutVGG import VGG
from model.knapsack_algorithm import Knapsack
from model.models import PassiveAuxiliaryLoss, AuxiliaryLossFunction
from torch.nn import CrossEntropyLoss
import numpy as np
from model.utils import compute_C, compute_Omega
from time import time

############################################
#  Phase 1: Detecting the Auxiliary Layers #
############################################
train_data, test_data = get_data_loader()
net = VGG('VGG11').cuda()

tic = time()
pax = PassiveAuxiliaryLoss(net, CrossEntropyLoss()).cuda()

compute_c = compute_C(net)
complex_o = compute_Omega(pax, train_data)

compute_c = [int(a / 10000) for a in compute_c]
p_2 = np.arange(1, len(compute_c) + 1, 1)
# complex_o = complex_o * 2 ** p_2
print(compute_c)
print(complex_o)

ka = Knapsack(complex_o, compute_c, 2000)
gamma = ka.optimize()

print('Selected Layer is ' + str(gamma))
gamma = [4, 5, 6]
toc = time()
print('Passive Elapse Time is {0}'.format(toc - tic))
############################################
#        Phase 2: Train the Network        #
############################################
ax_model = AuxiliaryLossFunction(gamma, net, CrossEntropyLoss)
ax_model.fit(train_data, test_data, 110, False)
