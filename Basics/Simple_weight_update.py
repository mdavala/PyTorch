import torch.optim as optim
import torch.nn as nn
from Simple_network import Net

# instantiate network
net = Net()
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
# define your criterion for optimization
criterion = nn.MSELoss()

#data_set comes from dataset loader
for data in data_set:
  # set gradient buffers to zero
  optimizer.zero_grad()  
  # Passes the data to network
  output = net.forward(data)
  # calculates the loss which is Mean Squated error
  loss = criterion(output, target)
  # Propagates backwards for loss
  loss.backward()
  # Updates all the weights of the network
  optimizer.step() 