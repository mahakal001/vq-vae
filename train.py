from src import trainloader
from src import VQVAETrainer
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from src.data import BATCH_SIZE

dataiter = iter(trainloader)
training_samples = len(trainloader) * BATCH_SIZE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = VQVAETrainer().to(device)
opt = optim.RMSprop(net.parameters(), lr=0.0001)

loss_list = []
epochs = 10
for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = inputs.to(device)
        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        outputs, losses = net(inputs)
        loss = F.mse_loss(inputs, outputs) + losses
        loss.backward()
        opt.step()

        # print statistics
        running_loss += loss.item()
    loss_list.append(running_loss )
    print("Loss: ", running_loss )

  
# save loss plot
epoch_list = [i+1 for i in range(epochs)]
print(epoch_list, loss_list)
plt.plot(epoch_list, loss_list)
plt.savefig("loss.jpg")
  


# Save vqvae model
torch.save(net.state_dict(), "vqvae.pt")