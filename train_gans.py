import torch
from torch.autograd.variable import Variable


def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data


def zeros_target(size):
    '''
    FAKE data
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data


def train_discriminator(discriminator, optimizer, real_data, fake_data, loss):
    cuda = next(discriminator.parameters()).is_cuda
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    target_real = ones_target(N)
    if cuda:
        target_real.cuda()

    error_real = loss(prediction_real, target_real)
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    target_fake = zeros_target(N)
    if cuda:
        target_fake.cuda()
    error_fake = loss(prediction_fake, target_fake)
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(discriminator, optimizer, fake_data, loss):
    cuda = next(discriminator.parameters()).is_cuda
    N = fake_data.size(0)  # Reset gradients
    optimizer.zero_grad()  # Sample noise and generate fake data
    prediction = discriminator(fake_data)  # Calculate error and backpropagate
    target = ones_target(N)
    if cuda:
        target.cuda()

    error = loss(prediction, target)
    error.backward()  # Update weights with gradients
    optimizer.step()  # Return error
    return error
