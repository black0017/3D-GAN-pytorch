# 3D GAN Pytorch
Responsible implementation of 3D-GAN NIPS 2016 paper that can be found https://papers.nips.cc/paper/6096-learning-a-probabilistic-latent-space-of-object-shapes-via-3d-generative-adversarial-modeling.pdf

## Google collab instructions and usage

## Usage


## Detailed Info

### Generator summary
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1             [-1, 1, 32768]       6,586,368
   ConvTranspose3d-2         [-1, 256, 8, 8, 8]       8,388,608
       BatchNorm3d-3         [-1, 256, 8, 8, 8]             512
              ReLU-4         [-1, 256, 8, 8, 8]               0
   ConvTranspose3d-5      [-1, 128, 16, 16, 16]       2,097,152
       BatchNorm3d-6      [-1, 128, 16, 16, 16]             256
              ReLU-7      [-1, 128, 16, 16, 16]               0
   ConvTranspose3d-8       [-1, 64, 32, 32, 32]         524,288
       BatchNorm3d-9       [-1, 64, 32, 32, 32]             128
             ReLU-10       [-1, 64, 32, 32, 32]               0
  ConvTranspose3d-11        [-1, 1, 64, 64, 64]           4,096
          Sigmoid-12        [-1, 1, 64, 64, 64]               0
================================================================
Total params: 17,601,408
Trainable params: 17,601,408
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 67.25
Params size (MB): 67.14
Estimated Total Size (MB): 134.39
----------------------------------------------------------------

### Discriminator summary

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv3d-1       [-1, 64, 32, 32, 32]           4,096
       BatchNorm3d-2       [-1, 64, 32, 32, 32]             128
         LeakyReLU-3       [-1, 64, 32, 32, 32]               0
            Conv3d-4      [-1, 128, 16, 16, 16]         524,288
       BatchNorm3d-5      [-1, 128, 16, 16, 16]             256
         LeakyReLU-6      [-1, 128, 16, 16, 16]               0
            Conv3d-7         [-1, 256, 8, 8, 8]       2,097,152
       BatchNorm3d-8         [-1, 256, 8, 8, 8]             512
         LeakyReLU-9         [-1, 256, 8, 8, 8]               0
           Conv3d-10         [-1, 512, 4, 4, 4]       8,388,608
      BatchNorm3d-11         [-1, 512, 4, 4, 4]           1,024
        LeakyReLU-12         [-1, 512, 4, 4, 4]               0
           Linear-13                    [-1, 1]          32,769
          Sigmoid-14                    [-1, 1]               0
================================================================
Total params: 11,048,833
Trainable params: 11,048,833
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1.00
Forward/backward pass size (MB): 63.75
Params size (MB): 42.15
Estimated Total Size (MB): 106.90
----------------------------------------------------------------






## Support 
If you **really** like this repository and find it useful, please consider (★) **starring** it, so that it can reach a broader audience of like-minded people.
