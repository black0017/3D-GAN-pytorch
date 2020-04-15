# 3D GAN Pytorch (Learning a Probabilistic Latent Space of ObjectShapes via 3D Generative-Adversarial Modeling)
**Responsible** implementation of 3D-GAN NIPS 2016 paper that can be found https://papers.nips.cc/paper/6096-learning-a-probabilistic-latent-space-of-object-shapes-via-3d-generative-adversarial-modeling.pdf

We did our best to follow the original guidelines based on the papers. However, it is always good to try to reproduce the publication results from the original work. We also included our **DCGAN** implementation since **3D-GAN** is the natural extension of DCGAN in 3D space. For completeness, a Vanilla GAN is also included. All models all available in Google COLLAB. You can train them with the same training script that exists in train_gans.py

Data loaders to be updated soon.

## Google collab instructions and Usage
1. Go to https://colab.research.google.com
2. **```File```** > **```Upload notebook...```** > **```GitHub```** > **```Paste this link:``` https://github.com/black0017/3D-GAN-pytorch/blob/master/notebooks/3D_GAN_pytorch.ipynb**
3. Ensure that **```Runtime```** > **```Change runtime type```** is ```Python 3``` with ```GPU```
4. Run the code-blocks and enjoy :) 



## Detailed Info

#### Generator/Discriminator summary for batch size of 1
Trainable params: 17,601,408/11,048,833

Forward/backward pass size (MB): 67.25/63.75

Params size (MB): 67.14/42.15

Estimated Total Size (MB): 134.39/106.90

## References

[1] Wu, J., Zhang, C., Xue, T., Freeman, B., & Tenenbaum, J. (2016). Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling. In Advances in neural information processing systems (pp. 82-90).

[2] Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.

[3] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).


## Support 
If you **really** like this repository and find it useful, please consider (★) **starring** it, so that it can reach a broader audience of like-minded people.
