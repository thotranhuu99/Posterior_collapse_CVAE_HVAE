# Posterior Collapse in Linear Conditional and Hierarchical Variational Autoencoders

This is the code for the [paper](https://arxiv.org/abs/2306.05023) "Posterior Collapse in Linear Conditional and Hierarchical Variational Autoencoders".

Conference on Neural Information Processing Systems (NeurIPS), 2023

## Experiments
***CVAEs experiments***\
**Varying $\beta, \eta_{dec}$ experiment**\
*Varying $`\beta`$ args*: --eta_enc 1.0 --eta_dec 1.0 --beta $\in$ [0.1, 0.5, 1.0, 2.0]\
*Varying $`\eta_{dec}`$ args*: --eta_enc 1.0 --beta 1.0 --eta_dec $\in$ [0.25, 0.5, 1.0, 2.0]
```
CUDA_VISIBLE_DEVICES=0 python MNIST_CVAE_Relu.py
```
**Collapse level of MNIST digit datasets**\
*args*: --eta_enc 0.5 --eta_dec 0.5 --beta 1.0
```
CUDA_VISIBLE_DEVICES=0 python MNIST_CVAE_Linear_single_digit.py
CUDA_VISIBLE_DEVICES=0 python MNIST_CVAE_Relu_single_digit.py
CUDA_VISIBLE_DEVICES=0 python MNIST_CVAE_CNN_single_digit.py
```
***HVAEs experiments***\
**Varying $\beta_1, \beta_2, \eta_{dec}$ experiment**\
*Varying $`\beta_1`$ args*: --eta_enc 0.5 --eta_dec 0.5 --beta_2 1.0 --beta_1 $\in$ [0.1, 0.5, 1.0, 2.0]\
*Varying $`\beta_2`$ args*: --eta_enc 0.5 --eta_dec 0.5 --beta_1 1.0 --beta_2 $\in$ [0.1, 0.5, 1.0, 2.0]\
*Varying $`\eta_{dec}`$ args*: --eta_dec 0.5 --beta_1 1.0 --beta_2 1.0 --eta_enc $\in$ [0.25, 0.5, 1.0, 2.0]
```
CUDA_VISIBLE_DEVICES=0 python MNIST_HVAE_Relu.py
```
**Samples reconstructed from ReLU MHVAE with varied $\beta_1$ and $\beta_2$**\
*args*: --eta_enc 0.5 --eta_dec 0.5 --beta_1 $\in$ [0.1, 1.0, 2.0, 4.0, 6.0] --beta_2 $\in$ [0.1, 1.0, 2.0, 4.0, 6.0]
```
CUDA_VISIBLE_DEVICES=0 python MNIST_HVAE_Relu.py
```
## Additional experiments
***VAEs additional experiment***\
**Effect of learnable and unlearnable $\Sigma$ on posterior collapse**\
*args*: --eta_dec 1.0 --c 1.0 --beta 1.0 --sigma 1.0
```
CUDA_VISIBLE_DEVICES=0 python MNIST_VAE_Linear_learnable_sigma.py
CUDA_VISIBLE_DEVICES=0 python MNIST_VAE_Linear_nonlearnable_sigma.py
```
**Log-likelihood, KL and AU of VAEs with learnable and unlearnable $\Sigma$**\
*args*: --eta_dec 1.0 --c 1.0 --beta 1.0 --sigma 0.5
```
CUDA_VISIBLE_DEVICES=0 python MNIST_VAE_Relu_learnable_sigma.py
CUDA_VISIBLE_DEVICES=0 python MNIST_VAE_Relu_nonlearnable_sigma.py
```
***CVAEs additional experiment***\
**Linear CVAEs experiment**\
*args*: --beta in [0.1, 0.2 ,..., 4.9, 5.0]
```
CUDA_VISIBLE_DEVICES=0 python MNIST_CVAE_Linear.py
```
**Verification of Theorem 2**
```
CUDA_VISIBLE_DEVICES=0 python matrix_CVAE.py
CUDA_VISIBLE_DEVICES=0 python MNIST_CVAE_Linear.py --beta 1.0
```
**Log-likelihood, KL and AU of CVAEs with varied $\beta, \eta_{dec}$**\
*Varying $`\beta`$ args*: --eta_enc 1.0 --eta_dec 1.0 --beta $\in$ [0.1, 0.5, 1.0, 2.0]\
*Varying $`\eta_{dec}`$ args*: --eta_enc 1.0 --beta 1.0 --eta_dec $\in$ [0.25, 0.5, 1.0, 2.0]
```
CUDA_VISIBLE_DEVICES=0 python MNIST_CVAE_Relu.py
```
**Effects of the correlation of $x,y$ and posterior collapse**\
*args*: --corr_type $\in$ ["Identical", "Gaussian_noise_1_8", "Gaussian_noise_1_4", "Gaussian_noise_1_2", "Random"]
```
CUDA_VISIBLE_DEVICES=0 python Synthetic_CVAE_Relu_correlation.py
```
***HVAEs additional experiment***\
**Samples reconstructed from CNN MHVAE with varied $\beta_1$ and $\beta_2$**\
*args*: --eta_enc 0.5 --eta_dec 0.5 --beta_1 $\in$ [1.0, 2.0, 4.0, 8.0, 16.0] --beta_2 $\in$ [1.0, 2.0, 4.0, 8.0, 16.0]
```
CUDA_VISIBLE_DEVICES=0 python MNIST_HVAE_CNN.py
```
**Linear MHVAEs**\
*args*: --beta_2 in [0.1, 0.2 ,..., 6.9, 7.0]
```
CUDA_VISIBLE_DEVICES=0 python MNIST_HVAE_Linear_learnable_sigma_2.py
```
**Verification of Theorem 3**
```
CUDA_VISIBLE_DEVICES=0 python matrix_HVAE_learnable_sigma_2.py
CUDA_VISIBLE_DEVICES=0 python MNIST_HVAE_Linear_learnable_sigma_2.py --beta_1 1.0 --beta_2 1.0
```
**Verification of Theorem 5**
```
CUDA_VISIBLE_DEVICES=0 python matrix_HVAE_nonlearnable_sigma_2.py
CUDA_VISIBLE_DEVICES=0 python MNIST_HVAE_Linear_nonlearnable_sigma_2.py
```
**Log-likelihood, KL and AU of HVAEs with varied $\beta, \eta_{dec}$**\
*Varying $`\beta_1`$ args*: --eta_enc 0.5 --eta_dec 0.5 --beta_2 1.0 --beta_1 $\in$ [0.25, 0.5, 1.0, 2.0, 3.0]\
*Varying $`\beta_2`$ args*: --eta_enc 0.5 --eta_dec 0.5 --beta_1 1.0 --beta_2 $\in$ [0.25, 0.5, 1.0, 2.0, 3.0]\
*Varying $`\eta_{dec}`$ args*: --eta_dec 0.5 --beta_1 1.0 --beta_2 1.0 --eta_enc $\in$ [0.25, 0.5, 0.70, 1.0, 2.0]
```
CUDA_VISIBLE_DEVICES=0 python MNIST_HVAE_Relu.py
```
## Citation and reference 
For technical details and full experimental results, please check [our paper](https://arxiv.org/abs/2306.05023).
```
@article{dang2024vanilla,
  title={Beyond Vanilla Variational Autoencoders: Detecting Posterior Collapse in Conditional and Hierarchical Variational Autoencoders},
  author={Hien Dang and Tho Tran and Tan Nguyen and Nhat Ho},
  journal={arXiv preprint arXiv:2306.05023},
  year={2023}
}
```
