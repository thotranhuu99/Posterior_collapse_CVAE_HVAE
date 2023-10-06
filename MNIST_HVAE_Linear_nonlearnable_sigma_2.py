import torch
import argparse
from utils import set_seed, calculate_metric, calculate_lambda_omega_2_linear_nonleanable_isotropic_Sigma_beta
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets import MNISTDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class LinearVAE(nn.Module):
    def __init__(self, exp_name="SGD_latent_2_nonlearnable_sigma", d_0=None, d_1=None, d_2=None,
                 d_hidden=None, eta_dec=None, eta_enc=None, dataset=None, sigma_1=None, sigma_2=None,
                 beta_1=None, beta_2=None):
        super().__init__()
        self.d_0 = d_0
        self.d_1 = d_1
        self.d_2 = d_2
        self.d_hidden = d_hidden
        self.eta_dec = torch.tensor(eta_dec, dtype=torch.float)
        self.eta_enc = torch.tensor(eta_enc, dtype=torch.float)
        self.c = eta_dec / eta_enc
        self.exp_name = exp_name

        self.sigma_1 = torch.tensor(sigma_1, dtype=torch.float)
        self.sigma_2 = torch.tensor(sigma_2, dtype=torch.float)
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.Sigma_1 = (self.sigma_1 * torch.eye(d_1)).to("cuda")
        self.Sigma_2 = (self.sigma_2 * torch.eye(d_2)).to("cuda")
        self.Phi_sqrt = torch.diag(torch.sqrt(dataset.phi)).to("cuda")
        self.P_A = dataset.P_A.to("cuda")

        self.Z = dataset.Z.to("cuda")
        self.W_1 = nn.Parameter(torch.normal(0, 0.1, size=(d_1, d_0)), requires_grad=True)
        self.U_1 = nn.Parameter(torch.normal(0, 0.1, size=(d_0, d_1)), requires_grad=True)
        self.W_2 = nn.Parameter(torch.normal(0, 0.1, size=(d_2, d_1)), requires_grad=True)
        self.U_2 = nn.Parameter(torch.normal(0, 0.1, size=(d_1, d_2)), requires_grad=True)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def encoder_z1(self, x):
        mu_z1_enc = (self.W_1 @ x.T).T
        return mu_z1_enc

    def encoder_z2(self, z_1):
        mu_z2_enc = (self.W_2 @ z_1.T).T
        return mu_z2_enc

    def decoder_z1(self, z_2):
        mu_z1_dec = (self.U_2 @ z_2.T).T
        return mu_z1_dec

    def decoder_y(self, z_1):
        mu_y = (self.U_1 @ z_1.T).T
        return mu_y

    def forward(self, x):
        mu_z1_enc = self.encoder_z1(x)
        epsilon_z1_enc = torch.randn_like(mu_z1_enc)
        z1_parameterized_enc = mu_z1_enc + torch.diag(self.Sigma_1) * epsilon_z1_enc

        mu_z2_enc = self.encoder_z2(z1_parameterized_enc)
        epsilon_z2_enc = torch.randn_like(mu_z2_enc)
        z2_parameterized_enc = mu_z2_enc + torch.diag(self.Sigma_2) * epsilon_z2_enc

        mu_z1_dec = self.decoder_z1(z2_parameterized_enc)
        epsilon_z1_dec = torch.randn_like(mu_z1_dec)
        z1_parameterized_dec = mu_z1_dec + self.eta_dec * epsilon_z1_dec

        mu_y = self.decoder_y(z1_parameterized_enc)
        epsilon_y = torch.randn_like(mu_y)
        y_parameterized = mu_y + self.eta_dec * epsilon_y

        return y_parameterized, mu_z1_enc, z1_parameterized_enc, mu_z2_enc, mu_z1_dec, mu_y

    def loss_fn(self, y_parameterized, mu_z1_enc, z1_parameterized_enc, mu_z2_enc, mu_z1_dec, mu_y, y):
        loss_reconstruct = (1 / (self.eta_dec ** 2)) * ((torch.norm(mu_y - y, p=2, dim=1) ** 2)).mean(dim=0)

        loss_KL_z_1 = (1 / (self.eta_dec ** 2)) * (torch.norm(mu_z1_dec - z1_parameterized_enc, p=2, dim=1) ** 2).mean(dim=0)
        loss_KL_z_1 -= 2 * torch.diag(self.Sigma_1).log().sum() - self.d_1 * torch.log(self.eta_dec ** 2)
        loss_KL_z_1 -= self.d_1
        loss_KL_z_1 *= self.beta_1

        loss_KL_z_2 = (1 / (self.eta_enc ** 2)) * (torch.norm(mu_z2_enc, p=2, dim=1) ** 2).mean(dim=0)
        loss_KL_z_2 += (1 / (self.eta_enc ** 2)) * torch.trace(self.Sigma_2 ** 2)
        loss_KL_z_2 -= 2 * torch.diag(self.Sigma_2).log().sum() - self.d_2 * torch.log(self.eta_enc ** 2)
        loss_KL_z_2 -= self.d_2
        loss_KL_z_2 *= self.beta_2

        loss_reconstruct *= (self.eta_dec ** 2)
        loss_KL_z_1 *= (self.eta_dec ** 2)
        loss_KL_z_2 *= (self.eta_dec ** 2)
        loss = loss_reconstruct + loss_KL_z_1 + loss_KL_z_2
        loss_elements = {"loss_reconstruct": loss_reconstruct.detach().clone(),
                         "loss_KL_z_1": loss_KL_z_1.detach().clone(),
                         "loss_KL_z_2": loss_KL_z_2.detach().clone()}
        return loss, loss_elements

    def encoding(self, x):
        mu_z1_enc = self.encoder_z1(x)
        epsilon_z1_enc = torch.randn_like(mu_z1_enc)
        z1_parameterized_enc = mu_z1_enc + torch.diag(self.Sigma_1) * epsilon_z1_enc

        mu_z2_enc = self.encoder_z2(z1_parameterized_enc)
        epsilon_z2_enc = torch.randn_like(mu_z2_enc)
        z2_parameterized_enc = mu_z2_enc + torch.diag(self.Sigma_2) * epsilon_z2_enc
        return z2_parameterized_enc

    def decoding(self, z_2):
        mu_z1_dec = self.decoder_z1(z_2)
        z1_parameterized_dec = mu_z1_dec

        mu_y = self.decoder_y(z1_parameterized_dec)
        y_parameterized = mu_y
        return y_parameterized

def main(args):
    set_seed(args.seed)
    name = "MNIST_" + str(args.exp_name)\
           + "-" + "beta_1_" + str(args.beta_1) \
           + "-" + "beta_2_" + str(args.beta_2) \
           + "-" + "d_0_" + str(args.d_0) \
           + "-" + "d_1_" + str(args.d_1) \
           + "-" + "d_2_" + str(args.d_2) \
           + "-" + "lr_" + str(args.lr) \
           + "-" + "epochs_" + str(args.num_epochs) \
           + "-" + "seed_" + str(args.seed)
    dataset = MNISTDataset(root='./data', train=True)
    theta = dataset.theta
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)

    model = LinearVAE(d_0=args.d_0, d_1= args.d_1, d_2=args.d_2, exp_name="SGD_latent_2_nonlearnable_sigma", d_hidden=args.d_hidden,
                      eta_dec=args.eta_dec, eta_enc=args.eta_enc, dataset=dataset, sigma_1=args.sigma_1, sigma_2=args.sigma_2,
                      beta_1=args.beta_1, beta_2=args.beta_2).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lambda_array_theory, omega_array_theory, theta_full_eigen_threshold = calculate_lambda_omega_2_linear_nonleanable_isotropic_Sigma_beta(theta_vector=theta,
                                                                                                          sigma_1=torch.as_tensor(args.sigma_1, dtype=torch.float),
                                                                                                          sigma_2=torch.as_tensor(args.sigma_2, dtype=torch.float),
                                                                                                          c=torch.as_tensor(model.c, dtype=torch.float), d_1=args.d_1,
                                                                                                          beta_1=torch.as_tensor(args.beta_1, dtype=torch.float),
                                                                                                          beta_2=torch.as_tensor(args.beta_2, dtype=torch.float))

    np.save("theta_eigen.npy", theta.cpu().detach().numpy())
    active_mode_lambda = lambda_array_theory.count_nonzero()
    active_mode_omega = omega_array_theory.count_nonzero()
    print("Active mode of: lambda: {}, omega: {}".format(active_mode_lambda, active_mode_omega))
    pbar = tqdm(range(args.num_epochs))
    
    for epoch in pbar:
        loss_array = []
        loss_elements_arrays = {"loss_reconstruct": [], "loss_KL_z_1": [], "loss_KL_z_2": []}
        for batch_idx, (x, y, label) in enumerate(dataloader):
            x, y = x.to("cuda"), y.to("cuda")
            model.train()
            optimizer.zero_grad()
            y_parameterized, mu_z1_enc, z1_parameterized_enc, mu_z2_enc, mu_z1_dec, mu_y = model(x)
            

            loss, loss_elements = model.loss_fn(y_parameterized, mu_z1_enc, z1_parameterized_enc,
                                                mu_z2_enc, mu_z1_dec, mu_y, y)
            for key in loss_elements_arrays.keys():
                loss_elements_arrays[key].append(loss_elements[key])

            loss.backward()
            loss_array.append(loss)
            optimizer.step()

        V_1 = model.W_1 @ model.P_A @ model.Phi_sqrt
        U_2 = model.U_2
        _, lambda_array, _ = torch.linalg.svd(V_1)
        _, omega_array, _ = torch.linalg.svd(U_2)
        lambda_metric = calculate_metric(lambda_array_theory, lambda_array)
        omega_metric = calculate_metric(omega_array_theory, omega_array)

    def inference(num_samples):
        images = []
        idx = 0
        with torch.no_grad():
            for idx, (x, y, label) in enumerate(testloader):
                z_2 = model.encoding(x.to("cuda").view(1, 784))
                out = model.decoding(z_2)
                out = out.view(-1, 1, 28, 28).to("cpu").numpy()
                plt.imsave(f"output/temp/generated_sampled_ex{idx}.png", out.squeeze(), vmin=0, vmax=1)
                if idx == num_samples:
                    break

    def inference_random(num_samples):
        with torch.no_grad():
            for example in range(num_samples):
                z_2 = torch.randn(args.d_2) * (args.eta_enc)
                out = model.decoding(z_2.to("cuda"))
                out = out.view(-1, 1, 28, 28).to("cpu").numpy()
                plt.imsave(f"output/temp/generated_random_ex{example}.png", out.squeeze(), vmin=0, vmax=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_0', type=int, default=784)
    parser.add_argument('--d_1', type=int, default=10)
    parser.add_argument('--d_2', type=int, default=10)
    parser.add_argument('--d_hidden', type=int, default=20)
    parser.add_argument('--exp_name', type=str, default="SGD_latent_2_nonlearnable_sigma")
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--eta_enc', type=float, default=0.5)
    parser.add_argument('--eta_dec', type=float, default=0.5)
    parser.add_argument('--sigma_1', type=float, default=0.5)
    parser.add_argument('--sigma_2', type=float, default=0.5)
    parser.add_argument('--beta_1', type=float, default=1)
    parser.add_argument('--beta_2', type=float, default=2)
    args = parser.parse_args()  
    main(args)