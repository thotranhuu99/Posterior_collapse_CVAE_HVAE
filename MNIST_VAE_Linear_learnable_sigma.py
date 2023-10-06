import torch
import argparse
from utils import set_seed, calculate_omega_lambda_1_linear_leanable_Sigma, calculate_metric
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_preprocessing import calculate_Z
from datasets import MNISTDataset
from torch.utils.data import DataLoader
import numpy as np
import math


class LinearVAE(nn.Module):
    def __init__(self, exp_name="SGD_latent_1_learnable_sigma", d_0=None, d_1=None,
                 eta_dec=None, c=None, beta=None, dataset=None):
        super().__init__()
        self.d_0 = d_0
        self.d_1 = d_1
        self.eta_dec = eta_dec
        self.c = c
        self.eta_enc = eta_dec / c
        self.exp_name = exp_name
        self.beta = beta
        self.W = nn.Parameter(torch.normal(0, 0.1, size = (d_1, d_0)), requires_grad=True)
        self.U = nn.Parameter(torch.normal(0, 0.1, size = (d_0, d_1)), requires_grad=True)
        self.Sigma_elements = nn.Parameter(torch.ones(d_1), requires_grad=True)
        self.Phi_sqrt = torch.diag(torch.sqrt(dataset.phi)).to("cuda")
        self.P_A = dataset.P_A.to("cuda")
        
        self.Z = dataset.Z.to("cuda")

    def forward(self, x, y):
        Sigma = torch.diag(self.Sigma_elements ** 2)
        V = self.W @ self.P_A @ self.Phi_sqrt
        loss_reconstruct = (torch.norm((self.U @ self.W @ x.T).T - y, p=2, dim=1) ** 2).mean(dim=0)
        loss_reconstruct += torch.trace(self.U.T @ self.U @ Sigma)
        loss_reconstruct /= self.eta_dec ** 2

        loss_KL = self.c**2 * (torch.norm((self.W @ x.T).T, p=2, dim=1) ** 2).mean(dim=0) / (self.eta_dec**2)
        loss_KL += self.c**2 * torch.trace(Sigma) / (self.eta_dec**2)
        loss_KL -= 2 *torch.log(self.Sigma_elements).sum()
        loss_KL += 2 * self.d_1 * math.log(self.eta_enc)
        loss_KL -= self.d_1
        loss_KL *= self.beta

        loss_KL_per_dim = self.c**2 * ((self.W @ x.T).T ** 2) / (self.eta_dec**2)
        loss_KL_per_dim += self.c**2 * torch.diag(Sigma) / (self.eta_dec**2)
        loss_KL_per_dim -= torch.diag(Sigma).log()
        loss_KL_per_dim += 2 * math.log(self.eta_enc)
        loss_KL_per_dim -= 1
        loss_KL_per_dim *= self.beta


        loss = loss_reconstruct + loss_KL

        return loss, self.U, self.W, V, loss_KL_per_dim

def main(args):
    set_seed(args.seed)
    name = "MNIST_" + str(args.exp_name)\
           + "-" + "d_0" + str(args.d_0) \
           + "-" + "d_1" + str(args.d_1) \
           + "-" + "epochs_" + str(args.num_epochs) \
           + "-" + "seed_" + str(args.seed)
    # X : [n, d_0]

    dataset = MNISTDataset(root='./data', train=True)
    theta = dataset.theta
    
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = LinearVAE(d_0=args.d_0, d_1= args.d_1, exp_name="SGD_latent_1_learnable_sigma",
                      eta_dec=args.eta_dec, c=args.c, beta=args.beta, dataset=dataset).to("cuda")
    omega_array_theory, lambda_array_theory, sigma_array = calculate_omega_lambda_1_linear_leanable_Sigma(theta_vector=theta, eta_enc = model.eta_enc,
                                                                        eta_dec=model.eta_dec, d_1=args.d_1, beta=args.beta)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    pbar = tqdm(range(args.num_epochs))
    lambda_metric = -1.0
    omega_metric = -1.0
    for epoch in pbar:
        loss_total = 0
        loss_KL_perdim_all = torch.empty([len(dataset), model.d_1], device="cuda")
        for batch_idx, (x, y, label) in enumerate(dataloader):
            x, y = x.to("cuda"), y.to("cuda")
            model.train()
            optimizer.zero_grad()
            loss, U, W, V, loss_KL_perdim = model(x, y)
            loss_KL_perdim_all[batch_idx*args.batch_size:batch_idx*args.batch_size + x.shape[0]] = loss_KL_perdim
            loss.backward()
            optimizer.step()
            loss_total += loss
        loss_avg = loss / (batch_idx + 1)
        pbar.set_description("Loss: {:.12f}, lambda: {:.6f}, omega: {:.6f}".format(loss_avg, lambda_metric, omega_metric))
        _, omega_array, _ = torch.linalg.svd(U)
        _, lambda_array, _ = torch.linalg.svd(V)
        omega_metric = calculate_metric(omega_array_theory, omega_array)
        lambda_metric = calculate_metric(lambda_array_theory, lambda_array)
    #     U_np = U.detach().cpu().numpy()
    #     np.savetxt("U_learnable_sigma_2_active.csv", U_np, delimiter=",")
    #     W_np = W.detach().cpu().numpy()
    #     np.savetxt("W_learnable_sigma_2_active.csv", W_np, delimiter=",")
    #     V_np = V.detach().cpu().numpy()
    #     np.savetxt("V_learnable_sigma_2_active.csv", V_np, delimiter=",")
            
    # file_name = f"output/vanilla_vae/vanilla_vae_learnable_sigma_beta_{args.beta}"
    # file_name = file_name.replace(".", "~") + ".npy"
    # np.save(file_name, loss_KL_perdim_all.detach().cpu().numpy())

    print("test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_0', type=int, default=784)
    parser.add_argument('--d_1', type=int, default=5)
    parser.add_argument('--exp_name', type=str, default="SGD_latent_1_learnable_sigma")
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=str, default=1)
    parser.add_argument('--eta_dec', type=float, default=1.0)
    parser.add_argument('--c', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=4.0)
    args = parser.parse_args()
    main(args)