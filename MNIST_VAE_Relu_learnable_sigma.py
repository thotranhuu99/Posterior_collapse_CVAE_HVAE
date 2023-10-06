import torch
import argparse
from utils import set_seed, calculate_omega_lambda_1_linear_nonlearnable_Sigma, calculate_metric
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_preprocessing import calculate_Z
from datasets import MNISTDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from merge_images import merger_image
from torch.distributions.multivariate_normal import MultivariateNormal
import math
import json


class LinearVAE(nn.Module):
    def __init__(self, exp_name="SGD_latent_1_learnable_sigma", d_0=None, d_1=None,
                 d_hidden=None, eta_dec=None, c=None, beta=None, dataset=None):
        super().__init__()
        self.d_0 = d_0
        self.d_1 = d_1
        self.d_hidden = d_hidden
        self.eta_dec = eta_dec
        self.c = c
        self.eta_enc = eta_dec / c
        self.exp_name = exp_name
        self.beta = beta
        # self.W = nn.Parameter(torch.normal(0, 0.1, size = (d_1, d_0)), requires_grad=True)
        # self.U = nn.Parameter(torch.normal(0, 0.1, size = (d_0, d_1)), requires_grad=True)
        # self.Sigma_elements = nn.Parameter(torch.ones(d_1), requires_grad=False)
        self.img_2_hid_enc = nn.Linear(d_0, d_hidden)
        self.hid_2_muz_enc = nn.Linear(d_hidden, d_1)
        self.z_2_hid_dec = nn.Linear(d_1, d_hidden)
        self.hid_2_y_dec = nn.Linear(d_hidden, d_0)
        # self.sigma_elements = nn.Parameter(torch.FloatTensor([1, 1, 1, 1, 1]), requires_grad=False)
        self.sigma_elements = nn.Parameter(torch.ones(d_1), requires_grad=True)
        self.Phi_sqrt = torch.diag(torch.sqrt(dataset.phi)).to("cuda")
        self.P_A = dataset.P_A.to("cuda")
        
        self.Z = dataset.Z.to("cuda")
        self.relu = nn.ReLU()

    def encoder(self, x):
        hidden = self.relu(self.img_2_hid_enc(x))
        mu_z_enc = self.relu(self.hid_2_muz_enc(hidden))
        return mu_z_enc

    def decoder(self, z):
        hidden = self.relu(self.z_2_hid_dec(z))
        mu_y = self.hid_2_y_dec(hidden)
        return mu_y
    
    def forward(self, x):
        # Sigma = torch.diag() ** 2
        
        mu_z_enc = self.encoder(x)
        epsilon_z_enc = torch.randn_like(mu_z_enc)
        z_parameterized_enc = mu_z_enc + (self.sigma_elements ** 2) * epsilon_z_enc

        mu_y = self.decoder(z_parameterized_enc)
        epsilon_y = torch.randn_like(mu_y)
        y_parameterized = mu_y + self.eta_dec * epsilon_y

        return y_parameterized, z_parameterized_enc, mu_z_enc, mu_y
    
    def loss_fn(self, y_parameterized, z_parameterized_enc, mu_z_enc, mu_y, y):
        Sigma = torch.diag(self.sigma_elements) ** 2
        loss_reconstruct = (1 / (self.eta_dec ** 2)) * ((torch.norm(mu_y - y, p=2, dim=1) ** 2)).mean(dim=0)

        loss_KL = self.c**2 * (torch.norm(mu_z_enc, p=2, dim=1) ** 2).mean(dim=0) / (self.eta_dec**2)
        loss_KL += self.c**2 * torch.trace(Sigma) / (self.eta_dec**2)
        loss_KL -= 2 *torch.log(self.sigma_elements).sum()
        loss_KL += 2 * self.d_1 * math.log(self.eta_enc)
        loss_KL -= self.d_1
        loss_KL *= self.beta

        loss_KL_per_dim = self.c**2 * (mu_z_enc ** 2) / (self.eta_dec**2)
        loss_KL_per_dim += self.c**2 * torch.diag(Sigma) / (self.eta_dec**2)
        loss_KL_per_dim -= torch.diag(Sigma).log()
        loss_KL_per_dim += 2 * math.log(self.eta_enc)
        loss_KL_per_dim -= 1
        loss_KL_per_dim *= self.beta

        loss_KL = loss_KL / 2

        loss = loss_reconstruct + loss_KL
        loss_elements = {"loss_reconstruct": loss_reconstruct.detach().clone(),
                         "loss_KL_z": loss_KL.detach().clone(),
                         "loss_KL_per_dim": loss_KL_per_dim.detach().clone}
        return loss, self.hid_2_muz_enc.weight, self.hid_2_y_dec.weight, loss_KL_per_dim, mu_z_enc, loss_elements

    @staticmethod
    def logprob_Multivarate_Normal(x, muy, sigma):
        num_sample, dim = x.shape
        log_det_sigma = 2 * torch.sum(torch.log(torch.abs(sigma)), dim=1)
        # log_det_sigma = torch.logdet(torch.diag_embed(sigma**2))
        nll = -1/2 * torch.einsum('bd, bd -> b', (x-muy)**2, (1/sigma**2))
        nll += -1/2 * log_det_sigma
        nll += -dim/2 * math.log(2*math.pi)
        return nll

    @staticmethod
    def logsumexp(x, dim=None):
        if dim is None:
            xmax = x.max()
            xmax_ = x.max()
            return xmax_ + torch.log(torch.exp(x - xmax).sum())
        else:
            xmax, _ = x.max(dim, keepdim=True)
            xmax_, _ = x.max(dim)
            return xmax_ + torch.log(torch.exp(x - xmax).sum(dim))
    
    def calc_nll_p_x(self, x, y):
        # p(z|x) = N(mu_z_enc, \Sigma)
        # p(x|z) = N(mu_y, \eta_dec^2 * I)
        # p(z) = N(0, \eta_enc^2 * I)
        # p(x) = (p(z)*p(x|z)/q(z|x))
        # z~q(z|x)
        num_samples = 10000
        nll_p_x = torch.zeros(x.shape[0])
        
        mu_z_enc = self.encoder(x)
        for idx in range(x.shape[0]):
            distr_z_given_x = MultivariateNormal(mu_z_enc[idx], scale_tril=(torch.diag_embed(self.sigma_elements.abs())))
            z = distr_z_given_x.sample((num_samples,))
            log_prob_x_given_z = self.logprob_Multivarate_Normal(x[idx].unsqueeze(0).repeat(num_samples, 1), self.decoder(z), 
                                                              self.eta_dec * torch.ones(num_samples, x.shape[1], device="cuda"))
            log_prob_z = self.logprob_Multivarate_Normal(z, 0,
                                                         self.eta_enc * torch.ones(num_samples, z.shape[1], device="cuda"))
            log_prob_z_given_x = distr_z_given_x.log_prob(z)
            ll_p_x = self.logsumexp(log_prob_z + log_prob_x_given_z - log_prob_z_given_x)
            ll_p_x -= math.log(num_samples)
            nll_p_x[idx] = -ll_p_x
        return nll_p_x


def main(args):
    set_seed(args.seed)
    name = "MNIST_" + str(args.exp_name)\
           + "-" + "d_0" + str(args.d_0) \
           + "-" + "d_1" + str(args.d_1) \
           + "-" + "epochs_" + str(args.num_epochs) \
           + "-" + "seed_" + str(args.seed)
    # X : [n, d_0]
    beta_text = str(args.beta).replace(".", "~")
    json_folder = args.json_folder

    if os.path.exists(args.image_folder) is False:
        os.makedirs(args.image_folder, exist_ok=True)
    
    if os.path.exists(args.npy_folder) is False:
        os.makedirs(args.npy_folder, exist_ok=True)
    
    if os.path.exists(args.weight_folder) is False:
        os.makedirs(args.weight_folder, exist_ok=True)
    
    if os.path.exists(args.json_folder) is False:
        os.makedirs(args.json_folder, exist_ok=True)
    
    json_dict = vars(args)

    dataset = MNISTDataset(root='./data', train=True)
    dataset_test = MNISTDataset(root='./data', train=False)
    theta = dataset.theta
    
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)
    evalloader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = LinearVAE(d_0=args.d_0, d_1= args.d_1, d_hidden=args.d_hidden, exp_name="SGD_latent_1_nonlearnable_sigma",
                      eta_dec=args.eta_dec, c=args.c, beta=args.beta, dataset=dataset).to("cuda")
    # omega_array_theory, lambda_array_theory = calculate_omega_lambda_1_linear_nonlearnable_Sigma(theta_vector=theta, sigma_array=model.Sigma_elements, eta_enc=model.eta_enc,
    #                                                                                              eta_dec=model.eta_dec, d_1=args.d_1, beta=args.beta)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    pbar = tqdm(range(args.num_epochs))
    for epoch in pbar:
        loss_total = 0
        loss_KL_perdim_all = torch.empty([len(dataset), model.d_1], device="cuda")
        z_all = torch.empty([len(dataset), model.d_1], device="cuda")
        for batch_idx, (x, y, label) in enumerate(dataloader):
            x, y = x.to("cuda"), y.to("cuda")
            model.train()
            optimizer.zero_grad()
            y_parameterized, z_parameterized_enc, mu_z_enc, mu_y = model(x)
            loss, W_nonlinear, U_nonlinear, loss_KL_perdim, \
                z, loss_elements = model.loss_fn(y_parameterized, z_parameterized_enc, mu_z_enc, mu_y, y)
            z_all[batch_idx*args.batch_size:batch_idx*args.batch_size + x.shape[0]] = z
            loss_KL_perdim_all[batch_idx*args.batch_size:batch_idx*args.batch_size + x.shape[0]] = loss_KL_perdim
            loss.backward()
            optimizer.step()
            loss_total += loss
        loss_avg = loss / (batch_idx + 1)
        pbar.set_description("Loss: {:.12f}".format(loss_avg))

    nll_p_x_total = 0
    with torch.no_grad():
        for batch_idx, (x, y, label) in enumerate(evalloader):
            x, y = x.to("cuda"), y.to("cuda")
            nll_p_x = model.calc_nll_p_x(x, y)

            nll_p_x_total += nll_p_x.sum()
        nll_p_x_mean = nll_p_x_total / len(dataset_test)
    print("NLL is {}".format(nll_p_x_mean))

    loss_KL_perdim_all = torch.empty([len(dataset_test), model.d_1], device="cuda")
    with torch.no_grad():
        loss_elements_arrays = {"loss_reconstruct": [],
                         "loss_KL_z": [],
                         "loss_KL_per_dim": []}
        model.eval()
        for batch_idx, (x, y, label) in enumerate(evalloader):
            x, y = x.to("cuda"), y.to("cuda")
            model.train()
            optimizer.zero_grad()
            y_parameterized, z_parameterized_enc, mu_z_enc, mu_y = model(x)
            loss, W_nonlinear, U_nonlinear, loss_KL_perdim, \
                z, loss_elements = model.loss_fn(y_parameterized, z_parameterized_enc, mu_z_enc, mu_y, y)
            loss_KL_perdim_all[batch_idx*args.batch_size:batch_idx*args.batch_size + x.shape[0]] = loss_KL_perdim
            for key in loss_elements_arrays.keys():
                loss_elements_arrays[key].append(loss_elements[key])
    loss_KL_perdim_all = loss_KL_perdim_all / args.beta
    AU_z = 1 - (loss_KL_perdim_all < 0.01).sum() / loss_KL_perdim_all.numel()

    json_dict["NLL"] = float(nll_p_x_mean)
    json_dict["KL_z"] = float(torch.stack(loss_elements_arrays["loss_KL_z"]).mean()) / args.beta
    json_dict["AU_z"] = float(AU_z)
    json_dict["loss_KL_z_1"] = float(torch.stack(loss_elements_arrays["loss_KL_z"]).mean())

    file_name = str(args.exp_name)\
           + "-" + "nonlinear_True"\
           + "-" + "beta_" + str(args.beta) \
           + "-" + "eta_enc_" + str(model.eta_enc) \
           + "-" + "eta_dec_" + str(model.eta_dec) \
           + "-" + "epochs_" + str(args.num_epochs) \
           + "-" + "seed_" + str(args.seed)
    
    file_name = file_name.replace(".", "~")
    print("AU_z is {}".format(json_dict["AU_z"]))


    with open(os.path.join(json_folder, f"{file_name}.json"), "w") as outfile:
        json.dump(json_dict, outfile)

    def inference(num_samples):
        X = []
        Y = []
        idx = 0
        with torch.no_grad():
            for example, (x, y, label) in enumerate(testloader):
                z = model.encoder(x.to("cuda"))
                out = model.decoder(z).squeeze()
                img = out.view(-1, 1, 28, 28).to("cpu").numpy()
                plt.imsave(os.path.join(args.image_folder, f"ex{example}_sigma_learnable-beta_{beta_text}.png"), img.squeeze(), vmin=0, vmax=1)
                if example == num_samples-1:
                    break
    inference(num_samples=100)
    merger_image(num_samples=100, image_name=f"sigma_learnable-beta_{beta_text}", image_folder=args.image_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_0', type=int, default=784)
    parser.add_argument('--d_hidden', type=int, default=256)
    parser.add_argument('--d_1', type=int, default=16)
    parser.add_argument('--exp_name', type=str, default="VAE_learnable_sigma")
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--eta_dec', type=float, default=1.0)
    parser.add_argument('--c', type=float, default=1.0) # eta_dec / eta_enc
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument("--image_folder", type=str, default="output/performance/vae_nonlinear_learnable/image_folder")
    parser.add_argument("--npy_folder", type=str, default="output/performance/vae_nonlinear_learnable/npy")
    parser.add_argument("--weight_folder", type=str, default="output/performance/vae_nonlinear_learnable/weight")
    parser.add_argument("--json_folder", type=str, default="output/performance/vae_nonlinear_learnable/json")
    args = parser.parse_args()
    main(args)