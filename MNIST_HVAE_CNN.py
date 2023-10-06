import torch
import argparse
from utils import set_seed, calculate_metric, calculate_lambda_omega_2_linear_nonlearnable_isotropic_Sigma_1_learnable_Sigma_2_beta
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets import MNISTDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from merge_images import merger_image


class LinearVAE(nn.Module):
    def __init__(self, exp_name="SGD_latent_2_learnable_sigma2", d_0=None, d_1=None, d_2=None,
                 d_hidden=None, eta_dec=None, eta_enc=None, dataset=None, sigma_1=None,
                 beta_1=None, beta_2=None, nonlinear=None):
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
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.nonlinear = nonlinear

        self.Sigma_1 = (self.sigma_1 * torch.eye(d_1)).to("cuda")
        self.Phi_sqrt = torch.diag(torch.sqrt(dataset.phi)).to("cuda")
        self.P_A = dataset.P_A.to("cuda")
        
        self.Z = dataset.Z.to("cuda")
        self.img_2_hid_enc = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64,
                                                      kernel_size=3, stride=2, padding=1), nn.ReLU(),
                                           nn.Conv2d(in_channels=64, out_channels=32,
                                                      kernel_size=3, stride=2, padding=1), nn.ReLU(),
                                           nn.Flatten())
        self.hid_2_muz1_enc = nn.Linear(1568, d_1)
        self.z1_2_hid_enc = nn.Sequential(nn.Linear(d_1, d_hidden), nn.ReLU())
        self.hid_2_muz2_enc = nn.Linear(d_hidden, d_2)
        self.hid_2_sigmaz2_enc = nn.Linear(d_hidden, d_2)

        # Decoder
        self.z2_2_hid_dec = nn.Sequential(nn.Linear(d_2, d_hidden), nn.ReLU())
        self.hid_2_muz1_dec = nn.Linear(d_hidden, d_1)
        self.z1_2_hid_dec = nn.Sequential(nn.Linear(d_1, 1568), nn.Unflatten(1, (32, 7, 7)), nn.ReLU())
        self.hid_2_y_dec = nn.Sequential(nn.ConvTranspose2d(in_channels=32, out_channels=64,
                                                            kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                                         nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                                            kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                                         nn.ConvTranspose2d(in_channels=32, out_channels=1,
                                                            kernel_size=3, stride=1, padding=1),
                                         nn.Flatten()
                                        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()


    def encoder_z1(self, x):
        hidden = self.img_2_hid_enc(x)
        mu_z1_enc = self.tanh(self.hid_2_muz1_enc(hidden))
        return mu_z1_enc

    def encoder_z2(self, z_1):
        hidden = self.z1_2_hid_enc(z_1)
        mu_z2_enc = self.tanh(self.hid_2_muz2_enc(hidden))
        sigma_z2_enc = self.tanh(self.hid_2_sigmaz2_enc(hidden))
        return mu_z2_enc, sigma_z2_enc
    
    def decoder_z1(self, z_2):
        hidden = self.z2_2_hid_dec(z_2)
        mu_z1_dec = self.tanh(self.hid_2_muz1_dec(hidden))
        return mu_z1_dec

    def decoder_y(self, z_1):
        hidden = self.z1_2_hid_dec(z_1)
        mu_y = self.tanh(self.hid_2_y_dec(hidden))
        # test = self.hid_2_y_dec_1(hidden)
        # test = self.hid_2_y_dec_2(test)
        # test = self.hid_2_y_dec_3(test)
        return mu_y
        # return test

    def forward(self, x):
        mu_z1_enc = self.encoder_z1(x)
        epsilon_z1_enc = torch.randn_like(mu_z1_enc)
        z1_parameterized_enc = mu_z1_enc + torch.diag(self.Sigma_1) * epsilon_z1_enc
        
        mu_z2_enc, sigma_z2_enc = self.encoder_z2(z1_parameterized_enc)
        epsilon_z2_enc = torch.randn_like(mu_z2_enc)
        z2_parameterized_enc = mu_z2_enc + sigma_z2_enc * epsilon_z2_enc


        mu_z1_dec = self.decoder_z1(z2_parameterized_enc)
        epsilon_z1_dec = torch.randn_like(mu_z1_dec)
        z1_parameterized_dec = mu_z1_dec + self.eta_dec * epsilon_z1_dec

        mu_y = self.decoder_y(z1_parameterized_enc)
        epsilon_y = torch.randn_like(mu_y)
        y_parameterized = mu_y + self.eta_dec * epsilon_y

        return y_parameterized, z1_parameterized_enc, mu_z2_enc, sigma_z2_enc, mu_z1_dec, mu_y

    def loss_fn(self, y_parameterized, z1_parameterized_enc, mu_z2_enc, sigma_z2_enc, mu_z1_dec, mu_y, y):
        Sigma_2 = torch.diag_embed(sigma_z2_enc ** 2)
        loss_reconstruct = (1 / (self.eta_dec ** 2)) * ((torch.norm(mu_y - y, p=2, dim=1) ** 2)).mean(dim=0)
        
        loss_KL_z_1 = (1 / (self.eta_dec ** 2)) * (torch.norm(mu_z1_dec - z1_parameterized_enc, p=2, dim=1) ** 2).mean(dim=0)

        loss_KL_z_1 -= 2 * torch.diag(self.Sigma_1).log().sum() - self.d_1 * torch.log(self.eta_dec ** 2)
        loss_KL_z_1 -= self.d_1
        loss_KL_z_1 *= self.beta_1

        diag_Sigma = torch.diagonal(Sigma_2, dim1=1, dim2=2)
        loss_KL_z_2 = (1 / (self.eta_enc ** 2)) * (torch.norm(mu_z2_enc, p=2, dim=1) ** 2).mean(dim=0)

        loss_KL_z_2 += ((1 / (self.eta_enc ** 2)) * diag_Sigma.sum(dim=-1)).mean(dim=0)
        loss_KL_z_2 -= (diag_Sigma.log().sum(dim=-1) - self.d_2 * torch.log(self.eta_enc ** 2)).mean()
        loss_KL_z_2 -= self.d_2
        loss_KL_z_2 *= self.beta_2

        loss_reconstruct *= (self.eta_dec ** 2)
        loss_KL_z_1 *= (self.eta_dec ** 2) / 2
        loss_KL_z_2 *= (self.eta_dec ** 2) / 2
        loss = loss_reconstruct + loss_KL_z_1 + loss_KL_z_2
        loss_elements = {"loss_reconstruct": loss_reconstruct.detach().clone(),
                         "loss_KL_z_1": loss_KL_z_1.detach().clone(),
                         "loss_KL_z_2": loss_KL_z_2.detach().clone()}
        return loss, loss_elements

    def encoding(self, x):
        mu_z1_enc = self.encoder_z1(x)
        epsilon_z1_enc = torch.randn_like(mu_z1_enc)
        z1_parameterized_enc = mu_z1_enc + torch.diag(self.Sigma_1) * epsilon_z1_enc

        mu_z2_enc, sigma_z2_enc = self.encoder_z2(z1_parameterized_enc)
        epsilon_z2_enc = torch.randn_like(mu_z2_enc)
        z2_parameterized_enc = mu_z2_enc + sigma_z2_enc * epsilon_z2_enc
        return z2_parameterized_enc

    def decoding(self, z_2):
        mu_z1_dec = self.decoder_z1(z_2)
        epsilon_z1_dec = torch.randn_like(mu_z1_dec)
        z1_parameterized_dec = mu_z1_dec + self.eta_dec * epsilon_z1_dec

        mu_y = self.decoder_y(z1_parameterized_dec)
        y_parameterized = mu_y
        return y_parameterized

def main(args):
    set_seed(args.seed)
    name = "MNIST_" + str(args.exp_name)\
           + "-" + "nonlinear_" + str(args.nonlinear)\
           + "-" + "beta_1_" + str(args.beta_1) \
           + "-" + "beta_2_" + str(args.beta_2) \
           + "-" + "d_0_" + str(args.d_0) \
           + "-" + "d_1_" + str(args.d_1) \
           + "-" + "d_2_" + str(args.d_2) \
           + "-" + "lr_" + str(args.lr) \
           + "-" + "epochs_" + str(args.num_epochs) \
           + "-" + "seed_" + str(args.seed)
    image_folder = args.image_folder
    json_folder = args.json_folder

    if os.path.exists(image_folder) is False:
        os.makedirs(image_folder, exist_ok=True)
    
    if os.path.exists(json_folder) is False:
        os.makedirs(json_folder, exist_ok=True)

    json_dict = vars(args)

    dataset = MNISTDataset(root='./data', train=True)
    theta = dataset.theta
    theta = torch.as_tensor(theta, dtype=torch.float, device="cuda")
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)

    model = LinearVAE(d_0=args.d_0, d_1= args.d_1, d_2=args.d_2, exp_name="SGD_latent_2_nonlearnable_sigma", d_hidden=args.d_hidden,
                      eta_dec=args.eta_dec, eta_enc=args.eta_enc, dataset=dataset, sigma_1=args.sigma_1,
                      beta_1=args.beta_1, beta_2=args.beta_2, nonlinear=args.nonlinear).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    (lambda_array_theory, omega_array_theory, 
     theta_full_eigen_threshold) = calculate_lambda_omega_2_linear_nonlearnable_isotropic_Sigma_1_learnable_Sigma_2_beta(theta_vector=torch.as_tensor(theta, dtype=torch.float, device="cuda"),
                                                                                                          sigma_1=torch.as_tensor(args.sigma_1, dtype=torch.float, device="cuda"),
                                                                                                          eta_enc=torch.as_tensor(args.eta_enc,  dtype=torch.float, device="cuda"),
                                                                                                          eta_dec=torch.as_tensor(args.eta_dec,  dtype=torch.float, device="cuda"),
                                                                                                          c=torch.as_tensor(model.c,  dtype=torch.float, device="cuda"), d_1=args.d_1,
                                                                                                          beta_1=torch.as_tensor(args.beta_1, dtype=torch.float, device="cuda"),
                                                                                                          beta_2=torch.as_tensor(args.beta_2, dtype=torch.float, device="cuda"))

    np.save("theta_eigen.npy", theta.cpu().detach().numpy())
    np.savetxt('theta_eigen_hvae_mnist.txt', theta.to('cpu').numpy())
    active_mode_lambda = lambda_array_theory.count_nonzero()
    active_mode_omega = omega_array_theory.count_nonzero()
    print("Active mode of: lambda: {}, omega: {}".format(active_mode_lambda, active_mode_omega))
    pbar = tqdm(range(args.num_epochs))
    
    for epoch in pbar:
        loss_array = []
        loss_elements_arrays = {"loss_reconstruct": [], "loss_KL_z_1": [], "loss_KL_z_2": []}
        for batch_idx, (x, y, label) in enumerate(dataloader):
            x, y = x.reshape(x.shape[0], 28, 28).unsqueeze(1).to("cuda"), y.to("cuda")
            model.train()
            optimizer.zero_grad()
            y_parameterized, z1_parameterized_enc, mu_z2_enc, sigma_z2_enc, mu_z1_dec, mu_y = model(x)
            

            loss, loss_elements = model.loss_fn(y_parameterized, z1_parameterized_enc,
                                                mu_z2_enc, sigma_z2_enc, mu_z1_dec, mu_y, y)
            for key in loss_elements_arrays.keys():
                loss_elements_arrays[key].append(loss_elements[key])
            
            loss.backward()
            loss_array.append(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

    json_dict["loss"] = float(torch.stack(loss_array).mean())
    json_dict["loss_reconstruct"] = float(torch.stack(loss_elements_arrays["loss_reconstruct"]).mean())
    json_dict["loss_KL_z_1"] = float(torch.stack(loss_elements_arrays["loss_KL_z_1"]).mean())
    json_dict["loss_KL_z_2"] = float(torch.stack(loss_elements_arrays["loss_KL_z_2"]).mean())
    
    file_name = str(args.exp_name)\
           + "-" + "nonlinear_" + str(args.nonlinear)\
           + "-" + "active_" + str(int(active_mode_lambda))\
           + "-" + "beta_1_" + str(args.beta_1) \
           + "-" + "beta_2_" + str(args.beta_2) \
           + "-" + "epochs_" + str(args.num_epochs) \

    file_name = file_name.replace(".", "~")
    
    def inference(num_samples):
        images = []
        idx = 0
        with torch.no_grad():
            for example, (x, y, label) in enumerate(testloader):
                x = x.reshape(x.shape[0], 28, 28).unsqueeze(1).to("cuda")
                z_2 = model.encoding(x)
                out = model.decoding(z_2)
                img = out.view(-1, 1, 28, 28).to("cpu").numpy()
                plt.imsave(os.path.join(image_folder, f"ex{example}_{file_name}.png"), img.squeeze(), vmin=0, vmax=1)
                if example == num_samples-1:
                    break
    inference(num_samples=100)
    merger_image(num_samples=100, image_name=file_name, image_folder=image_folder)

    with open(os.path.join(json_folder, f"{file_name}.json"), "w") as outfile:
        json.dump(json_dict, outfile)
    
    print("Active mode of: lambda: {}/{}".format(active_mode_lambda, lambda_array_theory.shape[0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_0', type=int, default=784)
    parser.add_argument('--d_1', type=int, default=16)
    parser.add_argument('--d_2', type=int, default=16)
    parser.add_argument('--d_hidden', type=int, default=64)
    parser.add_argument('--exp_name', type=str, default="cnn_hvae_learnable_sigma2")
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--eta_enc', type=float, default=0.5)
    parser.add_argument('--eta_dec', type=float, default=0.5)
    parser.add_argument('--sigma_1', type=float, default=0.5)
    parser.add_argument('--beta_1', type=float, default=1)
    parser.add_argument('--beta_2', type=float, default=1)
    parser.add_argument('--nonlinear', type=str, default="True")
    parser.add_argument('--image_folder', type=str, default="output/cnn_hvae/image/nonlinear")
    parser.add_argument('--json_folder', type=str, default="output/cnn_hvae/json/nonlinear")
    args = parser.parse_args()  
    main(args)