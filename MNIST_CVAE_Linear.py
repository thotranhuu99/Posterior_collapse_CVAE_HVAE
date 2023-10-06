import torch
import argparse
from utils import set_seed, calculate_metric, calculate_lambda_sigma_cvae_beta
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets import MNISTDatasetCVEQuarter
from torch.utils.data import DataLoader
from merge_images import merger_image
import matplotlib.pyplot as plt
import os
import json

class CVAE(nn.Module):
    def __init__(self, dim_x=None, dim_y=None, dim_z=None,
                 d_hidden=None, eta_enc=None, eta_dec=None, dataset=None, beta=None):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.d_hidden = d_hidden
        self.eta_dec = torch.tensor(eta_dec, dtype=torch.float)
        self.eta_enc = torch.tensor(eta_enc, dtype=torch.float)
        self.c = eta_dec / eta_enc
        self.beta = beta
        self.W_1 = nn.Parameter(torch.normal(0, 0.1, size = (dim_z, dim_x)), requires_grad=True)
        self.W_2 = nn.Parameter(torch.normal(0, 0.1, size = (dim_z, dim_y)), requires_grad=True)
        self.U_1 = nn.Parameter(torch.normal(0, 0.1, size = (dim_y, dim_z)), requires_grad=True)
        self.U_2 = nn.Parameter(torch.normal(0, 0.1, size = (dim_y, dim_x)), requires_grad=True)
        self.sigma_parameterized = nn.Parameter((0.1 * torch.ones(dim_z)), requires_grad=True)

    def encoder(self, x, y):
        mu_z_x = (self.W_1 @ x.T).T
        mu_z_y = (self.W_2 @ y.T).T
        Sigma_parameterized = self.sigma_parameterized ** 2
        sigma = Sigma_parameterized ** (1/2)        
        mu_z = mu_z_x + mu_z_y
        
        return mu_z, sigma

    def decoder(self, x, z):
        mu_y_x = (self.U_2 @ x.T).T
        mu_y_z = (self.U_1 @ z.T).T
        mu_y = mu_y_x + mu_y_z

        return mu_y

    def forward(self, x, y):
        mu_z_enc, sigma = self.encoder(x, y)
        epsilon_z_enc = torch.randn_like(mu_z_enc)
        z_parameterized_enc = mu_z_enc + sigma * epsilon_z_enc

        mu_y = self.decoder(x, z_parameterized_enc)
        epsilon_y = torch.randn_like(mu_y)
        y_parameterized = mu_y + self.eta_dec * epsilon_y

        return y_parameterized, mu_z_enc, sigma, mu_y

    def loss_fn(self, y_parameterized, mu_z_enc, sigma, mu_y, y):
        Sigma = torch.diag(sigma ** 2)
        
        loss_reconstruct = (1 / (self.eta_dec ** 2)) * ((torch.norm(mu_y - y, p=2, dim=1) ** 2)).mean(dim=0)
        
        loss_KL = (1 / (self.eta_enc ** 2)) * (torch.norm(mu_z_enc, p=2, dim=1) ** 2).mean(dim=0)
        loss_KL_perdim = (1 / (self.eta_enc ** 2)) * (mu_z_enc ** 2).mean(dim=0)
        loss_KL += (1 / (self.eta_enc ** 2)) * torch.trace(Sigma)
        loss_KL -= torch.diag(Sigma).log().sum() - self.dim_z * torch.log(self.eta_enc ** 2)
        loss_KL -= self.dim_z
        loss_KL *= self.beta

        loss_KL_perdim -= 1
        loss_KL_perdim *= self.beta

        loss = loss_reconstruct + loss_KL
        loss_elements = {"loss_reconstruct": loss_reconstruct.detach().clone(),
                         "loss_KL_z": loss_KL.detach().clone()}
        return loss, loss_elements, loss_KL_perdim

    def encoding(self, x, y):
        mu_z_enc, sigma = self.encoder(x, y)
        epsilon_z_enc = torch.randn_like(mu_z_enc)
        z_parameterized_enc = mu_z_enc + sigma * epsilon_z_enc
        return z_parameterized_enc

    def decoding(self, x, z):
        mu_y = self.decoder(x, z)
        y_parameterized = mu_y
        return y_parameterized

def main(args):
    set_seed(args.seed)
    name = "MNIST_" + str(args.exp_name)\
        + "-" + "beta_" + str(args.beta) \
        + "-" + "eta_enc_" + str(args.eta_enc) \
        + "-" + "eta_dec_" + str(args.eta_dec) \
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

    dataset = MNISTDatasetCVEQuarter(root='./data', train=True)
    _, theta, _ = torch.linalg.svd(dataset.E.to("cuda"))
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)

    model = CVAE(dim_x=args.dim_x, dim_y=args.dim_y, dim_z=args.dim_z, d_hidden=args.d_hidden,
                 eta_enc = args.eta_enc, eta_dec=args.eta_dec, dataset=dataset,
                 beta=args.beta).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    (lambda_array_theory,
    sigma_array_theory) = calculate_lambda_sigma_cvae_beta(theta_vector=theta,
                                                      eta_enc=torch.as_tensor(args.eta_enc, dtype=torch.float, device="cuda"),
                                                      eta_dec=torch.as_tensor(args.eta_dec, dtype=torch.float, device="cuda"),
                                                      dim_z=args.dim_z, beta=torch.as_tensor(args.beta, dtype=torch.float, device="cuda"))

    active_mode_lambda = lambda_array_theory.count_nonzero()
    json_dict["active_mode_lambda"] = float(active_mode_lambda)
    pbar = tqdm(range(args.num_epochs))
    
    for epoch in pbar:
        loss_array = []
        mean_z = torch.zeros(model.dim_z, device="cuda")
        loss_KL_perdim_mean = torch.zeros(model.dim_z, device="cuda")
        loss_elements_arrays = {"loss_reconstruct": [], "loss_KL_z": []}
        for batch_idx, (x, y, label) in enumerate(dataloader):
            x, y = x.to("cuda"), y.to("cuda")
            model.train()
            optimizer.zero_grad()
            y_parameterized, mu_z_enc, sigma, mu_y = model(x, y)
            mean_z += torch.abs(mu_z_enc).sum(dim=0)
            
            loss, loss_elements, loss_KL_perdim = model.loss_fn(y_parameterized, mu_z_enc, sigma, mu_y, y)
            loss_KL_perdim_mean += loss_KL_perdim
            for key in loss_elements_arrays.keys():
                loss_elements_arrays[key].append(loss_elements[key])
            
            loss.backward()
            loss_array.append(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            pbar.set_description("Loss: {:.12f}".format(loss))
        mean_z /= len(dataset)
        loss_KL_perdim_mean /= len(dataset)
        min_z = torch.min(torch.abs(mean_z))
        min_loss_KL_perdim = min(loss_KL_perdim_mean)
        z_collapse = torch.sum( mean_z <=1e-4 )
        z_collapse_KL = torch.sum( loss_KL_perdim_mean <=1e-4 )
        
        if epoch % 10==0 or epoch==args.num_epochs - 1:
            _, lambda_array, _ = torch.linalg.svd(model.U_1)

            lambda_metric = calculate_metric(lambda_array_theory, lambda_array)
            sigma_metric = calculate_metric(sigma_array_theory, sigma)
    
    json_dict["loss"] = float(torch.stack(loss_array).mean())
    json_dict["loss_reconstruct"] = float(torch.stack(loss_elements_arrays["loss_reconstruct"]).mean())
    json_dict["loss_KL_z"] = float(torch.stack(loss_elements_arrays["loss_KL_z"]).mean())
    
    file_name = str(args.exp_name)\
           + "-" + "active_" + str(int(active_mode_lambda))\
           + "-" + "beta_" + str(args.beta) \
           + "-" + "epochs_" + str(args.num_epochs) \

    file_name = file_name.replace(".", "~")

    def inference(num_samples):
        X = []
        Y = []
        idx = 0
        with torch.no_grad():
            for example, (x, y, label) in enumerate(test_loader):
                z = model.encoding(x.to("cuda"), y.to("cuda"))
                out = model.decoding(x.to("cuda"), z).squeeze()
                quarter_dim = x.shape[-1]
                quarter_4 = x.squeeze().reshape(14,14)
                quarter_1 = out[:quarter_dim].reshape(14,14)
                quarter_2 = out[quarter_dim:2*quarter_dim].reshape(14,14)
                quarter_3 = out[2*quarter_dim:].reshape(14,14)
                img = torch.zeros([28, 28]).to("cuda")
                img[:14, :14] = quarter_1
                img[:14, 14:] = quarter_2
                img[14:, 14:] = quarter_3
                img[14:, :14] = quarter_4
                img = img.to("cpu")

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
    parser.add_argument('--dim_x', type=int, default=196)
    parser.add_argument('--dim_y', type=int, default=588)
    parser.add_argument('--dim_z', type=int, default=64)
    parser.add_argument('--d_hidden', type=int, default=128)
    parser.add_argument('--exp_name', type=str, default="CVAE")
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--eta_enc', type=float, default=1)
    parser.add_argument('--eta_dec', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--image_folder', type=str, default="output/cvae/image/linear")
    parser.add_argument('--json_folder', type=str, default="output/cvae/json/linear")
    args = parser.parse_args()  
    main(args)