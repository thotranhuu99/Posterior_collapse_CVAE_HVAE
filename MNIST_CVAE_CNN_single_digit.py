import torch
import argparse
from utils import set_seed, calculate_lambda_sigma_cvae_beta
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets import MNISTDatasetCVEQuarterDigitCNN
from torch.utils.data import DataLoader
from merge_images import merger_image
import numpy as np
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
        # Encoder
        self.x_2_hid_enc = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32,
                                                      kernel_size=3, stride=2, padding=1), nn.ReLU(),
                                           nn.Conv2d(in_channels=32, out_channels=16,
                                                      kernel_size=3, stride=2, padding=1), nn.ReLU(),
                                           nn.Flatten())
        self.y_2_hid_enc = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32,
                                                      kernel_size=3, stride=2, padding=1), nn.ReLU(),
                                           nn.Conv2d(in_channels=32, out_channels=16,
                                                      kernel_size=3, stride=2, padding=1), nn.ReLU(),
                                           nn.Flatten())
        self.xhid_2_z_enc = nn.Linear(256, dim_z)
        self.xhid_2_zsigma_enc = nn.Linear(256, dim_z)
        self.yhid_2_z_enc = nn.Linear(784, dim_z)
        self.yhid_2_zsigma_dec = nn.Linear(784, dim_z)
        
        # Decoder
        self.x_2_hid_dec = nn.Sequential(nn.ConvTranspose2d(in_channels=1, out_channels=32,
                                                            kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                                         nn.ConvTranspose2d(in_channels=32, out_channels=1,
                                                            kernel_size=3, stride=1, padding=1),
                                         nn.Flatten()
                                        )
        self.xhid_2_y_dec = nn.Linear(784, dim_y)

        self.z_2_hid_dec = nn.Sequential(nn.Linear(dim_z, 784), nn.Unflatten(1, (16, 7, 7)), nn.ReLU())
        self.zhid_2_y_dec = nn.Sequential(nn.ConvTranspose2d(in_channels=16, out_channels=32,
                                                            kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                                         nn.ConvTranspose2d(in_channels=32, out_channels=16,
                                                            kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                                         nn.ConvTranspose2d(in_channels=16, out_channels=1,
                                                            kernel_size=3, stride=1, padding=1),
                                         nn.Flatten()
                                        )

        self.relu = nn.ReLU()

    def encoder(self, x, y):
        z_x_hid = self.x_2_hid_enc(x)
        mu_z_x = self.xhid_2_z_enc(z_x_hid)

        z_y_hid = self.y_2_hid_enc(y)
        mu_z_y = self.yhid_2_z_enc(z_y_hid)
        sigma = self.xhid_2_zsigma_enc(z_x_hid) + self.yhid_2_zsigma_dec(z_y_hid)
        
        mu_z = mu_z_x + mu_z_y
        
        return mu_z, sigma

    def decoder(self, x, z):
        mu_y_x = self.x_2_hid_dec(x)
        mu_y_x = self.xhid_2_y_dec(mu_y_x)

        mu_y_z = self.z_2_hid_dec(z)
        mu_y_z = self.zhid_2_y_dec(mu_y_z)
        mu_y_z = mu_y_z.reshape(-1, 28, 28)
        quarter_1 = mu_y_z[:, :14, :14].reshape(mu_y_z.shape[0], -1)
        quarter_2 = mu_y_z[:, :14, 14:].reshape(mu_y_z.shape[0], -1)
        quarter_3 = mu_y_z[:, 14:, 14:].reshape(mu_y_z.shape[0], -1)
        mu_y_z = torch.cat([quarter_1, quarter_2, quarter_3], dim=1)

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
        Sigma = torch.diag_embed(sigma ** 2)
        
        loss_reconstruct = (1 / (self.eta_dec ** 2)) * ((torch.norm(mu_y - y, p=2, dim=1) ** 2)).mean(dim=0)
        
        loss_KL = (1 / (self.eta_enc ** 2)) * (torch.norm(mu_z_enc, p=2, dim=1) ** 2).mean(dim=0)
        loss_KL_perdim = (1 / (self.eta_enc ** 2)) * (mu_z_enc ** 2)
        diag_Sigma = torch.diagonal(Sigma, dim1=1, dim2=2)
        loss_KL += ((1 / (self.eta_enc ** 2)) * diag_Sigma.sum(dim=-1)).mean(dim=0)
        loss_KL -= (diag_Sigma.log().sum(dim=-1) - self.dim_z * torch.log(self.eta_enc ** 2)).mean(dim=0)
        loss_KL_perdim += ((1 / (self.eta_enc ** 2)) * diag_Sigma)
        loss_KL_perdim -= (diag_Sigma.log() - 1 * torch.log(self.eta_enc ** 2))
        loss_KL -= self.dim_z
        loss_KL *= self.beta

        loss_KL_perdim -= 1
        loss_KL_perdim *= self.beta

        loss_KL *= 1/2
        loss_KL_perdim *= 1/2

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
        + "-" + "nonlinear_True"\
        + "-" + "digit_" + str(args.digit)\
        + "-" + "beta_" + str(args.beta) \
        + "-" + "eta_enc_" + str(args.eta_enc) \
        + "-" + "eta_dec_" + str(args.eta_dec) \
        + "-" + "lr_" + str(args.lr) \
        + "-" + "epochs_" + str(args.num_epochs) \
        + "-" + "seed_" + str(args.seed)
    
    image_folder = os.path.join(args.image_folder, f"digit_{args.digit}")
    json_folder = os.path.join(args.json_folder, f"digit_{args.digit}")
    theta_path = "theta_npy/theta_cvae_quarter"
    npy_path = "output/cnn_cvae/npy"
    
    if os.path.exists(image_folder) is False:
        os.makedirs(image_folder, exist_ok=True)
    
    if os.path.exists(json_folder) is False:
        os.makedirs(json_folder, exist_ok=True)
    
    if os.path.exists(theta_path) is False:
        os.makedirs(theta_path, exist_ok=True)
    
    if os.path.exists(npy_path) is False:
        os.makedirs(npy_path, exist_ok=True)
    
    json_dict = vars(args)

    dataset = MNISTDatasetCVEQuarterDigitCNN(root='./data', train=True, digit=args.digit)
    _, theta, _ = torch.linalg.svd(dataset.E.to("cuda"))
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)

    model = CVAE(dim_x=args.dim_x, dim_y=args.dim_y, dim_z=args.dim_z, d_hidden=args.d_hidden,
                 eta_enc = args.eta_enc, eta_dec=args.eta_dec, dataset=dataset, beta=args.beta).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    (lambda_array_theory,
    sigma_array_theory) = calculate_lambda_sigma_cvae_beta(theta_vector=theta,
                                                      eta_enc=torch.as_tensor(args.eta_enc, dtype=torch.float, device="cuda"),
                                                      eta_dec=torch.as_tensor(args.eta_dec, dtype=torch.float, device="cuda"),
                                                      dim_z=args.dim_z, beta=torch.as_tensor(args.beta, dtype=torch.float, device="cuda"))

    np.savetxt(os.path.join(theta_path, f"digit_{args.digit}.txt"), theta.to('cpu').numpy())
    active_mode_lambda = lambda_array_theory.count_nonzero()
    json_dict["active_mode_lambda"] = float(active_mode_lambda)
    pbar = tqdm(range(args.num_epochs))
    
    for epoch in pbar:
        loss_array = []
        loss_KL_perdim_all = torch.empty([len(dataset), model.dim_z], device="cuda")
        loss_elements_arrays = {"loss_reconstruct": [], "loss_KL_z": []}
        for batch_idx, (x, y, x_square, y_square, label) in enumerate(dataloader):
            # x, y = x.reshape(x.shape[0], 14, 14).to("cuda"), y.to("cuda")
            # x, y = x.reshape(x.shape[0], 28, 28).unsqueeze(1).to("cuda"), y.to("cuda")
            x_square, y, y_square = x_square.to("cuda"), y.to("cuda"), y_square.to("cuda")
            model.train()
            optimizer.zero_grad()
            y_parameterized, mu_z_enc, sigma, mu_y = model(x_square, y_square)
            
            loss, loss_elements, loss_KL_perdim = model.loss_fn(y_parameterized, mu_z_enc, sigma, mu_y, y)
            loss_KL_perdim_all[batch_idx*args.batch_size:batch_idx*args.batch_size + x.shape[0]] = loss_KL_perdim
            for key in loss_elements_arrays.keys():
                loss_elements_arrays[key].append(loss_elements[key])
            
            loss.backward()
            loss_array.append(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            pbar.set_description("Loss: {:.12f}".format(loss))
    
    json_dict["loss"] = float(torch.stack(loss_array).mean())
    json_dict["loss_reconstruct"] = float(torch.stack(loss_elements_arrays["loss_reconstruct"]).mean())
    json_dict["loss_KL_z"] = float(torch.stack(loss_elements_arrays["loss_KL_z"]).mean())

    file_name = str(args.exp_name)\
           + "-" + "nonlinear_True"\
           + "-" + "digit_" + str(args.digit)\
           + "-" + "active_" + str(int(active_mode_lambda))\
           + "-" + "beta_" + str(args.beta) \
           + "-" + "epochs_" + str(args.num_epochs) \

    file_name = file_name.replace(".", "~")

    np.save(os.path.join(npy_path, f'{file_name}.npy'), loss_KL_perdim_all.detach().cpu().numpy())
    
    def inference(num_samples):
        X = []
        Y = []
        idx = 0
        with torch.no_grad():
            for example, (x, y, x_square, y_square, label) in enumerate(test_loader):
                z = model.encoding(x_square.to("cuda"), y_square.to("cuda"))
                out = model.decoding(x_square.to("cuda"), z).squeeze()
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
    parser.add_argument('--dim_z', type=int, default=16)
    parser.add_argument('--d_hidden', type=int, default=16)
    parser.add_argument('--exp_name', type=str, default="CVAE_CNN")
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--eta_enc', type=float, default=0.5)
    parser.add_argument('--eta_dec', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--image_folder', type=str, default="output/cnn_cvae/image/nonlinear_single_digit/")
    parser.add_argument('--json_folder', type=str, default="output/cnn_cvae/json/nonlinear_single_digit")
    parser.add_argument('--digit', type=int, default=1)
    args = parser.parse_args()  
    main(args)