import torch
import argparse
from utils import set_seed, calculate_metric, calculate_lambda_sigma_cvae_beta
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_preprocessing import calculate_ZDE_CVAE

class CVAE_matrix(nn.Module):
    def __init__(self, exp_name="cvae_matrix", dim_x=5, dim_y=5, dim_z=None,
                 eta_enc=None, eta_dec=None, beta=None, phi=None, P_A=None, psi=None,
                 P_B=None, Z=None, D=None, E=None):
        super().__init__()
        self.d_x = dim_x
        self.d_y = dim_y
        self.d_z = dim_z
        self.eta_enc = eta_enc
        self.eta_dec = eta_dec
        self.c = eta_enc / eta_dec
        self.beta = beta
        self.exp_name = exp_name

        self.W_1 = nn.Parameter(torch.normal(0, 0.1, size = (dim_z, dim_x)), requires_grad=True)
        self.W_2 = nn.Parameter(torch.normal(0, 0.1, size = (dim_z, dim_y)), requires_grad=True)
        self.U_1 = nn.Parameter(torch.normal(0, 0.1, size = (dim_y, dim_z)), requires_grad=True)
        self.U_2 = nn.Parameter(torch.normal(0, 0.1, size = (dim_y, dim_x)), requires_grad=True)
        self.sigma_elements = nn.Parameter(torch.rand(dim_z), requires_grad=True)

        self.phi = phi
        self.P_A = P_A
        self.psi = psi
        self.P_B = P_B
        self.Z = Z
        self.D = D
        self.E = E

    def forward(self):
        V_1 = self.W_1 @ self.P_A @ torch.diag(self.phi ** (1/2))
        V_2 = self.W_2 @ self.P_B @ torch.diag(self.psi ** (1/2))
        T_2 = self.U_2 @ self.P_A @ torch.diag(self.phi ** (1/2))

        Sigma = torch.diag(self.sigma_elements ** 2)

        loss = torch.norm(self.U_1 @ V_1 + T_2, p="fro") ** 2
        loss += torch.norm(self.U_1 @ V_2 - self.D, p="fro") ** 2
        loss += 2 * torch.trace((self.U_1 @ V_1 + T_2) @ self.Z @ (self.U_1 @ V_2 - self.D).T)
        
        loss += self.beta * self.c**2 * torch.norm(V_1, p="fro") ** 2
        loss += self.beta * self.c**2 * torch.norm(V_2, p="fro") ** 2
        loss += self.beta * 2 * (self.c**2) * torch.trace(V_1 @ self.Z @ V_2.T)
        
        loss += torch.trace(self.U_1 @ Sigma @ self.U_1.T)
        loss += self.beta * self.c**2 * torch.trace(Sigma)
        loss /= self.eta_dec ** 2

        loss -= self.beta * (self.d_z + torch.logdet(Sigma))

        return loss, self.U_1, self.sigma_elements.data


def main(args):
    set_seed(args.seed)
    name = "synthetic_" + str(args.exp_name)\
           + "-" + "beta_" + str(args.beta) \
           + "-" + "dim_x" + str(args.dim_x) \
           + "-" + "dim_y" + str(args.dim_y) \
           + "-" + "dim_z" + str(args.dim_z) \
           + "-" + "epochs_" + str(args.num_iteration) \
           + "-" + "seed_" + str(args.seed)
    X = torch.normal(0, 0.1, size=(args.num_samples, args.dim_x)).cuda()
    Y = torch.normal(0, 0.1, size=(args.num_samples, args.dim_y)).cuda()
    phi, P_A, psi, P_B, Z, D, E = calculate_ZDE_CVAE(X, Y)
    _, theta, _ = torch.linalg.svd(E)  
    (lambda_array_theory,
    sigma_array_theory) = calculate_lambda_sigma_cvae_beta(theta_vector=theta,
                                                      eta_enc=torch.as_tensor(args.eta_enc, dtype=torch.float, device="cuda"),
                                                      eta_dec=torch.as_tensor(args.eta_dec, dtype=torch.float, device="cuda"),
                                                      dim_z=args.dim_z, beta=torch.as_tensor(args.beta, dtype=torch.float, device="cuda"))
 
    
    model = CVAE_matrix(dim_x=args.dim_x, dim_y=args.dim_y, dim_z=args.dim_z,
                        eta_enc=args.eta_enc, eta_dec=args.eta_dec, beta=args.beta, phi=phi,
                        P_A=P_A, psi=psi, P_B=P_B, Z=Z, D=D, E=E).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    pbar = tqdm(range(args.num_iteration))
    for iteration in pbar:
        model.train()
        optimizer.zero_grad()
        loss, U_1, sigma_array = model()
        _, lambda_array, _ = torch.linalg.svd(U_1)
        lambda_array = lambda_array.sort(descending=True)[0]
        sigma_array = sigma_array.sort(descending=True)[0]
        lambda_metric = calculate_metric(lambda_array_theory, lambda_array)
        sigma_metric = calculate_metric(sigma_array_theory, sigma_array)
        pbar.set_description("Loss: {:.12f}, lambda: {:.6f}, sigma: {:.6f}".format(loss, lambda_metric, sigma_metric))
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--dim_x', type=int, default=5)
    parser.add_argument('--dim_y', type=int, default=5)
    parser.add_argument('--dim_z', type=int, default=5)
    parser.add_argument('--exp_name', type=str, default="cvae_matrix")
    parser.add_argument('--num_iteration', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=str, default=2)
    parser.add_argument('--eta_enc', type=float, default=1.0)
    parser.add_argument('--eta_dec', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=2)
    args = parser.parse_args()
    main(args)