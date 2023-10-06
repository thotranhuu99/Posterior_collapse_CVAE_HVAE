import torch
import argparse
from utils import set_seed, calculate_lambda_omega_2_linear_nonleanable_isotropic_Sigma_beta, calculate_metric
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_preprocessing import calculate_Z


class LinearVAE_matrix(nn.Module):
    def __init__(self, exp_name="matrix_latent_2_nonlearnable_sigma", d_0=5, d_1=5, d_2=5,
                 sigma_1=None, sigma_2=None, eta_dec=None, eta_enc=None, beta_1=None, beta_2=None,
                 Phi_sqrt=None, P_A=None, Z=None):
        super().__init__()
        self.d_0 = d_0
        self.d_1 = d_1
        self.d_2 = d_2
        self.eta_enc = eta_enc
        self.eta_dec = eta_dec
        self.c = eta_dec / eta_enc
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.exp_name = exp_name
        self.W_1 = nn.Parameter(torch.normal(0, 0.1, size = (d_1, d_0)), requires_grad=True)
        self.U_1 = nn.Parameter(torch.normal(0, 0.1, size = (d_0, d_1)), requires_grad=True)
        self.W_2 = nn.Parameter(torch.normal(0, 0.1, size = (d_2, d_1)), requires_grad=True)
        self.U_2 = nn.Parameter(torch.normal(0, 0.1, size = (d_1, d_2)), requires_grad=True)
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.Sigma_1 = (self.sigma_1 **2 * torch.eye(d_1)).to("cuda")
        self.Sigma_2 = (self.sigma_2 **2 * torch.eye(d_2)).to("cuda")
        self.Phi_sqrt = Phi_sqrt
        self.P_A = P_A
        
        self.Z = Z

    def forward(self):
        Sigma_1 = self.Sigma_1
        Sigma_2 = self.Sigma_2
        I_d1 = torch.eye(self.d_1).to(self.W_1.device)
        V_1 = self.W_1 @ self.P_A @ self.Phi_sqrt
        loss = torch.norm(self.U_1 @ V_1 - self.Z, p="fro") ** 2
        loss += self.beta_1 * torch.norm((self.U_2 @ self.W_2 - I_d1 )@ V_1, p="fro") ** 2
        loss += torch.trace(self.U_1.T @ self.U_1 @ Sigma_1)
        loss += self.beta_1 * torch.trace(self.U_2.T @ self.U_2 @ Sigma_2)
        loss += self.beta_1 * torch.trace((self.U_2 @ self.W_2 - I_d1).T @ (self.U_2 @ self.W_2 - I_d1) @ Sigma_1)
        loss += self.beta_2 * self.c**2 * torch.norm(self.W_2 @ V_1, p="fro")**2
        loss += self.beta_2 * self.c**2 * torch.trace(self.W_2.T @ self.W_2 @ Sigma_1)
        loss += self.beta_2 * self.c**2 * torch.trace(Sigma_2)
        loss = loss / (self.eta_dec**2)
        loss += -torch.logdet(Sigma_1)
        loss += -torch.logdet(Sigma_2)
        return loss, V_1, self.U_2

def main(args):
    set_seed(args.seed)
    name = "synthetic_" + str(args.exp_name)\
           + "-" + "beta_1_" + str(args.beta_1) \
           + "-" + "beta_2_" + str(args.beta_2) \
           + "-" + "d_0_" + str(args.d_0) \
           + "-" + "d_1_" + str(args.d_1) \
           + "-" + "epochs_" + str(args.num_iteration) \
           + "-" + "seed_" + str(args.seed)

    X = torch.normal(0, 0.1, size=(args.num_samples, args.d_0)).cuda()
    Y = X.clone().detach().cuda()
    Z, theta, phi, P_A = calculate_Z(X, Y)
    
    Phi_sqrt = torch.diag(torch.sqrt(phi))
    
    
    model = LinearVAE_matrix(d_0=args.d_0, d_1=args.d_1, d_2=args.d_2, exp_name="matrix_latent_2_nonlearnable_sigma",
                             sigma_1=args.sigma_1, sigma_2=args.sigma_2, eta_enc=args.eta_enc, eta_dec=args.eta_dec,
                             beta_1=args.beta_1, beta_2=args.beta_2, Phi_sqrt=Phi_sqrt, P_A=P_A, Z=Z).to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    lambda_array_theory, omega_array_theory, theta_full_eigen_threshold = calculate_lambda_omega_2_linear_nonleanable_isotropic_Sigma_beta(theta_vector=theta.to("cuda"),
                                                                                                          sigma_1= torch.as_tensor(args.sigma_1,  dtype=torch.float, device="cuda"),
                                                                                                          sigma_2= torch.as_tensor(args.sigma_2,  dtype=torch.float, device="cuda"),
                                                                                                          c=torch.as_tensor(model.c,  dtype=torch.float, device="cuda"),
                                                                                                          d_1=args.d_1,
                                                                                                          beta_1=torch.as_tensor(args.beta_1,  dtype=torch.float, device="cuda"),
                                                                                                          beta_2=torch.as_tensor(args.beta_2,  dtype=torch.float, device="cuda"))
    
    pbar = tqdm(range(args.num_iteration))
    for iteration in pbar:
        model.train()
        optimizer.zero_grad()
        loss, V_1, U_2 = model()
        _, lambda_array, _ = torch.linalg.svd(V_1)
        _, omega_array, _ = torch.linalg.svd(U_2)
        loss.backward()

        lambda_metric = calculate_metric(lambda_array_theory.to("cuda"), lambda_array.to("cuda"))
        omega_metric = calculate_metric(omega_array_theory.to("cuda"), omega_array.to("cuda"))
        pbar.set_description("Loss is {:.6f}".format(loss))
        optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--d_0', type=int, default=5)
    parser.add_argument('--d_1', type=int, default=5)
    parser.add_argument('--d_2', type=int, default=5)
    parser.add_argument('--exp_name', type=str, default="matrix_latent_2_nonlearnable_sigma")
    parser.add_argument('--num_iteration', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=str, default=2)
    parser.add_argument('--eta_enc', type=float, default=1)
    parser.add_argument('--eta_dec', type=float, default=1),
    parser.add_argument('--sigma_1', type=float, default=1)
    parser.add_argument('--sigma_2', type=float, default=1)
    parser.add_argument('--beta_1', type=float, default=1)
    parser.add_argument('--beta_2', type=float, default=2)
    args = parser.parse_args()
    main(args)