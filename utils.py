import os
import random
import torch
import numpy as np


def set_seed(manualSeed=666):
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(manualSeed)

def calculate_omega_lambda_1_linear_leanable_Sigma(theta_vector, eta_enc, eta_dec, d_1, beta):
    theta_vector = theta_vector[:d_1]
    sigma_array = []
    omega_array = 1 / eta_enc * torch.sqrt(torch.clamp((theta_vector ** 2) - beta * (eta_dec ** 2), min=0))
    lambda_array = eta_enc / theta_vector * torch.sqrt(torch.clamp((theta_vector ** 2) - beta * (eta_dec ** 2), min=torch.tensor(0.0)))
    for theta in theta_vector:
        if theta >= (beta ** (1/2)) * eta_dec:
            sigma_array.append((beta ** (1/2)) * eta_enc * eta_dec / theta)
        elif theta < (beta ** (1/2)) * eta_dec:
            sigma_array.append(torch.tensor(eta_enc))
        else:
            raise Exception
    sigma_array = torch.stack(sigma_array).squeeze()

    return omega_array.to("cuda"), lambda_array.to("cuda"), sigma_array.to("cuda")

def calculate_omega_lambda_1_linear_nonlearnable_Sigma(theta_vector, sigma_array, eta_enc, eta_dec, d_1, beta):
    theta_vector = theta_vector[:d_1].to("cuda")
    sigma_array_sorted = torch.sort(sigma_array)[0].to("cuda")
    c = eta_dec / eta_enc
    omega_array = torch.sqrt(torch.clamp(((beta ** (1/2)) * c * (theta_vector - 
                                            ((beta) ** (1/2)) * c * sigma_array_sorted)) / sigma_array_sorted, min=0))
    lambda_array = torch.sqrt(torch.clamp(sigma_array_sorted * (theta_vector - 
                                                    ((beta) ** (1/2)) * c * sigma_array_sorted) / ((beta ** (1/2)) * c), min=0))
    return omega_array, lambda_array

def calculate_lambda_omega_2_linear_nonleanable_isotropic_Sigma(theta_vector, sigma_1, sigma_2, c, d_1):
    lambda_array = []
    omega_array = []
    theta_vector = theta_vector[:d_1]
    for theta in theta_vector:
        if theta >= (c*sigma_1*sigma_2) ** (1/2) and theta >= c**2 * sigma_2**2 / sigma_1:
            lambda_element = ((sigma_1 ** 2) / (c*sigma_2)) ** (1/3) 
            lambda_element *= (theta ** (4/3) - (c*sigma_1*sigma_2) ** (2/3)) ** (1/2)
            lambda_array.append(lambda_element)

            omega_element = (c*sigma_1 / (sigma_2 ** 2)) ** (1/3)
            omega_element *= (theta ** (2/3) - 
                              ((c ** 4 * sigma_2 ** 4) / (sigma_1 ** 2)) ** (1/3)) ** (1/2)
            omega_array.append(omega_element)
        elif sigma_1 >= c*sigma_2:
            lambda_array.append(torch.tensor(0.0))
            omega_array.append((c * sigma_1 / sigma_2 - c ** 2) ** (1/2) )
        elif theta < sigma_1 and sigma_1 < c*sigma_2:
            lambda_array.append(torch.tensor(0.0))
            omega_array.append(torch.tensor(0.0))
        elif theta >= sigma_1 and sigma_1 < c * sigma_2:
            lambda_array.append((sigma_1 * (theta - sigma_1)) ** (1/2))
            omega_array.append(torch.tensor(0.0))
        else:
            raise Exception

    lambda_array = torch.stack(lambda_array).squeeze()
    omega_array = torch.stack(omega_array).squeeze()
    return lambda_array.to("cuda"), omega_array.to("cuda")

def calculate_lambda_omega_2_linear_nonleanable_isotropic_Sigma_beta(theta_vector, sigma_1, sigma_2, 
                                                                     c, d_1, beta_1, beta_2):
    lambda_array = []
    omega_array = []
    theta_vector = theta_vector[:d_1]
    theta_full_eigen_threshold = [((beta_1 * beta_2) ** (1/2) * c*sigma_1*sigma_2) ** (1/2)]
    theta_full_eigen_threshold.append(((beta_2 ** 2) * beta_1) ** (1/2) * c**2 * sigma_2**2 / sigma_1)
    theta_full_eigen_threshold = torch.stack(theta_full_eigen_threshold).max()
    for idx, theta in enumerate(theta_vector):
        if ( theta >= ((beta_1 * beta_2) ** (1/2) * c*sigma_1*sigma_2) ** (1/2)
            and theta >= ((beta_2 ** 2) * beta_1) ** (1/2) * c**2 * sigma_2**2 / sigma_1 ):
            lambda_element = ((sigma_1 ** 2) / ((beta_1 * beta_2) ** (1/2) * c*sigma_2)) ** (1/3) 
            lambda_element *= (theta ** (4/3) - ((beta_1 * beta_2) ** (1/2) * c*sigma_1*sigma_2) ** (2/3)) ** (1/2)
            lambda_array.append(lambda_element)

            omega_element = (beta_2 ** (1/2) * c * sigma_1 / (beta_1 * sigma_2 ** 2)) ** (1/3)
            omega_element *= (theta ** (2/3) - 
                              ((beta_2 ** 2 * c ** 4 * sigma_2 ** 4) / (beta_1 * sigma_1 ** 2)) ** (1/3)) ** (1/2)
            omega_array.append(omega_element)
        elif beta_1 ** (1/2) * sigma_1 >= beta_2 ** (1/2) * c*sigma_2:
            lambda_array.append(torch.tensor(0.0))
            omega_element = ((beta_2 ** (1/2) * c * sigma_1) / (beta_1 ** (1/2) * sigma_2) - 
                                          (beta_2 / beta_1) * c ** 2) ** (1/2)
            omega_array.append(omega_element)
        elif theta < beta_1 ** (1/2) * sigma_1 and beta_1 ** (1/2) * sigma_1 < beta_2 ** (1/2) * c*sigma_2:
            lambda_array.append(torch.tensor(0.0))
            omega_array.append(torch.tensor(0.0))
        elif theta >= beta_1 ** (1/2) * sigma_1 and beta_1 ** (1/2) * sigma_1 < beta_2 ** (1/2) * c*sigma_2:
            lambda_array.append((sigma_1 * (theta / (beta_1 ** (1/2)) - sigma_1)) ** (1/2))
            omega_array.append(torch.tensor(0.0))
        else:
            raise Exception

    lambda_array = torch.stack(lambda_array).squeeze()
    omega_array = torch.stack(omega_array).squeeze()
    return lambda_array.to("cuda"), omega_array.to("cuda"), theta_full_eigen_threshold

def calculate_lambda_omega_2_linear_nonlearnable_isotropic_Sigma_1_learnable_Sigma_2(theta_vector, sigma_1, eta_enc, eta_dec, c, d_1):
    lambda_array = []
    omega_array = []
    theta_vector = theta_vector[:d_1]
    theta_full_eigen_threshold = [eta_dec]
    theta_full_eigen_threshold.append(eta_dec**2 / sigma_1)
    theta_full_eigen_threshold = torch.stack(theta_full_eigen_threshold).max()
    for idx, theta in enumerate(theta_vector):
        if sigma_1**(2) >= eta_dec**(2):
            lambda_element = (sigma_1 / eta_dec) * (torch.max(torch.tensor(0.0, device="cuda"), theta**2 - eta_dec**2)) ** (1/2)
            omega_element = torch.max((sigma_1**2 - eta_dec**2) / (eta_enc**2), 
                                        (sigma_1**2 * theta**2) / (eta_enc**2 * eta_dec**2) - c**2) ** (1/2)
            
        elif (sigma_1**2 * theta**2) / (eta_dec**2) < eta_dec**2:
            lambda_element = (torch.max(torch.tensor(0.0, device="cuda"), sigma_1 * (theta - sigma_1))) ** (1/2)
            omega_element = torch.tensor(0.0, device="cuda")
        elif (sigma_1**2 * theta**2) / (eta_dec**2) >= eta_dec**2:
            lambda_element = (sigma_1 / eta_dec) * (theta**2 - eta_dec**2) ** (1/2)
            omega_element = ((sigma_1**2 * theta**2)/(eta_enc**2 * eta_dec**2) - c**2) ** (1/2)
        else:
            raise Exception
        lambda_array.append(lambda_element)
        omega_array.append(omega_element)
    lambda_array = torch.stack(lambda_array).squeeze()
    omega_array = torch.stack(omega_array).squeeze()
    return lambda_array.to("cuda"), omega_array.to("cuda"), theta_full_eigen_threshold

def calculate_lambda_omega_2_linear_nonlearnable_isotropic_Sigma_1_learnable_Sigma_2_beta(theta_vector, sigma_1, eta_enc, eta_dec, c, d_1, beta_1, beta_2):
    lambda_array = []
    omega_array = []
    theta_vector = theta_vector[:d_1]
    theta_full_eigen_threshold = [eta_dec]
    theta_full_eigen_threshold.append(eta_dec**2 / sigma_1)
    theta_full_eigen_threshold = torch.stack(theta_full_eigen_threshold).max()
    for idx, theta in enumerate(theta_vector):
        if sigma_1**(2) >= (beta_2 / beta_1) * eta_dec**(2):
            lambda_element = (sigma_1 / (beta_2**(1/2) * eta_dec)) * (torch.max(torch.tensor(0.0, device="cuda"), theta**2 - beta_2 * eta_dec**2)) ** (1/2)
            omega_element = torch.max((sigma_1**2 - (beta_2 / beta_1) * eta_dec**2) / (eta_enc**2), 
                                        (sigma_1**2 * theta**2) / (beta_2 * eta_enc**2 * eta_dec**2) - c**2 * (beta_2 / beta_1)) ** (1/2)
            
        elif (sigma_1**2 * theta**2) / (beta_2 * eta_dec**2) < (beta_2 / beta_1) * eta_dec**2:
            lambda_element = (torch.max(torch.tensor(0.0, device="cuda"), (sigma_1 / beta_1**(1/2)) * (theta - beta_1**(1/2) * sigma_1))) ** (1/2)
            omega_element = torch.tensor(0.0, device="cuda")
        elif (sigma_1**2 * theta**2) / (beta_2 * eta_dec**2) >= (beta_2 / beta_1) * eta_dec**2:
            lambda_element = (sigma_1 / ((beta_2)**(1/2) * eta_dec)) * (theta**2 - (beta_2) * eta_dec**2) ** (1/2)
            omega_element = ((sigma_1**2 * theta**2)/(beta_2 * eta_enc**2 * eta_dec**2) - (beta_2 / beta_1) * c**2) ** (1/2)
        else:
            raise Exception
        lambda_array.append(lambda_element)
        omega_array.append(omega_element)
    lambda_array = torch.stack(lambda_array).squeeze()
    omega_array = torch.stack(omega_array).squeeze()
    return lambda_array.to("cuda"), omega_array.to("cuda"), theta_full_eigen_threshold
        
def calculate_lambda_sigma_cvae_beta(theta_vector, eta_enc, eta_dec, dim_z, beta):
    theta_vector = theta_vector[:dim_z]
    lambda_array = []
    sigma_array = []
    for idx, theta in enumerate(theta_vector):
        lambda_element = (1 / eta_enc) * torch.max(torch.tensor(0.0, device="cuda"),
                                                   theta - beta * eta_dec**2) ** (1/2)
        if theta >= beta**(1/2) * eta_dec:
            sigma_element = (beta**(1/2) * eta_enc * eta_dec) / (theta ** (1/2))
        else:
            sigma_element = eta_enc
        lambda_array.append(lambda_element)
        sigma_array.append(sigma_element)
    lambda_array = torch.stack(lambda_array).squeeze()
    sigma_array = torch.stack(sigma_array).squeeze()
    return lambda_array.sort(descending=True)[0], sigma_array.sort(descending=True)[0]

def calculate_metric(theory_array, empirical_array):
    return torch.abs(empirical_array - theory_array).mean()

def calculate_trace_3_matrices(XT, X, Sigma):
    if len(X.shape) == 3:
        return torch.einsum('bij, bji, bii -> b', XT, X, Sigma)
    elif len(X.shape) == 2:
        return torch.einsum('bij, bji, bii -> b', XT.unsqueeze(0), X.unsqueeze(0), Sigma.unsqueeze(0)).squeeze()
    else:
        raise Exception