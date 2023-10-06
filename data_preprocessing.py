import torch


def whitening_matrix(X):
    A = torch.einsum('na, nb -> ab', X, X) / X.shape[0]
    phi, P_A = torch.linalg.eigh(A.to("cuda"))
    phi = phi.to(X.device)
    P_A = P_A.to(X.device)
    phi = torch.clamp(phi, min=0)
    Phi_sqrt_inv = torch.diag(1. / (torch.sqrt(phi + 1e-9)))
    X_tilde = (Phi_sqrt_inv @ P_A.T @ X.T).T
    return X_tilde, phi, P_A


def calculate_Z(X, Y):
    X_tilde, phi, P_A = whitening_matrix(X)
    Z = torch.einsum('na, nb -> ab', Y, X_tilde) / X.shape[0]
    _, theta, _ = torch.linalg.svd(Z.to("cuda"))
    return Z, theta.to(X.device), phi, P_A


def calculate_ZDE_CVAE(X, Y):
    I_dimZ = torch.eye(Y.shape[1]).to(Y.device)
    X_tilde, phi, P_A = whitening_matrix(X)
    Y_tilde, psi, P_B = whitening_matrix(Y)
    Z = torch.einsum('na, nb -> ab', X_tilde, Y_tilde) / X.shape[0]
    D = P_B @ torch.diag((psi) ** (1/2))
    E = D @ (I_dimZ - Z.T @ Z) @ D.T
    return phi, P_A, psi, P_B, Z, D, E
