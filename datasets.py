import torch
from torch.utils.data import Dataset
from data_preprocessing import calculate_Z, calculate_ZDE_CVAE, whitening_matrix
from torchvision.datasets import MNIST
import math

class SyntheticDataset(Dataset):
    def __init__(self, num_samples, dim) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.dim = dim
        self.X = torch.normal(0, 1, size=(num_samples, dim))
        self.Y = self.X.clone().detach()
        self.Z, self.theta, self.phi, self.P_A = calculate_Z(self.X, self.Y)

    def __len__(self):
        return(self.num_samples)

    def __getitem__(self, index):
        return self.X[index, :], self.Y[index, :]

class SyntheticCVAEDataset(Dataset):
    def __init__(self, num_samples, corr_type, dim) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.dim = dim
        self.Y = torch.normal(0, 1, size=(num_samples, dim))
        if corr_type == "Identical":
            self.X = self.Y.clone().detach()
        elif corr_type == "Gaussian_noise_1_8":
            Z = torch.normal(0, 1, size=(num_samples, dim))
            self.X = 1/8 * self.Y.clone().detach() + math.sqrt(63) / 8 * Z 
        elif corr_type == "Gaussian_noise_1_4":
            Z = torch.normal(0, 1, size=(num_samples, dim))
            self.X = 1/4 * self.Y.clone().detach() + math.sqrt(15) / 4 * Z 
        elif corr_type == "Gaussian_noise_1_2":
            Z = torch.normal(0, 1, size=(num_samples, dim))
            self.X = 1/2 * self.Y.clone().detach() + math.sqrt(3) / 2 * Z 
        elif corr_type == "Quadratic":
            self.X = (self.Y.clone().detach() ** 2) - 1
        elif corr_type == "Random":
            self.X = torch.normal(0, 1, size=(num_samples, dim))
        self.phi, self.P_A, self.psi, self.P_B, self.Z, self.D, self.E = calculate_ZDE_CVAE(self.X, self.Y)

    def __len__(self):
        return(self.num_samples)

    def __getitem__(self, index):
        return self.X[index, :], self.Y[index, :]


class MNISTDataset(MNIST):
    def __init__(self, root, train) -> None:
        super(MNISTDataset, self).__init__(root, train, download=True)
        self.num_samples = len(self.data)
        data = self.data.float() / 255.0
        self.X = data.reshape(self.num_samples, -1)
        self.Y = self.X
        self.Z, self.theta, self.phi, self.P_A = calculate_Z(self.X, self.Y)

    def __getitem__(self, index):
        return self.X[index, :], self.Y[index, :], self.train_labels[index]


class MNISTDatasetCVEHalf(MNIST):
    def __init__(self, root, train) -> None:
        super(MNISTDatasetCVEHalf, self).__init__(root, train, download=True)
        self.num_samples = len(self.data)
        self.width = 28
        self.height = 28
        self.data = self.data.float() / 255.0
        self.X, self.Y = self.mask_img()
        self.phi, self.P_A, self.psi, self.P_B, self.Z, self.D, self.E = calculate_ZDE_CVAE(self.X, self.Y)
    
    def __getitem__(self, index):
        return self.X[index, :], self.Y[index, :], self.train_labels[index]

    def mask_img(self):
        X = self.data[:, :, :self.width // 2].clone().detach()
        Y = self.data[:, :, self.width // 2:].clone().detach()
        return X.reshape(self.num_samples, -1), Y.reshape(self.num_samples, -1)


class MNISTDatasetCVEQuarter(MNIST):
    def __init__(self, root, train) -> None:
        super(MNISTDatasetCVEQuarter, self).__init__(root, train, download=True)
        self.num_samples = len(self.data)
        self.width = 28
        self.height = 28
        self.quarter_dim = self.width * self.height // 4
        self.data = self.data.float() / 255.0
        self.X, self.Y = self.mask_img()
        self.phi, self.P_A, self.psi, self.P_B, self.Z, self.D, self.E = calculate_ZDE_CVAE(self.X, self.Y)

    def __getitem__(self, index):
        return self.X[index, :], self.Y[index, :], self.train_labels[index]

    def mask_img(self):
        quarter_1 = self.data[:, :self.width // 2, :self.height // 2].reshape(self.num_samples, -1)
        quarter_2 = self.data[:, :self.width // 2, self.height // 2:].reshape(self.num_samples, -1)
        quarter_3 = self.data[:, self.width // 2:, self.height // 2:].reshape(self.num_samples, -1)
        quarter_4 = self.data[:, self.width // 2:, :self.height // 2].reshape(self.num_samples, -1)

        X = quarter_4
        Y = torch.cat([quarter_1, quarter_2, quarter_3], dim=1)
        return X, Y


class MNISTDatasetCVEQuarterDigit(MNIST):
    def __init__(self, root, train, digit) -> None:
        super(MNISTDatasetCVEQuarterDigit, self).__init__(root, train, download=True)
        self.width = 28
        self.height = 28
        self.quarter_dim = self.width * self.height // 4

        idx = (self.train_labels == digit)
        self.data_digit = self.data[idx].float() / 255.0
        self.train_labels_digit = self.train_labels[idx]
        self.num_samples = len(self.data_digit)

        self.X, self.Y = self.mask_img(self.data_digit)
        self.phi, self.P_A, self.psi, self.P_B, self.Z, self.D, self.E = calculate_ZDE_CVAE(self.X, self.Y)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index):
        return self.X[index, :], self.Y[index, :], self.train_labels_digit[index]

    def mask_img(self, data_digit):
        quarter_1 = data_digit[:, :self.width // 2, :self.height // 2].reshape(self.num_samples, -1)
        quarter_2 = data_digit[:, :self.width // 2, self.height // 2:].reshape(self.num_samples, -1)
        quarter_3 = data_digit[:, self.width // 2:, self.height // 2:].reshape(self.num_samples, -1)
        quarter_4 = data_digit[:, self.width // 2:, :self.height // 2].reshape(self.num_samples, -1)

        X = quarter_4
        Y = torch.cat([quarter_1, quarter_2, quarter_3], dim=1)
        return X, Y


class MNISTDatasetCVEQuarterDigitCNN(MNIST):
    def __init__(self, root, train, digit) -> None:
        super(MNISTDatasetCVEQuarterDigitCNN, self).__init__(root, train, download=True)
        self.width = 28
        self.height = 28
        self.quarter_dim = self.width * self.height // 4

        idx = (self.train_labels == digit)
        self.data_digit = self.data[idx].float() / 255.0
        self.train_labels_digit = self.train_labels[idx]
        self.num_samples = len(self.data_digit)

        self.X, self.Y, self.X_square, self.Y_square = self.mask_img(self.data_digit)
        self.phi, self.P_A, self.psi, self.P_B, self.Z, self.D, self.E = calculate_ZDE_CVAE(self.X, self.Y)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index):
        return self.X[index, :], self.Y[index, :], self.X_square[index], self.Y_square[index], self.train_labels_digit[index]

    def mask_img(self, data_digit):
        quarter_1 = data_digit[:, :self.width // 2, :self.height // 2].reshape(self.num_samples, -1)
        quarter_2 = data_digit[:, :self.width // 2, self.height // 2:].reshape(self.num_samples, -1)
        quarter_3 = data_digit[:, self.width // 2:, self.height // 2:].reshape(self.num_samples, -1)
        quarter_4 = data_digit[:, self.width // 2:, :self.height // 2].reshape(self.num_samples, -1)

        X = quarter_4
        Y = torch.cat([quarter_1, quarter_2, quarter_3], dim=1)

        X_square = data_digit[:, self.width // 2:, :self.height // 2].unsqueeze(1).clone()
        
        Y_square = data_digit.clone()
        Y_square[:, self.width // 2:, :self.height // 2] = 0
        Y_square = Y_square.unsqueeze(1)
        return X, Y, X_square, Y_square


class MNISTDatasetCVEHalfDigit(MNIST):
    def __init__(self, root, train, digit) -> None:
        super(MNISTDatasetCVEHalfDigit, self).__init__(root, train, download=True)
        self.width = 28
        self.height = 28
        self.quarter_dim = self.width * self.height // 4

        idx = (self.train_labels == digit)
        self.data_digit = self.data[idx].float() / 255.0
        self.train_labels_digit = self.train_labels[idx]
        self.num_samples = len(self.data_digit)

        self.X, self.Y = self.mask_img(self.data_digit)
        self.phi, self.P_A, self.psi, self.P_B, self.Z, self.D, self.E = calculate_ZDE_CVAE(self.X, self.Y)
        _, self.theta, _ = torch.linalg.svd(self.E.to("cuda"))
        self.theta_max = self.theta.max()
        self.theta_sum = self.theta.sum()
        self.trace_E = torch.trace(self.E)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index):
        return self.X[index, :], self.Y[index, :], self.train_labels_digit[index]

    def mask_img(self, data_digit):
        X = data_digit[:, :self.width // 2, :]
        Y = data_digit[:, self.width // 2:, :]
        return X.reshape(self.num_samples, -1), Y.reshape(self.num_samples, -1)