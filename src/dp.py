import math
import torch

def compute_sigma(eps, delta, c2, batch_size, total_samples, num_iterations, rho=1.0):
    """
    Compute the noise scale sigma for DP-LoRA.
    
    Args:
        eps (float): Privacy budget (epsilon).
        delta (float): Privacy relaxation parameter.
        c2 (float): Privacy constant (e.g., 1.1).
        batch_size (int): Batch size used in training.
        total_samples (int): Total number of samples in the dataset.
        num_iterations (int): Total number of iterations (epochs * steps per epoch).
        rho (float): Weighted factor for nodes. Default is 1.0 (single node case).
        
    Returns:
        float: Computed noise scale sigma.
    """

    q = batch_size / total_samples
    
    numerator = c2 * q * math.sqrt(num_iterations * math.log(1 / delta))
    sigma = numerator / (rho * eps)

    return sigma

def add_noise_to_model(model, sigma, learning_rate, clipping_norm=1.0):
    """
    Add Gaussian noise directly to model weights after training.
    Args:
        model (torch.nn.Module): The model containing weights.
        sigma (float): Noise scale computed for gradients.
        learning_rate (float): Learning rate used during training.
        clipping_norm (float): Gradient clipping norm.
    """
    weight_noise_scale = learning_rate * sigma * clipping_norm
    for name, param in model.named_parameters():
        if param.requires_grad:  # LoRA parameters are trainable
            noise = torch.normal(0, weight_noise_scale, size=param.data.shape, device=param.device)
            param.data += noise
            print(f"Added noise to {name} with scale={weight_noise_scale:.4f}")
