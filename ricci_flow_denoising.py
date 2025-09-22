import numpy as np
import scipy.ndimage as ndimage
from scipy import sparse
from scipy.sparse.linalg import spsolve
from PIL import Image
import matplotlib.pyplot as plt

class RicciFlowDenoising:
    """
    Implementation of Ricci Flow for Image Denoising based on:
    "Ricci Curvature and Flow for Image Denoising and Super-Resolution"
    by Appleboim, Saucan, and Zeevi (2012)
    """
    
    def __init__(self, beta=1.0, dt=0.01, iterations=3, alpha = 0.3):
        """
        Parameters:
        -----------
        beta : float
            Parameter that scales differential change (controls edge sensitivity)
        dt : float
            Time step for the flow evolution
        iterations : int
            Number of flow iterations
        alpha : float
            Blending factor
        """
        self.beta = beta
        self.dt = dt
        self.iterations = iterations
        self.alpha = alpha
        
    def compute_geometric_weights(self, image):
        """
        Compute geometric weights based on image gradients.
        
        The metric of a gray level image is:
        G = [[β + Ix², IxIy], [IyIx, β + Iy²]]
        
        Weights are: w(ex) = sqrt(β + Ix²)dx, w(ey) = sqrt(β + Iy²)dy
        """
        # Compute gradients
        grad_x = np.gradient(image, axis=1)
        grad_y = np.gradient(image, axis=0)
        
        # Compute edge weights (lengths in the metric)
        w_ex = np.sqrt(self.beta + grad_x**2)
        w_ey = np.sqrt(self.beta + grad_y**2)
        
        # Compute cell weights (areas)
        # First order approximation: dA = ds(ex) * ds(ey)
        w_cells = w_ex * w_ey
        
        return w_ex, w_ey, w_cells, grad_x, grad_y
    
    def compute_ricci_curvature(self, image):
        """
        Compute Forman's combinatorial Ricci curvature for each edge.
        
        Ric(e0) = w(e0)[w(e0)/w(c1) + w(e0)/w(c2)] - 
                  [sqrt(w(e0)w(e1))/w(c1) + sqrt(w(e0)w(e2))/w(c2)]
        
        Where:
        - e0 is the edge of interest
        - c1, c2 are adjacent cells
        - e1, e2 are edges adjacent to e0
        """
        h, w = image.shape
        w_ex, w_ey, w_cells, _, _ = self.compute_geometric_weights(image)
        
        # Initialize Ricci curvature arrays for horizontal and vertical edges
        ric_x = np.zeros((h, w-1))
        ric_y = np.zeros((h-1, w))
        
        # Compute Ricci curvature for horizontal edges
        for i in range(1, h-1):
            for j in range(w-1):
                # Current edge weight
                w_e0 = w_ex[i, j]
                
                # Adjacent cell weights (above and below)
                w_c1 = w_cells[i-1, j] if i > 0 else w_cells[i, j]
                w_c2 = w_cells[i, j]
                
                # Adjacent edge weights
                # Vertical edges connected to this horizontal edge
                w_e1 = w_ey[i-1, j] if i > 0 else w_ey[i, j]
                w_e2 = w_ey[i-1, j+1] if i > 0 else w_ey[i, j+1]
                w_e3 = w_ey[i, j]
                w_e4 = w_ey[i, j+1]
                
                # Compute Ricci curvature
                term1 = w_e0 * (w_e0/max(w_c1, 1e-10) + w_e0/max(w_c2, 1e-10))
                term2 = (np.sqrt(w_e0 * w_e1)/max(w_c1, 1e-10) + 
                        np.sqrt(w_e0 * w_e2)/max(w_c1, 1e-10) +
                        np.sqrt(w_e0 * w_e3)/max(w_c2, 1e-10) + 
                        np.sqrt(w_e0 * w_e4)/max(w_c2, 1e-10))
                
                ric_x[i, j] = term1 - term2
        
        # Compute Ricci curvature for vertical edges
        for i in range(h-1):
            for j in range(1, w-1):
                # Current edge weight
                w_e0 = w_ey[i, j]
                
                # Adjacent cell weights (left and right)
                w_c1 = w_cells[i, j-1] if j > 0 else w_cells[i, j]
                w_c2 = w_cells[i, j]
                
                # Adjacent edge weights
                # Horizontal edges connected to this vertical edge
                w_e1 = w_ex[i, j-1] if j > 0 else w_ex[i, j]
                w_e2 = w_ex[i+1, j-1] if j > 0 else w_ex[i+1, j]
                w_e3 = w_ex[i, j]
                w_e4 = w_ex[i+1, j]
                
                # Compute Ricci curvature
                term1 = w_e0 * (w_e0/max(w_c1, 1e-10) + w_e0/max(w_c2, 1e-10))
                term2 = (np.sqrt(w_e0 * w_e1)/max(w_c1, 1e-10) + 
                        np.sqrt(w_e0 * w_e2)/max(w_c1, 1e-10) +
                        np.sqrt(w_e0 * w_e3)/max(w_c2, 1e-10) + 
                        np.sqrt(w_e0 * w_e4)/max(w_c2, 1e-10))
                
                ric_y[i, j] = term1 - term2
                
        return ric_x, ric_y
    
    def evolve_metric(self, grad_x, grad_y, ric_x, ric_y):
        """
        Evolve the metric according to the Ricci flow:
        ∂G/∂t = -Ric(I)
        
        Since the metric is related to gradients, we evolve the gradient field.
        """
        # Interpolate Ricci curvature to match gradient dimensions
        h, w = grad_x.shape
        
        # For horizontal gradients
        ric_x_interp = np.zeros_like(grad_x)
        ric_x_interp[:, :-1] = ric_x[:, :]
        ric_x_interp[:, -1] = ric_x_interp[:, -2]
        
        # For vertical gradients  
        ric_y_interp = np.zeros_like(grad_y)
        ric_y_interp[:-1, :] = ric_y[:, :]
        ric_y_interp[-1, :] = ric_y_interp[-2, :]
        
        # Update gradients based on Ricci flow
        # The metric evolves as dG/dt = -Ric
        # Since metric ~ gradient^2, we have d(grad)/dt ~ -Ric/(2*grad)
        grad_x_new = grad_x - self.dt * ric_x_interp * grad_x / (2 * (grad_x**2 + self.beta))
        grad_y_new = grad_y - self.dt * ric_y_interp * grad_y / (2 * (grad_y**2 + self.beta))
        
        return grad_x_new, grad_y_new
    
    def gaussian_filter_gradients(self, grad_x, grad_y, sigma=1.0):
        """
        Apply Gaussian filtering to gradient field to reduce artifacts.
        """
        grad_x_filtered = ndimage.gaussian_filter(grad_x, sigma)
        grad_y_filtered = ndimage.gaussian_filter(grad_y, sigma)
        return grad_x_filtered, grad_y_filtered
    
    def poisson_solver(self, grad_x, grad_y):
        """
        Reconstruct image from gradient field using Poisson solver.
        Solves: ∇²u = div(g) where g = (grad_x, grad_y)
        """
        h, w = grad_x.shape
        n = h * w
        
        # Compute divergence of gradient field
        div = np.zeros((h, w))
        
        # div = ∂gx/∂x + ∂gy/∂y
        div[1:-1, 1:-1] = (grad_x[1:-1, 2:] - grad_x[1:-1, :-2])/2 + \
                          (grad_y[2:, 1:-1] - grad_y[:-2, 1:-1])/2
        
        # Build sparse Laplacian matrix
        row = []
        col = []
        data = []
        
        def idx(i, j):
            return i * w + j
        
        for i in range(h):
            for j in range(w):
                current_idx = idx(i, j)
                
                # Diagonal element
                neighbors = 0
                if i > 0: neighbors += 1
                if i < h-1: neighbors += 1
                if j > 0: neighbors += 1
                if j < w-1: neighbors += 1
                
                row.append(current_idx)
                col.append(current_idx)
                data.append(-neighbors)
                
                # Off-diagonal elements
                if i > 0:
                    row.append(current_idx)
                    col.append(idx(i-1, j))
                    data.append(1)
                if i < h-1:
                    row.append(current_idx)
                    col.append(idx(i+1, j))
                    data.append(1)
                if j > 0:
                    row.append(current_idx)
                    col.append(idx(i, j-1))
                    data.append(1)
                if j < w-1:
                    row.append(current_idx)
                    col.append(idx(i, j+1))
                    data.append(1)
        
        # Create sparse matrix
        L = sparse.csr_matrix((data, (row, col)), shape=(n, n))
        
        # Solve Poisson equation
        b = div.flatten()
        u_flat = spsolve(L, b)
        
        # Reshape to image
        u = u_flat.reshape((h, w))
        
        # Normalize to original image range
        u = u - np.min(u)
        u = u / np.max(u) if np.max(u) > 0 else u
        
        return u
    
    def denoise(self, noisy_image):
        """
        Main denoising function using Ricci flow.
        
        Parameters:
        -----------
        noisy_image : np.ndarray
            Input noisy image (grayscale)
            
        Returns:
        --------
        denoised_image : np.ndarray
            Denoised image
        """
        # Store original range
        orig_min, orig_max = noisy_image.min(), noisy_image.max()
        
        # Normalize image to [0, 1]
        image = (noisy_image - orig_min) / (orig_max - orig_min)
        
        # Initialize with the noisy image
        current_image = image.copy()
        
        # Ricci flow iterations
        for iteration in range(self.iterations):
            print(f"Ricci flow iteration {iteration + 1}/{self.iterations}")
            
            # Compute geometric weights and gradients
            w_ex, w_ey, w_cells, grad_x, grad_y = self.compute_geometric_weights(current_image)
            
            # Compute Ricci curvature
            ric_x, ric_y = self.compute_ricci_curvature(current_image)
            
            # Evolve metric (gradient field)
            grad_x_new, grad_y_new = self.evolve_metric(grad_x, grad_y, ric_x, ric_y)
            
            # Apply Gaussian filter to gradients
            grad_x_filtered, grad_y_filtered = self.gaussian_filter_gradients(
                grad_x_new, grad_y_new, sigma=0.5
            )
            
            # Reconstruct image from evolved gradient field
            reconstructed = self.poisson_solver(grad_x_filtered, grad_y_filtered)
            
            # Blend with original for stability (optional)
            current_image = self.alpha * reconstructed + (1 - self.alpha) * current_image
            
            # Preserve DC component (average intensity)
            current_image = current_image - np.mean(current_image) + np.mean(image)
            
            # Clip values to valid range
            current_image = np.clip(current_image, 0, 1)
        
        # Restore original range
        denoised = current_image * (orig_max - orig_min) + orig_min
        
        return denoised


# Example usage and testing
def add_noise(image, noise_level=25):
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, noise_level, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255)

def psnr(original, denoised):
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def demo_ricci_flow_denoising():
    """
    Demonstration of Ricci flow denoising on a test image.
    """
    # Create or load a test image
    # For demo, create a simple synthetic image
    size = 128
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    
    # Create test image with multiple features
    image = np.zeros((size, size))
    # Add circles
    image += 200 * np.exp(-((X-2)**2 + (Y-2)**2) / 2)
    image += 150 * np.exp(-((X+2)**2 + (Y+2)**2) / 1)
    # Add edges
    image[size//3:2*size//3, size//4:size//4+10] = 255
    image[size//4:size//4+10, size//3:2*size//3] = 180
    
    # Add noise
    noisy_image = add_noise(image, noise_level=35)
    
    # Apply Ricci flow denoising
    denoiser = RicciFlowDenoising(beta=1.0, dt=0.01, iterations=3)
    denoised_image = denoiser.denoise(noisy_image)
    
    # Calculate metrics
    psnr_noisy = psnr(image, noisy_image)
    psnr_denoised = psnr(image, denoised_image)
    
    print(f"PSNR (noisy): {psnr_noisy:.2f} dB")
    print(f"PSNR (denoised): {psnr_denoised:.2f} dB")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(noisy_image, cmap='gray')
    axes[1].set_title(f'Noisy Image (PSNR: {psnr_noisy:.2f} dB)')
    axes[1].axis('off')
    
    axes[2].imshow(denoised_image, cmap='gray')
    axes[2].set_title(f'Denoised (Ricci Flow) (PSNR: {psnr_denoised:.2f} dB)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return image, noisy_image, denoised_image

# Additional utility for real image processing
def process_real_image(image_path, noise_level=25):
    """
    Process a real image file.
    
    Parameters:
    -----------
    image_path : str
        Path to input image
    noise_level : float
        Standard deviation of Gaussian noise to add (for testing)
    """
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    image = np.array(img, dtype=np.float64)
    
    # Add noise (for testing - skip if image is already noisy)
    noisy_image = add_noise(image, noise_level)
    
    # Apply Ricci flow denoising
    denoiser = RicciFlowDenoising(beta=1.0, dt=0.01, iterations=3)
    denoised_image = denoiser.denoise(noisy_image)
    
    return image, noisy_image, denoised_image

if __name__ == "__main__":
    # Run demonstration
    print("Running Ricci Flow Denoising Demonstration...")
    original, noisy, denoised = demo_ricci_flow_denoising()
    print("Demonstration complete!")