# The original SNP encoder, which feeds founder codes to a PCA as plain
# numbers 1-8.
#
# That treats founder 8 as eight times founder 1, which is meaningless for
# what are really unordered category labels. snp_encoding.py replaces it and
# is what the current model uses; this stays for loading older checkpoints.

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class SNPEncoder(nn.Module):

    def __init__(self, 
                 num_snps,              # Number of SNPs
                 embedding_dim=512,     # Output embedding dimension, matches diffusion model's cross-attention dimension size
                 num_tokens=8,          # Number of tokens for cross-attention
                 pca_components=None,   # None = no PCA, int = number of components
                 hidden_dim=1024):      # Hidden layer size (the higher, the more expressive the model, but also more parameters)
        
        super().__init__()
        self.num_snps = num_snps
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.pca_components = pca_components
        
        # Determine input dimension to the neural network
        # PCA reduces the input dimension if specified, otherwise use the original number of SNPs
        input_dim = num_snps if pca_components is None else pca_components
        
        # Neural network layers: compresses input to token embeddings
        self.net = nn.Sequential(
            # Transform input to hidden dimension
            nn.Linear(input_dim, hidden_dim),
            # Normalization of the layer so that the output has zero mean and unit variance
            nn.LayerNorm(hidden_dim),
            # Activation function, SiLU is smooth which allows it to be differentiable and helps with gradient flow
            # It allows the network to grow even with large positive values, which allows the network to represent stronger relationships in the SNP data
            nn.SiLU(),
            # Randomly sets 10% of neurons to zero during training to prevent overfitting
            nn.Dropout(0.1),
            
            # Further compress to half the hidden dimension
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            
            # Projects the final # of dimensions into the number of tokens times the embedding dimension, which is the final output shape
            nn.Linear(hidden_dim // 2, num_tokens * embedding_dim),
        )
        
        # torch.randn creates a tensor with random values from a normal distribution, and multiplying by 0.02 scales the values to be small, which is a common practice for initializing embeddings
        # So it creates [num_tokens] with each token having a vector of size [embedding_dim], and the values are initialized to small random numbers
        # This makes it so that each token has a unique position in the embedding space, which helps the model learn to differentiate between tokens during training
        self.token_positions = nn.Parameter(torch.randn(1, num_tokens, embedding_dim) * 0.02)
        
        # PCA preprocessing
        self.pca = None
        self.use_pca = pca_components is not None
        
    def set_pca(self, pca_model):
        self.pca = pca_model
        self.use_pca = True
        
    def forward(self, snp_vector):

        # Ensure float type
        if snp_vector.dtype != torch.float32:
            snp_vector = snp_vector.float()
        
        # Apply PCA if enabled and model is available
        if self.use_pca and self.pca is not None:
            # Convert to numpy for PCA transform
            snp_np = snp_vector.cpu().numpy()
            
            # Handle single sample case
            if len(snp_np.shape) == 1:
                snp_np = snp_np.reshape(1, -1)
            
            # PCA reduces the # of dimensions of the SNP vector while preserving the most important variance in the data
            snp_pca = self.pca.transform(snp_np)
            
            # Convert back to tensor
            snp_vector = torch.tensor(
                snp_pca, 
                dtype=torch.float32, 
                device=snp_vector.device
            )
        
        # Pass through neural network
        x = self.net(snp_vector)
        
        # Reshape to token format
        x = x.view(-1, self.num_tokens, self.embedding_dim)
        
        # Add positional embeddings
        x = x + self.token_positions
        
        return x


def fit_pca_from_snp(snp_matrix, n_components=100):

    # Fit PCA with specified number of components (will be optimized with build_snp_encoder_with_optimal_pca before training)
    pca = PCA(n_components=n_components)
    pca.fit(snp_matrix)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_.sum()
    
    info = {
        'n_components': n_components,
        'explained_variance': explained_variance,
        # Calculates the number of components needed to reach 95% explained variance
        'components_needed_95': find_components_for_variance(pca, 0.95),
    }
    
    print(f"PCA fitted: {n_components} components")
    print(f"Explained variance: {explained_variance:.2%}")
    print(f"Components needed for 95%: {info['components_needed_95']}")
    
    return pca, info

# Helper function to find the number of PCA components needed to reach a target explained variance
def find_components_for_variance(pca, target_variance=0.95):
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return np.argmax(cumulative >= target_variance) + 1


def build_snp_encoder_with_optimal_pca(snp_matrix, 
                                       target_variance=0.95,
                                       embedding_dim=512,
                                       num_tokens=8,
                                       hidden_dim=1024):

    # Finds the max number of components possible (the smaller dimension of the SNP matrix)
    max_components = min(snp_matrix.shape)
    # These numbers are tested to see if it reaches the target variance, and the first one that does is used as the number of components for PCA
    # Since the difference between incremented components in PCA aren't super significant and is computationally expensive, we test a few numbers to find the optimal number of components that reaches the target variance
    test_components = [10, 25, 50, 75, 100, 150, 200, 250]
    test_components = [c for c in test_components if c <= max_components]
    
    # If max_components is small, use it directly
    if max_components <= 10:
        n_components = max_components
    else:
        # Find first component count that reaches target variance
        n_components = max_components
        for n in test_components:
            pca_test = PCA(n_components=n)
            pca_test.fit(snp_matrix)
            if pca_test.explained_variance_ratio_.sum() >= target_variance:
                n_components = n
                break
    
    # Fit final PCA
    pca = PCA(n_components=n_components)
    pca.fit(snp_matrix)
    
    print(f"Optimal PCA components: {n_components}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Create encoder
    encoder = SNPEncoder(
        num_snps=snp_matrix.shape[1],
        embedding_dim=embedding_dim,
        num_tokens=num_tokens,
        pca_components=n_components,
        hidden_dim=hidden_dim,
    )
    encoder.set_pca(pca)
    
    return encoder, pca, n_components

def load_snp_data_from_parquet(parquet_path):
    
    # Load the data
    df = pd.read_parquet(parquet_path)

    df['ID'] = df['ID'].str.replace('_TC', '')
    
    print(f"Loaded {len(df)} SNP records")
    print(f"Unique samples: {df['ID'].nunique()}")
    print(f"Unique SNPs: {df['gene_model'].nunique()}")
    
    # ID = MEMA... gene_model = SNP name, Value = SNP value
    snp_matrix = df.pivot(index='ID', columns='gene_model', values='value')
    
    # Get sample names and SNP names
    sample_names = snp_matrix.index.tolist()
    snp_names = snp_matrix.columns.tolist()
    
    # Convert to numpy array
    snp_array = snp_matrix.values.astype(np.float32)
    
    # Replace any missing values (NaN) with -1
    snp_array = np.nan_to_num(snp_array, nan=-1.0)
    
    return sample_names, snp_names, snp_array
