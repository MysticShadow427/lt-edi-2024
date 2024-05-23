import numpy as np
import pandas as pd
import torch

def random_undersample(df,random_seed = 42,desired_samples = 450):
    np.random.seed(random_seed)
    class_1_df = df[df['label'] == 'None of the categories']
    class_2_df = df[df['label'] == 'Homophobia']
    class_3_df = df[df['label'] == 'Transphobia']

    if len(class_1_df) > desired_samples:
        undersampled_class_1_df = class_1_df.sample(n=desired_samples, replace=True)
    else:
        undersampled_class_1_df = class_1_df.sample(frac=1.0)   

    augmented_df = pd.concat([undersampled_class_1_df, class_2_df,class_3_df])

    augmented_df = augmented_df.sample(frac=1.0).reset_index(drop=True)

    return augmented_df

def augment_embeddings(embeddings, num_views=2, noise_std_range=(0.05, 0.15)):
    batch_size, embedding_dim = embeddings.shape
    augmented_embeddings = torch.zeros(batch_size, num_views + 1, embedding_dim, device=embeddings.device)
    
    # Include the original embeddings
    augmented_embeddings[:, 0, :] = embeddings
    
    for i in range(1, num_views + 1):
        # Randomly select noise standard deviation
        noise_std = torch.empty(1).uniform_(*noise_std_range).item()
        
        # Add Gaussian noise to the embeddings
        noise = torch.randn_like(embeddings) * noise_std
        augmented = embeddings + noise
        
        # Apply a random rotation matrix
        rotation_matrix = torch.randn(embedding_dim, embedding_dim).to(embeddings.device)
        rotation_matrix, _ = torch.linalg.qr(rotation_matrix)  # Ensure the matrix has orthogonal columns
        
        # Apply the rotation matrix to each embedding vector
        augmented = torch.matmul(augmented, rotation_matrix)
        
        # Store the augmented embeddings
        augmented_embeddings[:, i, :] = augmented
    
    return augmented_embeddings

