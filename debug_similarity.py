import numpy as np
import pandas as pd
import os

# Load the content similarity matrix
content_sim_matrix = np.load('data/processed/content_sim_matrix.npy')
ratings_df = pd.read_csv('data/processed/ratings.csv')
books_df = pd.read_csv('data/processed/books_clean.csv')

print("Content Similarity Matrix Stats:")
print(f"Shape: {content_sim_matrix.shape}")
print(f"Min value: {np.min(content_sim_matrix):.6f}")
print(f"Max value: {np.max(content_sim_matrix):.6f}")
print(f"Mean value: {np.mean(content_sim_matrix):.6f}")
print(f"Std value: {np.std(content_sim_matrix):.6f}")

# Check a few sample values
print(f"\nSample values:")
print(f"First 5x5 block:\n{content_sim_matrix[:5, :5]}")

# Check if there are any negative values
negative_count = np.sum(content_sim_matrix < 0)
print(f"\nNegative values count: {negative_count}")

# Check if there are any values > 1
over_one_count = np.sum(content_sim_matrix > 1)
print(f"Values > 1 count: {over_one_count}")

# Test the scaling issue
print(f"\nScaling test:")
test_similarity = 0.5
old_pred = test_similarity * 5.0  # Current (wrong) method
new_pred = 1.0 + test_similarity * 4.0  # Fixed method
print(f"Similarity: {test_similarity}")
print(f"Old prediction (similarity * 5): {old_pred}")
print(f"New prediction (1 + similarity * 4): {new_pred}")

# Check actual similarity range in the matrix
print(f"\nActual similarity range analysis:")
print(f"Values in [0, 0.1]: {np.sum((content_sim_matrix >= 0) & (content_sim_matrix <= 0.1))}")
print(f"Values in [0.1, 0.5]: {np.sum((content_sim_matrix > 0.1) & (content_sim_matrix <= 0.5))}")
print(f"Values in [0.5, 0.9]: {np.sum((content_sim_matrix > 0.5) & (content_sim_matrix <= 0.9))}")
print(f"Values in [0.9, 1.0]: {np.sum((content_sim_matrix > 0.9) & (content_sim_matrix <= 1.0))}")
print(f"Values > 1.0: {np.sum(content_sim_matrix > 1.0)}")
