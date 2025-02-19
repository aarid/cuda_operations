import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure the subplot layout
plt.figure(figsize=(15, 10))

# Dictionary of files and their titles
plots = {
    'matrix_A.csv': 'Matrix A',
    'matrix_B.csv': 'Matrix B',
    'matrix_C.csv': 'Matrix Result',
    'conv_input.csv': 'Conv Input',
    'conv_kernel.csv': 'Conv Kernel',
    'conv_output.csv': 'Conv Output'
}

# Create subplots in a 2x3 grid
for idx, (file, title) in enumerate(plots.items(), 1):
    try:
        # Read data
        data = pd.read_csv(file, header=None)
        
        # Create subplot
        plt.subplot(2, 3, idx)
        sns.heatmap(data, cmap='viridis')
        plt.title(title)
    except FileNotFoundError:
        print(f"Warning: {file} not found")

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig('results_visualization.png')
plt.show()