# %%
from EMPRwithfourbackends import NDEMPRCalculator
import torch
import numpy as np

backends = ['tensorflow', 'pytorch', 'numpy']

# %%
torch.manual_seed(0)
G = torch.rand(500,500,500)

# %%
# PyTorch 

empr_torch = NDEMPRCalculator(G, supports = 'das', backend='pytorch')
torch_times = {
                "approximation_time": empr_torch.approximation_time,
                "execution_time": empr_torch.excution_time,
              }

# TensorFlow

G = G.numpy()

empr_tf = NDEMPRCalculator(G, supports = 'das', backend='tensorflow')
tf_times    = {
                "approximation_time": empr_tf.approximation_time,
                "execution_time": empr_tf.excution_time,
              }

# Numpy

empr_np = NDEMPRCalculator(G, supports = 'das', backend='numpy')
np_times    = {
                "approximation_time": empr_np.approximation_time,
                "execution_time": empr_np.excution_time,
              }

# %%
import matplotlib.pyplot as plt

results = {'pytorch' : torch_times, 'tensorflow' : tf_times, 'numpy' : np_times}

colors = {
    'numpy': '#d4a15a',       # Light gold
    'tensorflow': '#d96557',  # Light red
    'pytorch': '#446785'      # Dark blue
}

# Define the categories to plot and their custom labels
categories = ['approximation_time', 'execution_time']
labels = {'approximation_time': 'Apprx. Time', 'execution_time': 'Execution Time'}

# Collect data for each category and backend
data = {cat: [results[backend][cat] for backend in backends] for cat in categories}

# Set up bar width and positions
bar_width = 0.2
x_positions = np.arange(len(categories))  # Positions for each category
plt.figure(figsize=(10, 6))

# Plot each backend’s data with the corresponding color
for i, backend in enumerate(backends):
    positions = [x + i * bar_width for x in x_positions]  # Offset each backend’s bars within the category group
    values = [data[cat][i] for cat in categories]

    # Plot bars with the assigned color for each backend
    bars = plt.bar(positions, values, color=colors[backend], width=bar_width, label=backend)

    # Add values on top of each bar with a slight offset
    for j, bar in enumerate(bars):
        # Instead of bar.get_height(), use the actual value directly
        yval = values[j]  # Directly use the value from `values` list

        # Display values only if the bar height (value) is non-zero
        if yval > 0:
            # Place text slightly above the bar
            plt.text(
                bar.get_x() + bar.get_width() / 2, 
                yval + 0.03 * yval,  # Offset of 3% of the bar height
                f'{yval:.5f}', 
                ha='center', va='bottom', fontsize=9, color='black'
            )

# Customize x-axis with the category labels at appropriate positions
plt.xticks([r + bar_width for r in range(len(categories))], [labels[cat] for cat in categories])

# Label and formatting
plt.xlabel('Benchmark Category')
plt.ylabel('Time (seconds)')
plt.title('EMPR Calculation Benchmark by Backend')
plt.legend(title="Backends")  # No border for legend

# Display the bar chart
plt.tight_layout()  # Ensure the layout is tight and there is no clipping of text
plt.grid()
plt.savefig('fig1.png')

# %%
print('PyTorch Times\t:', torch_times, '\nTensorFlow Times:', tf_times, '\nNumpy Times\t:', np_times)


