import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch

class EMPRBenchmark:
    def __init__(self, tensor_shape=(500, 500, 500), order=3, supports_type='das'):
        """
        Initializes the benchmark configuration with the specified tensor shape, approximation order, and support type.
        """
        self.tensor_shape = tensor_shape
        self.order = order
        self.supports_type = supports_type
        self.results = {}

    def benchmark_empr_calculator(self, backend, G):
        """
        Benchmarks the EMPRCalculator for a specific backend.
        Records the initialization, approximation, and MSE computation times.
        """
        # Initialization time
        start_time = time.time()
        empr_calculator = NDEMPRCalculator(G, supports=self.supports_type, backend=backend)
        
        # Approximation time
        approximation_start_time = time.time()
        approximation = empr_calculator.calculate_approximation(order=self.order)
        approximation_time = time.time() - approximation_start_time
        
        # Total time taken
        total_time = time.time() - start_time
        
        return {
            "approximation_time": approximation_time,
            "total_time": total_time,
        }

    def run_benchmarks(self):
        """
        Runs the benchmarks for different backends and stores the results.
        """
        # NumPy backend benchmark
        G_numpy = np.random.rand(*self.tensor_shape)
        print("Testing with NumPy backend...")
        self.results['numpy'] = self.benchmark_empr_calculator('numpy', G_numpy)

        # TensorFlow backend benchmark
        G_tensorflow = tf.random.uniform(self.tensor_shape, dtype=tf.float64)
        print("Testing with TensorFlow backend...")
        self.results['tensorflow'] = self.benchmark_empr_calculator('tensorflow', G_tensorflow)

        # PyTorch backend benchmark
        torch.manual_seed(0)
        G_pytorch = torch.rand(self.tensor_shape, dtype=torch.float64)
        print("Testing with PyTorch backend...")
        self.results['pytorch'] = self.benchmark_empr_calculator('pytorch', G_pytorch)

    def display_results(self):
        """
        Prints out the benchmark results in a formatted way.
        """
        print("\nBenchmark Results:")
        for backend, result in self.results.items():
            print(f"\nBackend: {backend}")
            print(f"Total Time: {result['total_time']:.4f} seconds")
            print(f"Approximation Time: {result['approximation_time']:.4f} seconds")

    def plot_results(self, colors=None):
        """
        Plots a grouped bar chart for benchmark results where each timing category is grouped,
        and each backend has a consistent color across groups.

        Parameters:
        - colors: Dictionary mapping each backend to a specific color (e.g., {'numpy': 'blue', 'tensorflow': 'green', 'pytorch': 'red'}).
        """
        # Define default colors if not provided
        if colors is None:
            colors = {
                'numpy': '#d4a15a',       # Light gold
                'tensorflow': '#d96557',  # Light red
                'pytorch': '#446785'      # Dark blue
            }

        # Define the categories to plot and their custom labels
        categories = ['approximation_time', 'total_time']
        labels = {'approximation_time': 'Apprx. Time', 'total_time': 'Total Time'}

        # Backend names
        backends = list(self.results.keys())

        # Collect data for each category and backend
        data = {cat: [self.results[backend][cat] for backend in backends] for cat in categories}

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
                        f'{yval:.2f}', 
                        ha='center', va='bottom', fontsize=9, color='black'
                    )

        # Customize x-axis with the category labels at appropriate positions
        plt.xticks([r + bar_width for r in range(len(categories))], [labels[cat] for cat in categories])

        # Label and formatting
        plt.xlabel('Benchmark Category')
        plt.ylabel('Time (seconds)')
        plt.title('EMPR Calculation Benchmark by Backend')
        plt.legend(title="Backends", loc="upper right", frameon=False)  # No border for legend

        # Display the bar chart
        plt.tight_layout()  # Ensure the layout is tight and there is no clipping of text
        plt.show()





if __name__ == "__main__":
    # Initialize the benchmark
    benchmark = EMPRBenchmark(tensor_shape=(3, 4, 5), order=3, supports_type='das')
    # Run the benchmark
    benchmark.run_benchmarks()
    # Display the results
    benchmark.display_results()
    # Plot the results as a bar chart
    benchmark.plot_results()
