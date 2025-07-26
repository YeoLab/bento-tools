import json
import matplotlib.pyplot as plt
import numpy as np

# Load the benchmark data
with open('benchmarks/distance_benchmarks_updated.json', 'r') as f:
    data = json.load(f)

# Extract data for plotting
benchmarks = data['benchmarks']
cells = []
times_mean = []
times_median = []
times_min = []
times_max = []
labels = []

for benchmark in benchmarks:
    # Extract number of cells (shapes) from extra_info
    num_cells = benchmark['extra_info']['shapes']
    cells.append(num_cells)
    
    # Extract timing statistics
    stats = benchmark['stats']
    times_mean.append(stats['mean'])
    times_median.append(stats['median'])
    times_min.append(stats['min'])
    times_max.append(stats['max'])
    
    # Extract parameter label
    labels.append(benchmark['param'])

# Create the plot
plt.figure(figsize=(12, 8))

# Plot mean times with error bars (showing min/max range)
plt.errorbar(cells, times_mean, 
             yerr=[np.array(times_mean) - np.array(times_min), 
                   np.array(times_max) - np.array(times_mean)],
             fmt='o-', capsize=5, capthick=2, label='Mean execution time', 
             linewidth=2, markersize=8)

# Plot median times
plt.plot(cells, times_median, 's--', label='Median execution time', 
         linewidth=2, markersize=8, alpha=0.7)

# Set log scale for both axes
plt.xscale('log')
plt.yscale('log')

# Customize the plot
plt.xlabel('Number of Cells', fontsize=14)
plt.ylabel('Execution Time (seconds)', fontsize=14)
plt.title('Distance Statistics Benchmark Results\nExecution Time vs Number of Cells', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Add annotations for each data point
for i, (x, y, label) in enumerate(zip(cells, times_mean, labels)):
    plt.annotate(f'{label}\n({x:,} cells)', 
                xy=(x, y), xytext=(10, 10), 
                textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Format axes
plt.gca().tick_params(axis='both', which='major', labelsize=12)

# Add some statistics to the plot
commit_info = data['commit_info']
machine_info = data['machine_info']

info_text = f"Machine: {machine_info['cpu']['brand_raw']} ({machine_info['cpu']['count']} cores)\n"
info_text += f"Commit: {commit_info['id'][:8]}\n"
info_text += f"Branch: {commit_info['branch']}"

plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('benchmarks/distance_benchmarks_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Print some summary statistics
print("Distance Statistics Benchmark Summary:")
print("=" * 50)
for i, benchmark in enumerate(benchmarks):
    param = benchmark['param']
    points_count = benchmark['extra_info']['points']
    shapes_count = benchmark['extra_info']['shapes']
    mean_time = benchmark['stats']['mean']
    ops_per_sec = benchmark['stats']['ops']
    
    print(f"{param.upper():>7}: {points_count:>11,} points, {shapes_count:>6,} cells")
    print(f"         Mean time: {mean_time:>8.2f}s, Ops/sec: {ops_per_sec:.6f}")
    print() 