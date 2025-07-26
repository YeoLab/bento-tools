#!/usr/bin/env python3
"""
Plot benchmark results showing dataset size vs execution time with theoretical complexity lines.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_benchmark_data():
    """Load benchmark data from JSON files."""
    
    # Load all benchmark files
    files = {
        'distance': 'distance_benchmarks.json',
        'polarity': 'polarity_benchmarks.json', 
        'moments': 'moments_benchmarks.json',
        'ripley': 'ripley_benchmarks.json'
    }
    
    data = {}
    for feature, filename in files.items():
        try:
            with open(filename, 'r') as f:
                data[feature] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {filename} not found, skipping {feature}")
            continue
    
    return data

def extract_timing_data(benchmark_data):
    """Extract timing data organized by feature and dataset size."""
    
    timing_data = {}
    
    for feature, data in benchmark_data.items():
        timing_data[feature] = {}
        
        for benchmark in data['benchmarks']:
            dataset_size = benchmark['extra_info']['shapes']
            mean_time = benchmark['stats']['mean']
            
            timing_data[feature][dataset_size] = {
                'time': mean_time,
                'shapes': dataset_size,
                'points': benchmark['extra_info']['points']
            }
    
    return timing_data

def generate_complexity_lines(shapes_range, base_time):
    """Generate theoretical complexity lines for comparison."""
    
    shapes = np.array(shapes_range)
    n_min = min(shapes)
    
    # Normalize all lines to start at roughly the same point as the smallest dataset
    complexity_lines = {
        'O(1)': np.full_like(shapes, base_time, dtype=float),
        'O(log n)': base_time * np.log(shapes) / np.log(n_min),
        'O(n)': base_time * shapes / n_min,
        'O(n log n)': base_time * (shapes * np.log(shapes)) / (n_min * np.log(n_min)),
        'O(n²)': base_time * (shapes ** 2) / (n_min ** 2),
        'O(n³)': base_time * (shapes ** 3) / (n_min ** 3)
    }
    
    return complexity_lines

def plot_benchmarks():
    """Create the main benchmark plot."""
    
    # Load and process data
    benchmark_data = load_benchmark_data()
    timing_data = extract_timing_data(benchmark_data)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Benchmark Results: Dataset Size vs Execution Time\nwith Theoretical Complexity Lines', 
                 fontsize=16, fontweight='bold')
    
    # Color palette for features
    feature_colors = {
        'distance': '#1f77b4',
        'polarity': '#ff7f0e', 
        'moments': '#2ca02c',
        'ripley': '#d62728'
    }
    
    # Plot each feature
    axes = [ax1, ax2, ax3, ax4]
    features = ['distance', 'polarity', 'moments', 'ripley']
    feature_names = {
        'distance': 'Distance Stats',
        'polarity': 'Polarity', 
        'moments': 'Moments',
        'ripley': 'Ripley\'s K'
    }
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        if feature not in timing_data:
            ax.text(0.5, 0.5, f'{feature_names[feature]}\nNo data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(feature_names[feature])
            continue
        
        # Extract data points
        data_points = timing_data[feature]
        shapes = sorted(data_points.keys())
        times = [data_points[s]['time'] for s in shapes]
        
        # Plot actual data points
        ax.scatter(shapes, times, color=feature_colors[feature], s=100, 
                  label=f'{feature_names[feature]} (actual)', zorder=5, alpha=0.8)
        ax.plot(shapes, times, color=feature_colors[feature], linewidth=2, zorder=4, alpha=0.8)
        
        # Generate theoretical complexity lines
        if shapes:  # Only if we have data
            shapes_range = np.linspace(min(shapes), max(shapes), 100)
            base_time = min(times)  # Use smallest measurement as baseline
            
            complexity_lines = generate_complexity_lines(shapes_range, base_time)
            
            # Plot complexity lines with different styles
            line_styles = {
                'O(1)': '--',
                'O(log n)': '-.',
                'O(n)': ':',
                'O(n log n)': '--',
                'O(n²)': '-.',
                'O(n³)': ':'
            }
            
            line_colors = ['gray', 'lightcoral', 'lightblue', 'lightgreen', 'orange', 'purple']
            
            for j, (complexity, line_data) in enumerate(complexity_lines.items()):
                # Only plot complexity lines that are reasonable for the data range
                max_theoretical = max(line_data)
                max_actual = max(times)
                
                if max_theoretical <= max_actual * 5:  # Only show if within 5x of actual data
                    ax.plot(shapes_range, line_data, 
                           linestyle=line_styles[complexity], 
                           color=line_colors[j % len(line_colors)], 
                           alpha=0.6, linewidth=1.5,
                           label=complexity)
        
        # Formatting
        ax.set_xlabel('Number of Shapes')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(feature_names[feature])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        
        # Add data point annotations
        for shape, time in zip(shapes, times):
            ax.annotate(f'{time:.3f}s', 
                       (shape, time), 
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save plot
    plt.savefig('benchmark_scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('benchmark_scaling_analysis.pdf', bbox_inches='tight')
    
    return fig

def create_summary_plot():
    """Create a summary plot with all features on one axis."""
    
    # Load data
    benchmark_data = load_benchmark_data()
    timing_data = extract_timing_data(benchmark_data)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    feature_colors = {
        'distance': '#1f77b4',
        'polarity': '#ff7f0e', 
        'moments': '#2ca02c',
        'ripley': '#d62728'
    }
    
    feature_names = {
        'distance': 'Distance Stats',
        'polarity': 'Polarity', 
        'moments': 'Moments',
        'ripley': 'Ripley\'s K'
    }
    
    # Plot each feature
    for feature, data_points in timing_data.items():
        if not data_points:
            continue
            
        shapes = sorted(data_points.keys())
        times = [data_points[s]['time'] for s in shapes]
        
        ax.scatter(shapes, times, color=feature_colors[feature], s=100, 
                  label=feature_names[feature], alpha=0.8)
        ax.plot(shapes, times, color=feature_colors[feature], linewidth=2, alpha=0.8)
        
        # Add annotations
        for shape, time in zip(shapes, times):
            ax.annotate(f'{time:.2f}s', 
                       (shape, time), 
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=9, alpha=0.7)
    
    # Add theoretical O(n) and O(n²) lines for reference
    shapes_range = np.linspace(100, 20000, 100)
    base_time = 0.1  # 100ms baseline
    
    # O(n) line
    linear_line = base_time * shapes_range / 100
    ax.plot(shapes_range, linear_line, '--', color='gray', alpha=0.5, label='O(n) reference')
    
    # O(n²) line  
    quadratic_line = base_time * (shapes_range ** 2) / (100 ** 2)
    ax.plot(shapes_range, quadratic_line, ':', color='darkgray', alpha=0.5, label='O(n²) reference')
    
    ax.set_xlabel('Number of Shapes', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_title('Benchmark Results: All Features Comparison\nDataset Size vs Execution Time', 
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('benchmark_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig('benchmark_summary.pdf', bbox_inches='tight')
    
    return fig

def print_scaling_analysis():
    """Print numerical analysis of scaling behavior."""
    
    benchmark_data = load_benchmark_data()
    timing_data = extract_timing_data(benchmark_data)
    
    print("=== SCALING ANALYSIS ===\n")
    
    for feature, data_points in timing_data.items():
        if len(data_points) < 2:
            continue
            
        shapes = sorted(data_points.keys())
        times = [data_points[s]['time'] for s in shapes]
        
        print(f"{feature.upper()} SCALING:")
        
        for i in range(len(shapes)):
            print(f"  {shapes[i]:,} shapes: {times[i]:.3f}s")
        
        # Calculate scaling factors
        if len(shapes) >= 2:
            for i in range(1, len(shapes)):
                shape_ratio = shapes[i] / shapes[i-1]
                time_ratio = times[i] / times[i-1]
                
                # Estimate complexity based on ratio
                if time_ratio <= 1.1:
                    complexity_est = "~O(1)"
                elif time_ratio <= shape_ratio * 1.1:
                    complexity_est = "~O(n)"
                elif time_ratio <= (shape_ratio * np.log(shape_ratio)) * 1.1:
                    complexity_est = "~O(n log n)"
                elif time_ratio <= (shape_ratio ** 2) * 1.1:
                    complexity_est = "~O(n²)"
                else:
                    complexity_est = ">O(n²)"
                
                print(f"    {shapes[i-1]:,} → {shapes[i]:,}: {shape_ratio:.1f}x shapes, {time_ratio:.2f}x time ({complexity_est})")
        
        print()

if __name__ == "__main__":
    print("Loading benchmark data and creating plots...")
    
    # Create individual feature plots
    plot_benchmarks()
    print("Created: benchmark_scaling_analysis.png/pdf")
    
    # Create summary plot
    create_summary_plot()
    print("Created: benchmark_summary.png/pdf")
    
    # Print analysis
    print_scaling_analysis()
    
    print("Plotting complete!") 