import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter
from typing import List, Tuple


def load_json(file_path: str) -> dict:
    """
    Load JSON data from file

    Args:
        file_path: Path to the JSON file

    Returns:
        JSON data as a dictionary
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def compute_timing_metrics(timings: List[int]) -> Tuple[float, int, int, float]:
    """
    Compute metrics on timings: median, mean, min, and max

    Args:
        timings: Raw timing measurements

    Returns:
        Tuple of timing metrics (median, mean, min, max)
    """
    if len(timings) == 0:
        print("Plotting error: empty timings")
        exit(1)

    # Convert timings from nanoseconds to seconds
    timings_ms = np.array(timings) / 1e9

    # Calculate  metrics
    median_timings = np.median(timings_ms)
    min_timings = np.min(timings_ms)
    max_timings = np.max(timings_ms)
    mean_timings = np.mean(timings_ms)

    return median_timings, mean_timings, min_timings, max_timings

def plot_setup(ylabel='[Gflop/s]'):
    # Set background color to light gray
    plt.gca().set_facecolor('0.95')

    # Add horizontal lines
    plt.grid(axis='y', color='white', linestyle='-')

    # Remove all borders except bottom
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.xlabel('Input size')
    plt.xscale('log', base=2)

    plt.ylabel(ylabel, rotation='horizontal', horizontalalignment='left')

    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
    plt.gca().ticklabel_format(style='plain', axis='y')  # This disables scientific notation

    plt.gca().yaxis.set_label_coords(0, 1.02)

def generate_speedup_plot(data: dict, input_folder: str):
    """
    Generate speedup plot and save it at timings/input_folder/
    
    Args:
        data: JSON data containing performance information
        input_folder: Name of input folder at timings
    """
    plot_setup(ylabel='Speedup')
    all_performances = []
    line_colors = ['#FFBF00', '#FF7F50', '#DE3163', '#51de94', '#40E0D0', '#6495ED']

    # Compute performance for each run
    for i in range(len(data)):
        d = data[i]
        runs = d['runs']
        avg_timings = [compute_timing_metrics(run['timings'])[0] for run in runs]
        sizes = [run['N'] for run in runs] # Assume square matrices only for now
        gflops = [run['math_flops']/1E9 for run in runs] # disregard different flop types
        performances = [gflops[i] / avg_timings[i] for i in range(len(runs))]
        all_performances.append(performances)

    # Compute speedup for each run compared to the first run
    for i in range(1, len(all_performances)):
        speedup = [all_performances[i][j] / all_performances[0][j] for j in range(len(all_performances[0]))]
        plt.plot(sizes, speedup, color='0.0', linewidth=0.5)

    # Fill area between speedup lines
    for i in range(len(data)-1, 0, -1):
        speedup = [all_performances[i][j] / all_performances[0][j] for j in range(len(all_performances[0]))]
        d = data[i]
        label = d['meta']['function name']
        # Add compiler info if contained in data
        if 'compiler' in d['meta']:
            label += f" ({d['meta']['compiler']})"
        if 'flags' in d['meta']:
            label += f" {d['meta']['flags']}"
        plt.fill_between(sizes, speedup, 1, label=label, alpha=1.0, color=line_colors[i%len(line_colors)])

    # Make y-axis logarithmic, but keep x-axis linear
    plt.yscale('log', base=2)
    # Set tick label format for y-axis to plain and only integers without fractional part
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter()) 
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

    # Make sure x-ticks are integers and correspond to data points
    plt.xticks(sizes, sizes)

    # Only show gpu if all data is from the same gpu
    title_suffix = ""
    if all(d['meta']['gpu model'] == data[0]['meta']['gpu model'] for d in data):
        title_suffix = f" on {data[0]['meta']['gpu model']}"

    plt.title("Speedup compared to " + data[0]['meta']['function name'] + title_suffix, loc='left', fontsize=12, fontweight='bold', x=0, y=1.05)
    plt.legend()
    plt.gca().set_ylim(bottom=1)

    output_dir = input_folder
    output_file = "speedups.png"
    print(output_dir)
    plt.savefig(os.path.join(output_dir, output_file))
    plt.close()

def generate_performance_comparison_plot(data: List[dict], input_folder: str):
    """
    Generate performance comparison plot and save it at timings/input_folder/

    Args:
        data: JSON data containing performance information
        input_folder: Name of input folder at timings
    """
    plot_setup()
    # -- Comparison Plot specific setup --
    #line_colors = ['#c28e0d', '#903315', '#6b1a1f', '#5e331e', '#341a09', '#52236a']
    line_colors = ['#FFBF00', '#FF7F50', '#DE3163', '#51de94', '#40E0D0', '#6495ED']
    plt.gca().set_prop_cycle(marker=['o', '^', 'v', 's', 'D', 'p'])
    all_sizes = list(set([size for d in data for size in [run['N'] for run in d['runs']]])) # Assuming square matrices
    plt.xticks(all_sizes, all_sizes) # Force x-ticks to match union of all data

    # Compute performance for each run
    for i in range(len(data)):
        d = data[i]
        runs = d['runs']
        avg_timings = [compute_timing_metrics(run['timings'])[0] for run in runs]
        sizes = [run['N'] for run in runs]
        gflops = [run['math_flops']/1E9 for run in runs] # disregard different flop types
        performances = [gflops[i] / avg_timings[i] for i in range(len(runs))]
        label = d['meta']['function name']
        # Add compiler info if contained in data
        if 'compiler' in d['meta']:
            label += f" ({d['meta']['compiler']})"
        if 'flags' in d['meta']:
            label += f" {d['meta']['flags']}"
        plt.plot(sizes, performances, color=line_colors[i%len(line_colors)], label=label)

    # Only show gpu if all data is from the same gpu
    title_suffix = ""
    if all(d['meta']['gpu model'] == data[0]['meta']['gpu model'] for d in data):
        title_suffix = f" on {data[0]['meta']['gpu model']}"

    plt.title("Performance Comparison" + title_suffix, loc='left', fontsize=12, fontweight='bold', x=0, y=1.05)
    plt.legend()
    plt.gca().set_ylim(bottom=0)

    output_dir = input_folder
    output_file = "comparison.png"
    print(output_dir)
    plt.savefig(os.path.join(output_dir, output_file))
    plt.close()


def generate_performance_plot(data: dict, input_folder: str, plot_filename: str):
    """
    Generate performance plot and save it at timings/input_folder/plot_filename.png

    Args:
        data: JSON data containing performance information
        input_folder: Name of input folder at timings
        plot_filename: Plot will be saved as plot_filename.png
    """
    plot_setup()

    # Compute performance for each run
    runs = data['runs']
    avg_timings = [compute_timing_metrics(run['timings'])[0] for run in runs]
    gflops = [run['math_flops']/1E9 for run in runs] # disregard different flop types
    performances = [gflops[i] / avg_timings[i] for i in range(len(runs))]
    sizes = [run['N'] for run in runs] # Only use square matrices for now

    plt.plot(sizes, performances, marker='o', color='0.0')
    plt.xticks(sizes, sizes) # Force x-ticks to match data
    plt.title(f"Performance of {data['meta']['function name']} on {data['meta']['gpu model']}", loc='left', fontsize=12, fontweight='bold', x=0, y=1.05)
    plt.gca().set_ylim(bottom=0)

    output_dir = input_folder
    output_file = f"{plot_filename}.png"
    print(output_dir)
    plt.savefig(os.path.join(output_dir, f"{output_file}"))
    plt.close()


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir('../')
    
    parser = argparse.ArgumentParser(description='Generate performance plot(s) from JSON data.')
    parser.add_argument('--input_folder', type=str, help='Name of input folder at timings', required=True)
    parser.add_argument('--input_file', type=str, 
        help='Set if you only want to generate plot of a specific JSON file, otherwise a plot for all JSON files in input_folder will be generated', default=None)
    parser.add_argument('--compare', action='store_true', help='Flag to combine performance plots into a single plot.')
    parser.add_argument('--speedup', action='store_true', help='Flag to generate speedup plot.')
    args = parser.parse_args()

    rcParams['font.sans-serif'] = ['Tahoma', 'Verdana', 'Gill Sans MT', 'Calibri', 'DejaVu Sans']
    rcParams['font.family'] = 'sans-serif'

    input_folder = args.input_folder
    if args.input_file is None:
        if args.compare:
            json_data = [load_json(os.path.join(input_folder, file)) for file in sorted(os.listdir(input_folder)) if file.endswith('.json')]
            generate_performance_comparison_plot(json_data, args.input_folder)
        elif args.speedup:
            json_data = [load_json(os.path.join(input_folder, file)) for file in sorted(os.listdir(input_folder)) if file.endswith('.json')]
            generate_speedup_plot(json_data, args.input_folder)
        else:
            for file in os.listdir(input_folder):
                if file.endswith('.json'):
                    json_data = load_json(os.path.join(input_folder, file))
                    generate_performance_plot(json_data, args.input_folder, file[:-5])
    else:
        input_file = os.path.join(input_folder, args.input_file)
        json_data = load_json(input_file)
        generate_performance_plot(json_data, args.input_folder, args.input_file[:-5])


if __name__ == "__main__":
    main()
