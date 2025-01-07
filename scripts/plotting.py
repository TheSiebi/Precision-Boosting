import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter
from typing import List, Tuple
#from scipy import stats
#import statsmodels.api as sm


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


def compute_metrics(timings: List[int]) -> Tuple[float, int, int, float, float, float]:
    """
    Compute metrics on timings: median, mean, min, and max

    Args:
        timings: Raw timing measurements

    Returns:
        Tuple of timing metrics (median, mean, min, max, CI lower, CI upper)
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

    n = len(timings_ms)

    # Perform Shapiro-Wilk Test (deprecated, analysis has revealed that both shapiro-wilk always rejects normality assumption and q-q plots are not straight)
    # if (n >= 3):
    #     p_value = stats.shapiro(timings_ms)[1]
    #     print(p_value)
    #     is_normal = p_value > 0.05
    # else:
    #     is_normal = False # too small sample size for doing shapiro-wilk test
    is_normal = False

    # Calculate 95% confidence interval
    if is_normal:
        #print("Data is normally distributed")
        # Parametric confidence interval    
        sem = np.std(timings_ms, ddof=1) / np.sqrt(n) 
        z_critical = 1.959963984540054 # = norm.ppf(0.975) = 95% critical value (two-tailed)
        ci_lower = mean_timings - z_critical * sem
        ci_upper = mean_timings + z_critical * sem
    else:
        #print("Data is not normally distributed, using non-parametric CI")
        # # Non-parametric bootstrap confidence interval
        # ci_bootstrap = stats.bootstrap((timings_ms,), np.mean, confidence_level=0.95)
        # ci_lower, ci_upper = ci_bootstrap.confidence_interval.low, ci_bootstrap.confidence_interval.high

        # Non-parametric CI using Le Boudec's method described in scientific benchmarking paper
        sorted_timings = np.sort(timings_ms)
        z_critical = 1.959963984540054 # = norm.ppf(0.975) = 95% critical value (two-tailed)
        # Subtract 1 for zero indexing
        lower_rank = int(np.floor((n - z_critical * np.sqrt(n)) / 2)) - 1
        upper_rank = int(np.ceil(1 + (n + z_critical * np.sqrt(n)) / 2)) - 1
        
        #print(f"Lower rank: {lower_rank}, upper rank: {upper_rank}")

        # Ensure valid bounds
        lower_rank = max(lower_rank, 0)
        upper_rank = min(upper_rank, n - 1)

        ci_lower = sorted_timings[lower_rank]
        ci_upper = sorted_timings[upper_rank]

    return median_timings, mean_timings, min_timings, max_timings, ci_lower, ci_upper

def plot_setup(ylabel='[Gflop/s]', scientific=False):
    # Set background color to light gray
    plt.gca().set_facecolor('0.95')

    # Add horizontal lines
    plt.grid(axis='y', color='white', linestyle='-')

    # Remove all borders except bottom
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    plt.xlabel('Matrix size m : matmul-(m,m,m)')
    plt.xscale('log', base=2)

    plt.ylabel(ylabel, rotation='horizontal', horizontalalignment='left')

    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))

    if not(scientific):
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
        median_timings = [compute_metrics(run['timings'])[0] for run in runs]
        sizes = [run['N'] for run in runs] # Assume square matrices only for now
        gflops = [run['math_flops']/1E9 for run in runs] # disregard different flop types
        performances = [gflops[i] / median_timings[i] for i in range(len(runs))]
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
    line_colors = ['#FFBF00', '#FF7F50', '#DE3163', '#51de94', '#40E0D0', '#6495ED', '#0022b8', '#000000']
    plt.gca().set_prop_cycle(marker=['o', '^', 'v', 's', 'D', 'p'])
    all_sizes = list(set([size for d in data for size in [run['N'] for run in d['runs']]])) # Assuming square matrices
    plt.xticks(all_sizes, all_sizes) # Force x-ticks to match union of all data

    # Compute performance for each run
    for i in range(len(data)):
        d = data[i]
        runs = d['runs']
        metrics = [compute_metrics(run['timings']) for run in runs]
        median_timings = [metric[0] for metric in metrics]
        ci_lower = [metric[4] for metric in metrics]
        ci_upper = [metric[5] for metric in metrics]
        gflops = [run['math_flops']/1E9 for run in runs] # disregard different flop types
        performances = [gflops[i] / median_timings[i] for i in range(len(runs))]
        ci_lower_perf = [gflops[i] / ci_upper[i] for i in range(len(runs))]
        ci_upper_perf = [gflops[i] / ci_lower[i] for i in range(len(runs))]

        sizes = [run['N'] for run in runs] # Only use square matrices for now
        label = d['meta']['function name']
        # Add compiler info if contained in data
        if 'compiler' in d['meta']:
            label += f" ({d['meta']['compiler']})"
        if 'flags' in d['meta']:
            label += f" {d['meta']['flags']}"
        plt.fill_between(sizes, ci_lower_perf, ci_upper_perf, color=line_colors[i%len(line_colors)], alpha=0.25, edgecolor=None)
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
    plt.savefig(os.path.join(output_dir, f"comparison.pdf"), dpi=300, pad_inches=0, bbox_inches='tight') # PDF version
    plt.close()

def generate_matmul_performance_comparison_plot(data: List[dict], input_folder: str):
    """
    Generate performance comparison plot and save it at timings/input_folder/

    Args:
        data: JSON data containing performance information
        input_folder: Name of input folder at timings
    """
    plot_setup()
    # -- Comparison Plot specific setup --
    line_colors = ['#FFBF00', '#FF7F50', '#DE3163', '#51de94', '#40E0D0', '#6495ED', '#0022b8', '#000000']
    plt.gca().set_prop_cycle(marker=['o', '^', 'v', 's', 'D', 'p'])
    all_sizes = list(set([size for d in data for size in [run['N'] for run in d['runs']]])) # Assuming square matrices
    plt.xticks(all_sizes, all_sizes) # Force x-ticks to match union of all data

    # Plot horizontal line for peak theoretical performance of machine
    plt.axhline(y=52.22E3, color='#000000', linestyle='--')
    plt.text(x=1800, y=52.22E3 - 2.2E3, s="FP32 peak", color='#000000', ha='center')

    plt.axhline(y=104.4E3 / 3, color='#FF7F50', linestyle='--')
    plt.text(x=1000, y=(104.4E3 / 3) - 2.2E3, s="FP16 w. FP32 acc. peak/3", color='#FF7F50', ha='center')

    plt.axhline(y=104.4E3 / 4, color='#0022b8', linestyle='--')
    plt.text(x=1000, y=(104.4E3 / 4) - 2.2E3, s="FP16 w. FP32 acc. peak/4", color='#0022b8', ha='center')


    # Compute performance for each run
    for i in range(len(data)):
        d = data[i]
        runs = d['runs']
        matmul_fractions = []
        if d['meta']['function name'] != "cuBLAS":
            matmul_fractions = [parse_profile_text_fractions(run['profile_output'])["split & matmul & merge"] for run in runs]
        else:
            matmul_fractions = [parse_profile_text_fractions(run['profile_output'])["matmul"] for run in runs]
        sizes = [run['N'] for run in runs] # Only use square matrices for now
        metrics = [compute_metrics(run['timings']) for run in runs]
        median_timings = [metric[0] for metric in metrics]
        ci_lower = [metric[4] for metric in metrics]
        ci_upper = [metric[5] for metric in metrics]
        gflops = [run['math_flops']/1E9 for run in runs] # disregard different flop types
        performances = [gflops[i] / median_timings[i] / matmul_fractions[i] for i in range(len(runs))]
        ci_lower_perf = [gflops[i] / ci_upper[i] / matmul_fractions[i] for i in range(len(runs))]
        ci_upper_perf = [gflops[i] / ci_lower[i] / matmul_fractions[i] for i in range(len(runs))]

        label = d['meta']['function name']
        # Add compiler info if contained in data
        if 'compiler' in d['meta']:
            label += f" ({d['meta']['compiler']})"
        if 'flags' in d['meta']:
            label += f" {d['meta']['flags']}"
        plt.plot(sizes, performances, color=line_colors[i%len(line_colors)], label=label)
        plt.fill_between(sizes, ci_lower_perf, ci_upper_perf, color=line_colors[i%len(line_colors)], alpha=0.25, edgecolor=None)

    # Only show gpu if all data is from the same gpu
    title_suffix = ""
    if all(d['meta']['gpu model'] == data[0]['meta']['gpu model'] for d in data):
        title_suffix = f" on {data[0]['meta']['gpu model']}"

    plt.title("MatMul Comparison" + title_suffix, loc='left', fontsize=12, fontweight='bold', x=0, y=1.05)
    plt.legend()
    plt.gca().set_ylim(bottom=0)

    output_dir = input_folder
    output_file = "matmul_comparison.png"
    print(output_dir)
    plt.savefig(os.path.join(output_dir, output_file))
    plt.savefig(os.path.join(output_dir, f"matmul_comparison.pdf"), dpi=300, pad_inches=0, bbox_inches='tight') # PDF version
    plt.close()

def generate_precision_plots(data: List[dict], input_folder: str):
    """
    Generates a precision plot for each input type and save it at timings/input_folder/

    Args:
        data: JSON data containing precision information
        input_folder: Name of input folder at timings
    """
    unique_input_types = set([m['input_type'] for d in data for run in d['runs'] if 'precMs' in run for m in run['precMs']])

    # Compute performance for each run
    for input_type in unique_input_types:
        plot_setup(ylabel="Relative residual", scientific=True)
        plt.yscale('log', base=10)
        # -- Comparison Plot specific setup --
        line_colors = ['#FFBF00', '#FF7F50', '#DE3163', '#51de94', '#40E0D0', '#6495ED', '#0022b8', '#000000']
        plt.gca().set_prop_cycle(marker=['o', '^', 'v', 's', 'D', 'p'])
        all_sizes = list(set([size for d in data for size in [run['N'] for run in d['runs']]])) # Assuming square matrices
        plt.xticks(all_sizes, all_sizes) # Force x-ticks to match union of all data

        for i in range(len(data)):
            d = data[i]
            runs = [run for run in d['runs'] if 'precMs' in run]
            
            # If no runs have 'precMs', skip
            if not runs:
                continue
            
            all_run_residuals = [m['residuals'] for run in runs for m in run['precMs'] if m['input_type'] == input_type]

            avg_residual = [np.mean(np.array(run_residual)) for run_residual in all_run_residuals]
            sizes = [run['N'] for run in runs]
            residuals = [avg_residual[i] for i in range(len(runs))]
            label = d['meta']['function name']
            # Add compiler info if contained in data
            if 'compiler' in d['meta']:
                label += f" ({d['meta']['compiler']})"
            if 'flags' in d['meta']:
                label += f" {d['meta']['flags']}"
            plt.plot(sizes, residuals, color=line_colors[i%len(line_colors)], label=label)

        # Only show gpu if all data is from the same gpu
        title_suffix = ""
        if all(d['meta']['gpu model'] == data[0]['meta']['gpu model'] for d in data):
            title_suffix = f" on {data[0]['meta']['gpu model']}"

        plt.title("Type " + str(input_type) + " Precision Comparison" + title_suffix, loc='left', fontsize=12, fontweight='bold', x=0, y=1.05)
        plt.legend()
        #plt.gca().set_ylim(bottom=0)

        output_dir = input_folder
        output_file = "prec_comparison_type_" + str(input_type) + ".png"
        print(output_dir)
        plt.savefig(os.path.join(output_dir, output_file))
        plt.savefig(os.path.join(output_dir, f"prec_comparison_type_{str(input_type)}.pdf"), dpi=300, pad_inches=0, bbox_inches='tight') # PDF version
        plt.close()


def generate_precision_comparison_plot(data: List[dict], input_folder: str):
    """
    Generates a row of precision comparison plots and saves it at timings/input_folder/

    Args:
        data: JSON data containing precision information
        input_folder: Name of input folder at timings
    """
    unique_input_types = set([m['input_type'] for d in data for run in d['runs'] if 'precMs' in run for m in run['precMs']])

    # Remove Type 0 from comparison plot
    unique_input_types.remove(0)

    num_plots = len(unique_input_types)
    width_in_inches = 6.77*2  # Two-column width of A4 in inches
    height_per_plot = 2*2     # Approximate height per subplot in inches
    fig, axes = plt.subplots(1, num_plots, figsize=(width_in_inches, height_per_plot), sharey=True)
    line_colors = ['#FFBF00', '#FF7F50', '#DE3163', '#51de94', '#40E0D0', '#6495ED', '#0022b8', '#000000']

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable even for a single plot

    handles = []
    labels = []
    # Compute performance for each run
    for idx, input_type in enumerate(unique_input_types):
        ax = axes[idx]
        # Customize subplot
        # Set background color to light gray
        ax.set_facecolor('0.95')

        # Add horizontal lines
        ax.grid(axis='y', color='white', linestyle='-')
        # Remove all borders except bottom
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Add horizontal lines
        ax.grid(axis='y', color='white', linestyle='-')

        ax.set_xscale('log', base=2)
        ax.set_title("Type " + str(input_type))
        ax.set_xlabel(".", color=(0,0,0,0)) # invisible x-axis label to keep space
        if idx == 0:
            ax.set_ylabel("Mean Relative Error (Capped at 1)")
        ax.set_yscale('log', base=10)
        ax.minorticks_off()

        if idx != 0:
            # Disable y ticks of axis
            ax.yaxis.set_tick_params(size=0)

        # -- Comparison Plot specific setup --
        line_colors = ['#FFBF00', '#FF7F50', '#DE3163', '#51de94', '#40E0D0', '#6495ED', '#0022b8', '#000000']
        ax.set_prop_cycle(marker=['o', '^', 'v', 's', 'D', 'p'])
        #all_sizes = list(set([size for d in data for run in d['runs'] if 'precMs' in run for size in [run['N']]])) # Assuming square matrices

        for i in range(len(data)):
            d = data[i]
            runs = [run for run in d['runs'] if 'precMs' in run]
            
            # If no runs have 'precMs', skip
            if not runs:
                continue
            
            all_run_residuals = [m['residuals'] for run in runs for m in run['precMs'] if m['input_type'] == input_type]

            avg_residual = [np.mean(np.array(run_residual)) for run_residual in all_run_residuals]
            sizes = [run['N'] for run in runs]
            residuals = [avg_residual[i] for i in range(len(runs))]
            label = d['meta']['function name']
            # Add compiler info if contained in data
            if 'compiler' in d['meta']:
                label += f" ({d['meta']['compiler']})"
            if 'flags' in d['meta']:
                label += f" {d['meta']['flags']}"
            p, = ax.plot(sizes, residuals, color=line_colors[i%len(line_colors)], label=label)
            if idx == 0:
                handles.append(p)
                labels.append(label)

        #ax.legend()

    # Add a common legend below the x-axis label
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncols=len(labels), fontsize=10)
    fig.text(0.5, 0.15, 'Matrix size m : matmul-(16,m,16)', ha='center')
    # Add a common legend
    #handles, labels = axes[0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.5), ncol=len(profile_data[0].keys()), fontsize=10)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    output_dir = input_folder
    output_file = "precision_comparison.png"
    print(output_dir)
    plt.savefig(os.path.join(output_dir, output_file))
    plt.savefig(os.path.join(output_dir, "precision_comparison.pdf"), dpi=300, pad_inches=0, bbox_inches='tight') # PDF version
    plt.close()


# def plot_histogram_qq_grid(runs, input_folder, file_prefix, num_rows=3):
#     num_datasets = len(runs)
#     num_cols = 2  # Each dataset will have 2 columns: histogram and Q-Q plot

#     # Calculate the number of rows based on the specified grid
#     rows = max(num_rows, num_datasets)
    
#     # Set up the figure and axes for paired plots
#     fig, axes = plt.subplots(rows, num_cols, figsize=(10, rows * 4))
    
#     # If there's only one dataset, axes might not be a nested array
#     if rows == 1:
#         axes = np.array([axes])

#     for i, run in enumerate(runs):
#         if i >= rows:  # Limit plots to the number of rows specified
#             break
        
#         # Access the data and filter out NaN values
#         timings = np.array(run['timings'])
#         timings = timings[~np.isnan(timings)]  # Remove NaNs

#         m, k, n = run['M'], run['K'], run['N']
        
#         # Plot histogram
#         axes[i, 0].hist(timings, bins=20, color='skyblue', edgecolor='black')
#         axes[i, 0].set_title(f"Histogram for M={m}, K={k}, N={n}")
#         axes[i, 0].set_xlabel('Values')
#         axes[i, 0].set_ylabel('Frequency')
        
#         # Plot Q-Q plot
#         sm.qqplot(timings, line='s', ax=axes[i, 1])
#         axes[i, 1].set_title(f"Q-Q Plot for M={m}, K={k}, N={n}")
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(input_folder, f"{file_prefix}_qq_plots.png"))


def parse_profile_text_fractions(profile_text: str) -> dict:
    profile_data = {}

    # Split text by , and iterate over pairs, also add previous value if add_last_value is True
    tmp = profile_text.split(',')
    sum = 0
    for i in range(0, len(tmp)-1, 2):
        key = tmp[i].strip()
        value = float(tmp[i+1].strip())
        profile_data[key] = value
        sum += value

    # Normalize profile data to values between 0 and 1
    for key in profile_data.keys():
        profile_data[key] /= sum

    return profile_data

def parse_profile_text(profile_text: str, add_last_value=True) -> dict:
    """
    Parse profile text into a dictionary

    Args:
        profile_text: Text containing profile data

    Returns:
        Dictionary containing profile data
    """
    profile_data = {}

    # Split text by , and iterate over pairs, also add previous value if add_last_value is True
    tmp = profile_text.split(',')
    last_value = 0
    for i in range(0, len(tmp)-1, 2):
        key = tmp[i].strip()
        value = float(tmp[i+1].strip())
        if add_last_value:
            value += last_value
        last_value = value
        profile_data[key] = value

    return profile_data

def generate_profile_plot(data: dict, input_folder: str, plot_filename: str):
    """
    Generate profile plot showcasing profile segments and save it at timings/input_folder/plot_filename_profile.png

    Args:
        data: JSON data containing performance information
        input_folder: Name of input folder at timings
        plot_filename: Plot will be saved as plot_filename_profile.png
    """
    plot_setup(ylabel='Fraction of total runtime')
    line_colors = ['#FFBF00', '#FF7F50', '#DE3163', '#51de94', '#40E0D0', '#6495ED', '#0022b8', '#000000']

    # Get profile data
    runs = data['runs']
    sample_counts = [len(run['timings']) for run in runs]
    sizes = [run['N'] for run in runs] # Only use square matrices for now
    profile_data = [parse_profile_text(run['profile_output']) for run in runs]

    # Normalize profile data to values between 0 and 1
    for i in range(len(profile_data)):
        max_value = max(profile_data[i].values())
        for key in profile_data[i].keys():
            profile_data[i][key] /= max_value

    # Plot segment runtimes
    for key in profile_data[0].keys():
        segment_runtimes = [profile_data[i][key] for i in range(len(profile_data))]
        plt.plot(sizes, segment_runtimes, color='0.0', linewidth=0.5)

    # Fill area between profile segment runtimes
    i = 0
    for key in reversed(profile_data[0].keys()):
        segment_runtimes = [profile_data[i][key] for i in range(len(profile_data))]
        plt.fill_between(sizes, segment_runtimes, 0, label=key, alpha=1.0, color=line_colors[i%len(line_colors)])
        i+=1

    # Make sure x-ticks are integers and correspond to data points
    plt.xticks(sizes, sizes)

    title_suffix = f" on {data['meta']['gpu model']}"
    plt.title("Profile of " + data['meta']['function name'] + title_suffix, loc='left', fontsize=12, fontweight='bold', x=0, y=1.05)
    #plt.legend()
    plt.gca().set_ylim(bottom=0)
    output_dir = input_folder
    output_file = f"{plot_filename}_profile.png"
    print(output_dir)
    plt.savefig(os.path.join(output_dir, output_file))
    plt.savefig(os.path.join(output_dir, f"{plot_filename}_profile.pdf"), dpi=300, pad_inches=0, bbox_inches='tight') # PDF version
    plt.close()


def generate_profile_comparison_plot(data: List[dict], input_folder: str):
    """
    Generate a row of profile plot showcasing profile segments and save it at timings/input_folder/plot_filename_profile.png

    Args:
        data: JSON data containing performance information
        input_folder: Name of input folder at timings
        plot_filename: Plot will be saved as plot_filename_profile.png
    """
    num_plots = len(data)
    width_in_inches = 6.77*2  # Two-column width of A4 in inches
    height_per_plot = 2*2     # Approximate height per subplot in inches
    fig, axes = plt.subplots(1, num_plots, figsize=(width_in_inches, height_per_plot), sharey=True)
    line_colors = ['#FFBF00', '#FF7F50', '#DE3163', '#51de94', '#40E0D0', '#6495ED', '#0022b8', '#000000']

    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable even for a single plot

    for idx, (ax, dataset) in enumerate(zip(axes, data)):
        runs = dataset['runs']
        sizes = [run['N'] for run in runs]  # Only use square matrices for now
        profile_data = [parse_profile_text(run['profile_output']) for run in runs]

        # Normalize profile data to values between 0 and 1
        for i in range(len(profile_data)):
            max_value = max(profile_data[i].values())
            for key in profile_data[i].keys():
                profile_data[i][key] /= max_value

        # Plot segment runtimes
        for key in profile_data[0].keys():
            segment_runtimes = [profile_data[i][key] for i in range(len(profile_data))]
            ax.plot(sizes, segment_runtimes, color='0.0', linewidth=0.5)

        # Fill area between profile segment runtimes
        i = 0
        for key in reversed(profile_data[0].keys()):
            segment_runtimes = [profile_data[i][key] for i in range(len(profile_data))]
            ax.fill_between(sizes, segment_runtimes, 0, label=key, alpha=1.0, color=line_colors[i % len(line_colors)])
            i += 1

        # Customize subplot
        # Remove all borders except bottom
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Add horizontal lines
        ax.grid(axis='y', color='white', linestyle='-')

        ax.set_xscale('log', base=2)
        ax.set_title(dataset['meta']['function name'], fontsize=12, fontweight='bold')
        ax.set_xticks(sizes)
        ax.set_xlim(min(sizes), max(sizes))  # Ensure xticks span full width of plot
        ax.set_xlabel(".", color=(0,0,0,0)) # invisible x-axis label to keep space
        if idx == 0:
            ax.set_ylabel("Fraction of total runtime")
        ax.set_ylim(0, 1)
        #ax.legend()

    all_labels = [("allocate", line_colors[4]), ("memcpy host2device", line_colors[3]), ("matmul (+ split & merge)", line_colors[2]), ("memcpy device2host", line_colors[1]), ("free", line_colors[0])]

     # Add a common legend below the x-axis label
    handles = [plt.Line2D([0], [0], color=color, lw=6, label=label) for label, color in all_labels]

    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=len(all_labels), fontsize=10)
    fig.text(0.5, 0.15, 'Matrix size m : matmul-(m,m,m)', ha='center')

    # Add a common legend
    #handles, labels = axes[0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.5), ncol=len(profile_data[0].keys()), fontsize=10)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Save the figure
    output_dir = input_folder
    output_file = f"profile_comparison.png"
    plt.savefig(os.path.join(output_dir, output_file))
    plt.savefig(os.path.join(output_dir, f"profile_comparison.pdf"), dpi=300, bbox_inches='tight')
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
    metrics = [compute_metrics(run['timings']) for run in runs]
    median_timings = [metric[0] for metric in metrics]
    ci_lower = [metric[4] for metric in metrics]
    ci_upper = [metric[5] for metric in metrics]

    gflops = [run['math_flops']/1E9 for run in runs] # disregard different flop types
    performances = [gflops[i] / median_timings[i] for i in range(len(runs))]

    # Performance bounds for confidence intervals
    ci_lower_perf = [gflops[i] / ci_upper[i] for i in range(len(runs))]
    ci_upper_perf = [gflops[i] / ci_lower[i] for i in range(len(runs))]

    sizes = [run['N'] for run in runs] # Only use square matrices for now

    plt.fill_between(sizes, ci_lower_perf, ci_upper_perf, color='lightgrey', alpha=0.5, label='95% Confidence Interval')

    plt.plot(sizes, performances, marker='o', color='0.0')
    plt.xticks(sizes, sizes) # Force x-ticks to match data
    plt.title(f"Performance of {data['meta']['function name']} on {data['meta']['gpu model']}", loc='left', fontsize=12, fontweight='bold', x=0, y=1.05)
    plt.gca().set_ylim(bottom=0)

    #plot_histogram_qq_grid(runs, input_folder, plot_filename)
    output_dir = input_folder
    output_file = f"{plot_filename}.png"
    print(output_dir)
    plt.savefig(os.path.join(output_dir, f"{output_file}"))
    plt.savefig(os.path.join(output_dir, f"{plot_filename}.pdf"), dpi=300, pad_inches=0, bbox_inches='tight') # PDF version
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
    parser.add_argument('--precision', action='store_true', help='Flag to generate precision plot.')
    parser.add_argument('--profile', action='store_true', help='Flag to generate profile plot.')
    parser.add_argument('--compare-profile', action='store_true', help='Flag to generate profile comparison plot.')
    parser.add_argument('--compare-matmul', action='store_true', help='Flag to only consider matmul segments for comparison plot.')
    parser.add_argument('--compare-precision', action='store_true', help='Flag to generate precision comparison plot.')
    args = parser.parse_args()

    rcParams['font.sans-serif'] = ['Tahoma', 'Verdana', 'Gill Sans MT', 'Calibri', 'DejaVu Sans']
    rcParams['font.family'] = 'sans-serif'

    input_folder = args.input_folder

    # If input folder is empty, do nothing
    if not os.listdir(input_folder):
        #print(f"The input folder '{input_folder}' is empty. Exiting.")
        return

    if args.input_file is None:
        if args.compare:
            json_data = [load_json(os.path.join(input_folder, file)) for file in sorted(os.listdir(input_folder)) if file.endswith('.json')]
            generate_performance_comparison_plot(json_data, args.input_folder)
        elif args.speedup:
            json_data = [load_json(os.path.join(input_folder, file)) for file in sorted(os.listdir(input_folder)) if file.endswith('.json')]
            generate_speedup_plot(json_data, args.input_folder)
        elif args.precision:
            json_data = [load_json(os.path.join(input_folder, file)) for file in os.listdir(input_folder) if file.endswith('.json')]
            generate_precision_plots(json_data, args.input_folder)
        elif args.compare_matmul:
            json_data = [load_json(os.path.join(input_folder, file)) for file in sorted(os.listdir(input_folder)) if file.endswith('.json')]
            generate_matmul_performance_comparison_plot(json_data, args.input_folder)
        elif args.compare_profile:
            json_data = [load_json(os.path.join(input_folder, file)) for file in sorted(os.listdir(input_folder)) if file.endswith('.json')]
            generate_profile_comparison_plot(json_data, args.input_folder)
        elif args.compare_precision:
            json_data = [load_json(os.path.join(input_folder, file)) for file in os.listdir(input_folder) if file.endswith('.json')]
            generate_precision_comparison_plot(json_data, args.input_folder)
        elif args.profile:
            for file in os.listdir(input_folder):
                if file.endswith('.json'):
                    json_data = load_json(os.path.join(input_folder, file))
                    generate_profile_plot(json_data, args.input_folder, file[:-5])
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
