"""
Plots different stuff
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from constraints import OptimizationProblemProperties
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import matplotlib.patches as mpatches
import os
from collections import deque

sns.set_style("darkgrid")
plt.rcParams.update({'font.size':16})

INIT_PROD = {
    "sample6": 
        {"oil": 97.7639356,
         "water": 71.593,
         "gaslift": 0.0,
         "water_max": 72.,
         "gaslift_max": 5.0,
         "opt_prod": 106.5},
    "gaslift_choke50%": 
        {"oil": 33.269,
         "water": 15.488,
         "gaslift": 0.0,
         "water_max": 20.0,
         "gaslift_max": 10.0,
         "opt_prod": 56.4},
    "gaslift_choke50%_highnoise": 
        {"oil": 33.269,
         "water": 15.488,
         "gaslift": 0.0,
         "water_max": 20.0,
         "gaslift_max": 10.0,
         "opt_prod": 56.4},
    "high_choke50%_highnoise": 
        {"oil": 93.5196,
         "water": 28.149,
         "gaslift": 0.0,
         "water_max": 40.0,
         "gaslift_max": 10.0,
         "opt_prod": 112.7},
    "high_choke50%_lownoise": 
        {"oil": 93.5196,
         "water": 28.149,
         "gaslift": 0.0,
         "water_max": 40.0,
         "gaslift_max": 10.0,
         "opt_prod": 112.7},
    "high_choke50%_nonoise": 
        {"oil": 93.5196,
         "water": 28.149,
         "gaslift": 0.0,
         "water_max": 40.0,
         "gaslift_max": 10.0,
         "opt_prod": 112.7},
    "high_choke50%_strictwater": 
        {"oil": 93.5196,
         "water": 28.149,
         "gaslift": 0.0,
         "water_max": 30.0,
         "gaslift_max": 10.0,
         "opt_prod": 112.7},
    "mixed_choke0": 
        {"oil": 0.0,
         "water": 0.0,
         "gaslift": 0.0,
         "water_max": 20.0,
         "gaslift_max": 10.0,
         "opt_prod": 68},
    "mixed_prod_badchoke": 
        {"oil": 29.048,
         "water": 19.76,
         "gaslift": 0.0,
         "water_max": 20.0,
         "gaslift_max": 10.0,
         "opt_prod": 68},
    "mixed_prod_choke50%": 
        {"oil": 62.636,
         "water": 19.594,
         "gaslift": 0.0,
         "water_max": 20.0,
         "gaslift_max": 10.0,
         "opt_prod": 68},
}

def keep_last_unique_pairs(df, correct_i):
    skips = []
    df_grouped = df.groupby("ID")
    for g in df_grouped.groups:
        if len(df_grouped.get_group(g)) == correct_i:
            skips.append(g)

    last_pairs = deque(maxlen=2)
    last_pairs.append(sorted([df.iloc[0]["CHK"], df.iloc[1]["CHK"]]))
    last_pairs.append(sorted([df.iloc[1]["CHK"], df.iloc[2]["CHK"]]))

    return_df = df.copy()

    skipped_last = False
    for idx, row in df.iloc[3:].iterrows():
        if row["ID"] in skips:
            continue
        if skipped_last:
            skipped_last = False
            continue
        current_pair = sorted([df.iloc[idx - 1]["CHK"], row["CHK"]])
        if current_pair == last_pairs[0] and all(
                                            df.iloc[i]["DCNT"] == "Perturbation" for i in range(idx-3, idx+1)):
            print(f"Removin pairs at index {idx - 3} and {idx - 2} with values {last_pairs[0]}")
            return_df = return_df.drop(index=[idx - 3, idx-2])
            skipped_last = True
        last_pairs.append(current_pair)
    
    return return_df.reset_index(drop=True)
        

def cleanup(path):
    #TODO NOT RIGHT
    for i in range(10):
        data_path = f"{path}/run{i}"
        for j in range(10, 101, 10):
            dataset_name = f'iteration_{j}'
            if not os.path.exists(os.path.join(data_path, dataset_name)):
                print(f"Reached end of runs for {dataset_name} in {data_path} at {j}.")
                break
            filename = os.path.join(data_path, dataset_name, f"{dataset_name}.csv")
            cleanup_df(pd.read_csv(filename), filename)

def cleanup_df(df, filename):
    drops = []
    p_counter = 0
    start_drop = 0
    for i in range(len(df)):
        if df["DCNT"].iloc[i] == "Perturbation":
            p_counter += 1
        elif df["DCNT"].iloc[i] == "Optimizing":
            if p_counter > 3:
                drops.append(list([start_drop, i]))
            p_counter = 0
        if p_counter == 3:
            start_drop = i

    for tup in drops:
        df = df.drop(index=range(tup[0], tup[1]))
    df = df.reset_index(drop=True)
    df.to_csv(filename, index=False)

import re

def extract_spsa_params(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Regular expressions to extract the required parameters
    pattern = r'a:\s*([\d.]+).*?c:\s*([\d.]+).*?alpha:\s*([\d.]+).*?gamma:\s*([\d.]+)'
    match = re.search(pattern, content, re.DOTALL)

    if match:
        a = float(match.group(1))
        c = float(match.group(2))
        alpha = float(match.group(3))
        gamma = float(match.group(4))
        return a, c, alpha, gamma
    else:
        raise ValueError("Could not extract SPSA parameters from file.")

def plot_gain_seqence(df: pd.DataFrame, save_path: str | None = None):
    df = df[(df["ID"] == 0) & (df["DCNT"] == "Optimizing")]

    # Combine the default and actual values
    choke_vals = pd.concat([pd.Series([0.5]), df["CHK"].reset_index(drop=True)], ignore_index=True)
    
    iterations = choke_vals.index
    gain_length = [abs(choke_vals[i-1] - choke_vals[i]) for i in range(1, len(choke_vals))]
    plt.figure(figsize=(10, 6))
    plt.bar(iterations[1:], gain_length, color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Gain Length')
    plt.title('Gradient Updates Over Iterations')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_single_well(df: pd.DataFrame, col_names = ["CHK", "WGL", "WOIL"]):
    df = df[col_names]
    df.plot()
    plt.show()

def plot_decision_iterates_side_by_side(base_paths: list[str], n_runs: int, titles: list[str] = None, save_path: str | None = None):
    """
    Plots decision iterates from three different experiments side by side.

    Parameters:
        base_paths (list[str]): List of 3 paths to the base directories of experiments.
        n_runs (int): Number of runs to include from each base path.
        titles (list[str]): Optional titles for the subplots.
        save_path (str | None): Optional path to save the combined figure.
    """
    assert len(base_paths) == 3, "You must provide exactly three base paths."

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    if titles is None:
        titles = [f"Experiment {i+1}" for i in range(3)]

    # First pass: Collect all unique well IDs across all datasets
    all_well_ids = set()
    for base_path in base_paths:
        for run_idx in range(n_runs):
            for i in range(100, 0, -10):
                run_path = f"{base_path}/run{run_idx}/iteration_{i}/iteration_{i}_config.csv"
                if os.path.exists(run_path):
                    df = pd.read_csv(run_path)
                    all_well_ids.update(df['wp.L'].unique())
                    break

    well_ids = sorted(all_well_ids)
    colormap = cm.get_cmap('Accent', len(well_ids))
    color_mapping = {well_id: colormap(i % 10) for i, well_id in enumerate(well_ids)}

    for ax, base_path, title in zip(axs, base_paths, titles):
        for run_idx in range(n_runs):
            df = None
            for i in range(100, 0, -10):
                run_path = f"{base_path}/run{run_idx}/iteration_{i}/iteration_{i}_config.csv"
                if os.path.exists(run_path):
                    df = pd.read_csv(run_path)
                    break

            if df is None:
                continue

            for _, row in df.iterrows():
                well_id = row['wp.L']
                u = row['bc.u']
                w_lg = row['bc.w_lg']
                color = color_mapping.get(well_id, 'gray')
                ax.scatter(u, w_lg, color=color, s=50)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Choke")

    axs[0].set_ylabel("Gas Lift")

    # Shared legend
    legend_patches = [
        mpatches.Patch(color=color_mapping[well_id], label=f"Well {i}")
        for i, well_id in enumerate(well_ids)
    ]
    axs[2].legend(handles=legend_patches, fontsize='small', loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_decicion_iterate(base_path: str, n_runs: int, save_path: str | None = None):
    """
    Plots the decision iterate for all wells across multiple runs.
    Each well (identified by 'wp.L') is plotted with a consistent color.
    Choke (bc.u) on x-axis vs Gas Lift (bc.w_lg) on y-axis.
    """
    fig, ax = plt.subplots()
    well_ids = set()
    well_data = {}

    # First pass: collect all well IDs across runs to assign colors consistently
    for run_idx in range(n_runs):
        for i in range(100, 0, -10):
            run_path = f"{base_path}/run{run_idx}/iteration_{i}/iteration_{i}_config.csv"
            if os.path.exists(run_path):
                df = pd.read_csv(run_path)
                well_ids.update(df['wp.L'].unique())
                break

    # Assign a unique color to each well using wp.L
    well_ids = sorted(well_ids)  # Optional: consistent order
    print(well_ids)
    colormap = cm.get_cmap('Accent', len(well_ids))
    color_mapping = {well_id: colormap(i % 10) for i, well_id in enumerate(well_ids)}

    # Second pass: plot values
    for run_idx in range(n_runs):
        print(f"Processing Run {run_idx + 1}/{n_runs}...")

        df = None
        for i in range(100, 0, -10):
            run_path = f"{base_path}/run{run_idx}/iteration_{i}/iteration_{i}_config.csv"
            if os.path.exists(run_path):
                df = pd.read_csv(run_path)
                break

        if df is None:
            print(f"No valid data found for Run {run_idx}. Skipping.")
            continue

        for _, row in df.iterrows():
            well_id = row['wp.L']
            u = row['bc.u']
            w_lg = row['bc.w_lg']
            color = color_mapping[well_id]

            ax.scatter(u, w_lg, color=color, label=f"Well {well_ids.index(well_id)}" if run_idx == 0 else "", s=30)

    # ax.set_ylim(-0.1, 5.1)
    # ax.set_xlim(-0.1, 1.1)
    ax.set_xlabel("Choke")
    ax.set_ylabel("Gas Lift")
    # ax.set_title("Decision Iterate per Well Across Runs")

    legend_patches = [
        mpatches.Patch(color=color_mapping[well_id], label=f"Well {i}")
        for i, well_id in enumerate(well_ids)
    ]

    ax.legend(handles=legend_patches, fontsize='small', loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_multiple_runs(n_runs: int, base_path: str, init_prod: float,
                       col_names = ["CHK", "WGL", "WOIL", "WWAT"],
                       only_optimizing_iterations: bool = False,
                       save_path: str | None = None):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    col_names = col_names + ['ID']
    if only_optimizing_iterations:
        perc_gain = []
        prod = []
        good_its = []

        meangl = []
        meanw = []
        lastgl = []
        lastw = []
    
    for i in range(n_runs):
        try:
            a, c, alpha, gamma = extract_spsa_params(f"{base_path}/run{i}/iteration_10/settings.txt")
            break
        except FileNotFoundError:
            continue

    for run_idx in range(n_runs):
        print(f"Processing Run {run_idx}/{n_runs-1}...")

        df = None
        for i in range (100, 0, -10):
            run_path = f"{base_path}/run{run_idx}/iteration_{i}/iteration_{i}.csv"
            if os.path.exists(run_path):
                df = pd.read_csv(run_path)
                break
        if df is None:
            print(f"No valid data found for Run {run_idx}. Skipping.")
            continue

        n_wells = len(df.groupby('ID'))
        constr = OptimizationProblemProperties()

        df = keep_last_unique_pairs(df, correct_i = 3*i)

        n_sims = int(len(df) / (2 * min(constr.max_wells, n_wells) + n_wells))
        print("n_sims = ", n_sims)

        if only_optimizing_iterations:
            df = df[2::3]
        else:
            n_sims *= 3

        df = df[col_names]
        well_data = df.groupby('ID')

        oil, gasl, water = [init_prod["oil"]], [init_prod["gaslift"]], [init_prod["water"]]
        last_5_prod = []
        first_good_it = False

        for sim_idx in range(n_sims):
            o = g = w = 0.0
            for well_id in range(n_wells):
                well = well_data.get_group(well_id)
                o += well['WOIL'].iloc[sim_idx]
                g += well['WGL'].iloc[sim_idx]
                w += well['WWAT'].iloc[sim_idx]

            oil.append(o)
            gasl.append(g)
            water.append(w)

            if only_optimizing_iterations and sim_idx >= n_sims - 5:
                last_5_prod.append(o)
            if only_optimizing_iterations and o > init_prod["opt_prod"] * 0.975 and not first_good_it:
                first_good_it = sim_idx

        # x_vals = range(1, len(oil) + 1)
        axs[0].plot(oil, label=f'Run {run_idx+1}', color='brown', alpha=0.5)
        axs[1].plot(gasl, label=f'Run {run_idx+1}', color='green', alpha=0.5)
        axs[2].plot(water, label=f'Run {run_idx+1}', color='blue', alpha=0.5)

        axs[0].plot(len(oil)-1, oil[-1], '|', color='brown', markersize=4)
        axs[1].plot(len(gasl)-1, gasl[-1], '|', color='green', markersize=4)
        axs[2].plot(len(water)-1, water[-1], '|', color='blue', markersize=4)

        if only_optimizing_iterations:
            mean_prod_run = np.mean(last_5_prod)
            prod.append(mean_prod_run)
            perc_gain_run = (mean_prod_run - init_prod["oil"]) / init_prod["oil"]
            perc_gain.append(perc_gain_run)

            good_its.append(first_good_it if first_good_it else n_sims)

            mgl, mw, lgl, lw = calculate_constr_break(gasl, water, constr_gl=init_prod["gaslift_max"], constr_w=init_prod["water_max"])
            meangl.append(mgl)
            meanw.append(mw)
            lastgl.append(lgl)
            lastw.append(lw)

    axs[0].set_ylim(top=110, bottom=95)
    axs[1].set_ylim(top=10, bottom=0)
    axs[2].set_ylim(top=78, bottom=38)
    axs[0].set_title('Oil Production')
    axs[1].set_title('Gas Lift')
    axs[2].set_title('Water Production')
    axs[2].set_xlabel('Simulation Steps')

    axs[1].axhline(y=init_prod["gaslift_max"], color='k', linestyle='--', linewidth=1.5)
    axs[2].axhline(y=init_prod["water_max"], color='k', linestyle='--', linewidth=1.5)

    for ax in axs:
        # ax.legend()
        ax.grid(True)

    # fig.suptitle(fr"Prodcution under no noise : $\sigma = 0$")
    fig.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}prod.png", dpi=300, bbox_inches="tight")
    

    if only_optimizing_iterations:
        print(f"Average Oil Production over last 5 iterations: {np.mean(prod):.2f}")
        print(f"Percentage Gain over last 5 iterations: {np.mean(perc_gain):.2%}")
        print(f"Maximum Oil Production: {max(prod):.2f}")
        print(f"Maximum Percentage Gain: {max(perc_gain):.2%}")
        print(f"All last productions: {prod}")
        print(f"All percentage gains: {perc_gain}")
        print(f"\nFirst good iteration: {good_its}")
        print(f"Average first good iteration: {np.mean(good_its)}")

        print(f"\nConstraint Break Metrics:")
        print(f"Mean Gas Lift: {np.mean(meangl):.2f}, Mean Water: {np.mean(meanw):.2f}")
        print(f"Last Gas Lift: {np.mean(lastgl):.2f}, Last Water: {np.mean(lastw):.2f}")

    plt.show()

def plot_sum_of_wells(df: pd.DataFrame, col_names = ["WWAT", "WGL", "WOIL", "ID", "DCNT"], only_optimizing_iterations: bool = False, start_i: int = 0):
    
    n_wells = len(df.groupby('ID'))
    i_counter = [0] * n_wells
    constr = OptimizationProblemProperties()

    if start_i < 0:
        n_sims = 1
    else:
        n_sims = int(len(df) / (2 * min(constr.max_wells, n_wells) + n_wells))

    df = df[col_names]
    #TODO: Does not take into account that not all wells are perturbed in every iteration
    if only_optimizing_iterations:
        df = df[df['DCNT'] == 'Optimizing']

    well_data = df.groupby('ID')

    oil = []
    gasl = []
    water = []

    #TODO: Does not take into account that not all wells are perturbed in every iteration
    if only_optimizing_iterations:
        for _ in range(n_sims):
            o = 0.0
            g = 0.0
            w = 0.0
            for i in range(n_wells):
                well = well_data.get_group(i)
                o += well['WOIL'].iloc[_]
                g += well['WGL'].iloc[_]
                w += well['WWAT'].iloc[_]

            oil.append(o)
            gasl.append(g)
            water.append(w)
    else:    
        for _ in range(3 * n_sims):
            o = 0.0
            g = 0.0
            w = 0.0
            for i in range(n_wells):
                well = well_data.get_group(i)
                if start_i:
                    well = well.iloc[start_i:start_i + 3]
                o += well['WOIL'].iloc[i_counter[i]]
                g += well['WGL'].iloc[i_counter[i]]
                w += well['WWAT'].iloc[i_counter[i]]

                if well["DCNT"].iloc[i_counter[i]] == "Perturbation":
                    i_counter[i] += 1
                elif _ % 3 == 2:
                    i_counter[i] += 1 

            oil.append(o)
            gasl.append(g)
            water.append(w)

    # Separate plots
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(oil, label='Oil', color='brown')
    plt.ylabel('Oil Production')
    plt.title('Oil Production')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(gasl, label='Gas Lift', color='green')
    plt.ylabel('Gas Lift Production')
    plt.title('Gas Lift')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(water, label='Water', color='blue')
    plt.xlabel('Simulation')
    plt.ylabel('Water Production')
    plt.title('Water Production')
    plt.grid(True)

    plt.tight_layout()
    plt.show()         

def plot_overshoot(df: pd.DataFrame, start_i: int, included_ids: list[int] | None = None, save_path: str | None = None):
    """
    Plots:
    1. Choke vs Gas Lift for selected wells over two iterations.
    2. Total oil production over three iterations (includes all wells).
    
    Parameters:
    - df: DataFrame with columns including ['ID', 'CHK', 'WGL', 'WOIL']
    - start_i: Starting iteration index
    - included_ids: List of well IDs to include in the first plot. If None, include all.
    """
    grouped = df.groupby('ID')
    if included_ids is not None:
        grouped = {k: v for k, v in grouped if k in included_ids}
    else:
        grouped = dict(grouped)

    n_wells = len(grouped)
    color_map = cm.get_cmap('Accent', n_wells)  # Use a colormap with enough distinct colors

        # --- First Figure: Choke vs Gas Lift (filtered wells) ---
    plt.figure(figsize=(12, 8))

    well_patches = []  # For mapping well colors to legend
    perturbation_patches = [  # Static legend for perturbations
        mpatches.Patch(color='red', alpha=0.5, label='Perturbation 1'),
        mpatches.Patch(color='blue', alpha=0.5, label='Perturbation 2'),
        mpatches.Patch(color='green', alpha=0.5, label=r'Optimal $\theta$')
    ]

    for idx, (well_id, well) in enumerate(grouped.items()):
        cropped = well.iloc[start_i - 1:start_i + 3]

        color = color_map(idx)
        well_patches.append(mpatches.Patch(color=color, label=f'Well {well_id}'))

        x_values = [cropped.iloc[0]['CHK'], cropped.iloc[1]['CHK'], cropped.iloc[2]['CHK'], cropped.iloc[3]['CHK']]
        y_values = [cropped.iloc[0]['WGL'], cropped.iloc[1]['WGL'], cropped.iloc[2]['WGL'], cropped.iloc[3]['WGL']]

        plt.scatter(x_values[0], y_values[0], color=color, alpha=1)
        plt.scatter(x_values[1], y_values[1], color="red", alpha=0.5)
        plt.scatter(x_values[2], y_values[2], color="blue", alpha=0.5)
        plt.scatter(x_values[3], y_values[3], color="green", alpha=0.5)

        plt.plot([x_values[0], x_values[1]], [y_values[0], y_values[1]], color=color, alpha=0.8, linestyle='--')
        plt.plot([x_values[0], x_values[2]], [y_values[0], y_values[2]], color=color, alpha=0.8, linestyle='--')

        plt.annotate(
            '', 
            xy=(x_values[3], y_values[3]), 
            xytext=(x_values[2], y_values[2]), 
            arrowprops=dict(arrowstyle='->', color=color, lw=1.5)
        )

    # Combine legends
    legend1 = plt.legend(handles=well_patches, title="Well ID", loc='upper left')
    plt.gca().add_artist(legend1)  # Add first legend separately
    plt.legend(handles=perturbation_patches, title="Perturbation Types", loc='upper right')

    plt.xlim(0, 1)
    plt.xlabel('Choke (CHK)')
    plt.ylabel('Gas Lift (WGL)')
    plt.title('Choke vs Gas Lift per Perturbation for Selected Wells', fontsize=16, weight='bold')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}welltheta.pdf", bbox_inches="tight")
        plt.savefig(f"{save_path}welltheta.png", dpi=600, bbox_inches="tight")
    plt.show()

    # Second figure
    all_grouped = df.groupby('ID')
    oil_per_iteration = [0.0, 0.0, 0.0]
    for _, well in all_grouped:
        cropped = well.iloc[start_i:start_i + 3]
        for i in range(min(3, len(cropped))):
            oil_per_iteration[i] += cropped.iloc[i]['WOIL']

    x_vals = [0,1,2]

    # Nicer plot settings
    colors = ['red', 'blue', 'green']
    labels = ['Perturbation 1', 'Perturbation 2', r'Optimal $\theta$']

    plt.figure(figsize=(10, 6))

    # Plot each point with color and a label (for the legend)
    for i in range(3):
        plt.plot(x_vals[i], oil_per_iteration[i],
                marker='o', markersize=8, color=colors[i], alpha=0.5,
                label=labels[i])

    # Draw a line connecting the points
    plt.plot(x_vals, oil_per_iteration, color='gray', linewidth=2)

    # Axes and aesthetics
    plt.xticks(x_vals, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Total Oil Produced', fontsize=14)
    plt.title('Total Oil Production Over 1 SPSA iteration', fontsize=16, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add legend
    plt.legend(fontsize=14, loc='best')
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}totalprod.pdf", bbox_inches="tight")
        plt.savefig(f"{save_path}totalprod.png", dpi=600, bbox_inches="tight")
    plt.show()

    #Third Figure
    x_vals = [0, 1, 2]  # Fixed x-values for the three iterations

    plt.figure(figsize=(10, 6))

    for idx, (well_id, well) in enumerate(grouped.items()):
        cropped = well.iloc[start_i:start_i + 3]
        if len(cropped) == 3:
            oil_values = cropped['WOIL'].values
            for i in range(3):
                plt.plot(x_vals[i], oil_values[i],
                marker='o', markersize=8, color=colors[i], alpha=0.5,)
            plt.plot(x_vals, oil_values, label=f'Well {well_id}', color=color_map(idx))

    plt.ylabel('Oil Produced (WOIL)', fontsize=13)
    plt.title('Oil Production Over 1 SPSA iteration for Selected Wells', fontsize=16, weight='bold')
    plt.xticks(x_vals)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Wells')
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}wellprod.pdf", bbox_inches="tight")
        plt.savefig(f"{save_path}wellprod.png", dpi=600, bbox_inches="tight")
    plt.show()

def calculate_constr_break(gaslift: list[float], water: list[float], constr_gl: float, constr_w: float):
    gaslift  = np.array(gaslift)
    water = np.array(water)

    gl_breaks = np.where(gaslift > constr_gl, gaslift - constr_gl, 0)
    w_breaks = np.where(water > constr_w, water - constr_w, 0)

    mean_gl_break = np.mean(gl_breaks)
    mean_w_break = np.mean(w_breaks)

    last_gl_break = gl_breaks[-1]
    last_w_break = w_breaks[-1]

    print(f"Gas Lift Constraint Breaks: {mean_gl_break:.2f} (Last: {last_gl_break:.2f})")
    print(f"Water Constraint Breaks: {mean_w_break:.2f} (Last: {last_w_break:.2f})")

    return mean_gl_break, mean_w_break, last_gl_break, last_w_break

if __name__ == '__main__':
    path = '../results/hyperparam_setting/param_set_2/'
    # df = pd.read_csv(path)
    save_path=f"../../Resultater/hyperparam/"
    base_path='../results/hyperparam_setting/'

    # cleanup(base_path)

    # plot_sum_of_wells(df=df, only_optimizing_iterations=False)
    # plot_overshoot(df=df, start_i=18, included_ids=[2,9,11,16], save_path=save_path)  # Adjust included_ids as needed
    # plot_gain_seqence(df=df, save_path=f"{save_path}gain_sequence")

    # cleanup_df(pd.read_csv(f'{base_path}{filename}/run1/iteration_50/iteration_50.csv'), f'{base_path}{filename}/run1/iteration_50/iteration_50.csv')
    """
    This is for hyperparam 
    """
    # for i in range(1, 2):
    #     if i == 3 or i == 6:
    #         continue

    #     print("RUNNING PARAM SET", i)
    #     plot_multiple_runs(n_runs=10, 
    #                            base_path=f'{base_path}/param_set_{i}',
    #                            init_prod=INIT_PROD["sample6"],
    #                            only_optimizing_iterations=False, 
    #                            save_path=f'{save_path}param_set_{i}',
    #                            )
        

    # plot_decicion_iterate(base_path=f'{path}', n_runs=10)

    """
    This is for other prods
    """
    save_path = f"../../Resultater/"
    base_path = '../results/prod_runs/'

    filename = "high_choke50%_nonoise"  # Change this to the desired filename
    # init_prod = INIT_PROD[filename]

    # plot_multiple_runs(n_runs=20,
    #                    base_path=f'{base_path}{filename}',
    #                    init_prod=init_prod,
    #                    only_optimizing_iterations=True,
    #                    save_path=f'{save_path}{filename}')
    
    plot_decicion_iterate(base_path=f'{base_path}/{filename}', n_runs=20,
                          save_path=f'{save_path}/{filename}decision_iterate')

    # plot_decision_iterates_side_by_side(
    #     base_paths=[
    #                 f'{base_path}mixed_prod_choke50%',
    #                 f'{base_path}mixed_prod_badchoke',
    #                 f'{base_path}mixed_choke0'],
    #     n_runs=10,
    #     titles=["Nominal Start", "Inverted Start", "Zero Start"],
    #     save_path=f"{save_path}decision_iterates_side_by_side"
    # )
