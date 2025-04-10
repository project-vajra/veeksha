import json
import math
import numpy as np
from scipy import stats as sp_stats
from sklearn.neighbors import KernelDensity
from collections import defaultdict
import matplotlib.pyplot as plt
import warnings
import argparse
import os
import sys

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", category=UserWarning, message="Dataset has 0 variance; skipping density estimation.")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in subtract") # For Beta SF/CDF

# --- Helper Functions ---
# (load_trace, find_longest_prefix_match, calculate_match_percentage, fit_beta_mom - remain the same as previous version)
def load_trace(filename):
    """Loads trace data from a JSONL file."""
    data = []
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line {line_num}: {line.strip()}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Trace file '{filename}' not found.", file=sys.stderr)
        # Using the sample data provided if file not found
        print("Using sample data provided in the prompt.", file=sys.stderr)
        data = [
            {"timestamp": 0, "input_length": 6758, "output_length": 500, "hash_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]},
            {"timestamp": 0, "input_length": 7322, "output_length": 490, "hash_ids": [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]},
            {"timestamp": 0, "input_length": 7236, "output_length": 794, "hash_ids": [0, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]},
            {"timestamp": 0, "input_length": 2290, "output_length": 316, "hash_ids": [0, 42, 43, 44, 45]},
            {"timestamp": 0, "input_length": 9013, "output_length": 3, "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]}
        ]
        # Simulate saving this data to the file for subsequent runs if needed
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True) # Ensure dir exists
            with open(filename, 'w') as f:
                for entry in data:
                    f.write(json.dumps(entry) + '\n')
            print(f"Created '{filename}' with sample data.", file=sys.stderr)
        except IOError as e:
            print(f"Warning: Could not write sample data to '{filename}'. Error: {e}", file=sys.stderr)
    return data

def find_longest_prefix_match(request_hashes, sessions):
    """Finds the session with the longest prefix match."""
    best_match_len = -1
    best_session_id = None
    if not request_hashes: return None, 0
    for session_id, session_data in sessions.items():
        session_hashes = session_data['hashes']
        match_len = 0
        max_possible = min(len(request_hashes), len(session_hashes))
        for i in range(max_possible):
            if request_hashes[i] == session_hashes[i]: match_len += 1
            else: break
        if match_len > best_match_len:
            best_match_len = match_len
            best_session_id = session_id
    return best_session_id, best_match_len

def calculate_match_percentage(match_len, request_len):
    """Calculates match percentage, handling zero request length."""
    if request_len == 0: return 0.0
    match_len = max(0, match_len)
    return (match_len / request_len) * 100.0

def fit_beta_mom(data, prior_alpha, prior_beta, prior_strength, epsilon=1e-6):
    """Fits Beta distribution using Method of Moments, incorporating prior."""
    safe_data = [max(epsilon, min(1 - epsilon, p)) for p in data]
    prior_mean = prior_alpha / (prior_alpha + prior_beta) if (prior_alpha + prior_beta) > epsilon else 0.5
    combined_data = [prior_mean] * int(prior_strength) + list(safe_data)
    if len(combined_data) < 2: return prior_alpha, prior_beta
    mean = np.mean(combined_data)
    var = np.var(combined_data, ddof=0)
    mean = max(epsilon, min(1 - epsilon, mean))
    var = max(epsilon / 10, var)
    if var >= mean * (1 - mean) - epsilon:
        pseudo_alpha = mean * 2
        pseudo_beta = (1-mean) * 2
        return max(epsilon, pseudo_alpha), max(epsilon, pseudo_beta)
    common_factor = (mean * (1 - mean) / var) - 1
    if common_factor <= epsilon:
        alpha = mean * 100
        beta = (1-mean) * 100
        return max(epsilon, alpha), max(epsilon, beta)
    alpha = mean * common_factor
    beta = (1 - mean) * common_factor
    return max(epsilon, alpha), max(epsilon, beta)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze prefix caching performance and learn match distributions.")
    parser.add_argument("trace_file", help="Path to the JSONL trace file.")
    parser.add_argument("--output-dir", default="reports",
                        help="Base directory to save the report and plots (default: reports)")
    parser.add_argument("--min-match-percent", type=float, default=10.0,
                        help="Minimum prefix match percentage required to assign to an existing session (default: 10.0)")
    parser.add_argument("--hist-bins", type=int, default=10,
                        help="Number of bins for the histogram distribution (default: 10)")
    parser.add_argument("--prior-strength", type=float, default=5.0,
                        help="Effective number of samples for Beta/Histogram prior (default: 5.0)")
    parser.add_argument("--kde-bandwidth", type=float, default=0.05,
                        help="Bandwidth for Kernel Density Estimation (default: 0.05)")
    parser.add_argument("--prob-thresholds", type=float, nargs='+', default=[50.0, 80.0, 90.0],
                        help="List of percentage thresholds for P(match >= X%%) calculation (default: 50 80 90)")
    parser.add_argument("--max-plots", type=int, default=100,
                        help="Maximum number of session plots to generate (plots top N sessions by observation count) (default: 100)")
    parser.add_argument("--epsilon", type=float, default=1e-6,
                        help="Small value to prevent division by zero or log(0) errors (default: 1e-6)")
    return parser.parse_args()

# --- Main Simulation ---

def main():
    args = parse_arguments()

    # --- Configuration from Args ---
    TRACE_FILE = args.trace_file
    BASE_OUTPUT_DIR = args.output_dir
    MIN_MATCH_PERCENT = args.min_match_percent
    HISTOGRAM_BINS = args.hist_bins
    PRIOR_STRENGTH = args.prior_strength
    KDE_BANDWIDTH = args.kde_bandwidth
    PROB_THRESHOLDS = sorted(args.prob_thresholds)
    MAX_PLOTS = args.max_plots
    EPSILON = args.epsilon

    # --- Create output subdirectory based on trace name ---
    trace_basename = os.path.basename(TRACE_FILE)
    trace_name_no_ext = os.path.splitext(trace_basename)[0]
    output_subdir = os.path.join(BASE_OUTPUT_DIR, trace_name_no_ext)
    os.makedirs(output_subdir, exist_ok=True)
    report_path = os.path.join(output_subdir, 'analysis_report.txt')

    print(f"Trace File: {TRACE_FILE}")
    print(f"Output Directory: {output_subdir}")
    print(f"Min Match Percentage: {MIN_MATCH_PERCENT}%")
    print(f"Max Plots: {MAX_PLOTS}")

    trace_data = load_trace(TRACE_FILE)

    if not trace_data:
        print("No trace data loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    # --- Phase 1: Processing Trace for Global Prior ---
    print("--- Phase 1: Processing Trace for Global Prior ---")
    # (Phase 1 logic remains the same)
    all_match_percentages = []
    temp_sessions = {}
    temp_next_session_id = 0
    for i, request in enumerate(trace_data):
        req_hashes = request.get("hash_ids", [])
        req_len = len(req_hashes)
        matched_session_id, match_len = find_longest_prefix_match(req_hashes, temp_sessions)
        is_match_sufficient = False
        potential_match_perc = 0.0
        if matched_session_id is not None:
            potential_match_perc = calculate_match_percentage(match_len, req_len)
            if potential_match_perc >= MIN_MATCH_PERCENT: is_match_sufficient = True
        if is_match_sufficient:
            match_perc_0_1 = potential_match_perc / 100.0
            all_match_percentages.append(match_perc_0_1)
            temp_sessions[matched_session_id]['hashes'].extend(req_hashes[match_len:])
        else:
            new_id = temp_next_session_id
            temp_sessions[new_id] = {'hashes': list(req_hashes)}
            temp_next_session_id += 1
    print(f"Phase 1 complete. Found {len(all_match_percentages)} match percentage observations globally.")


    # --- Calculate Global Priors ---
    print("--- Calculating Global Priors ---")
    with open(report_path, 'w') as report_f: # Start writing report
        report_f.write(f"Analysis Report for Trace: {TRACE_FILE}\n")
        report_f.write(f"Minimum Match Percentage Criterion: {MIN_MATCH_PERCENT}%\n")
        report_f.write(f"Maximum Plots Generated: {MAX_PLOTS}\n")
        report_f.write("-" * 30 + "\n\n")
        report_f.write("--- Global Prior Calculation ---\n")

        # (Global Prior calculation logic remains the same)
        global_hist_counts = np.zeros(HISTOGRAM_BINS)
        bin_edges_global = np.linspace(0, 100, HISTOGRAM_BINS + 1) # Store edges
        if all_match_percentages:
            hist, _ = np.histogram([p * 100 for p in all_match_percentages], bins=bin_edges_global)
            total_obs = len(all_match_percentages)
            global_hist_counts = (hist / total_obs) * PRIOR_STRENGTH if total_obs > 0 else np.zeros(HISTOGRAM_BINS)
            report_f.write(f"Global Histogram Prior (scaled to {PRIOR_STRENGTH:.1f} obs):\n{np.round(global_hist_counts, 3)}\n")
            print(f"Global Histogram Prior calculated.")
        else:
            global_hist_counts = np.ones(HISTOGRAM_BINS) * (PRIOR_STRENGTH / HISTOGRAM_BINS)
            report_f.write(f"No historical matches, using uniform histogram prior (strength {PRIOR_STRENGTH:.1f}).\n")
            print("No historical matches, using uniform histogram prior.")
        global_prior_alpha = 1.0; global_prior_beta = 1.0
        if len(all_match_percentages) >= 2:
            safe_percentages = [max(EPSILON, min(1 - EPSILON, p)) for p in all_match_percentages]
            mean_hist = np.mean(safe_percentages); var_hist = np.var(safe_percentages)
            report_f.write(f"Global Historical Mean (0-1): {mean_hist:.4f}, Variance: {var_hist:.4f}\n")
            print(f"Global Historical Mean: {mean_hist:.4f}, Variance: {var_hist:.4f}")
            if var_hist > EPSILON and var_hist < mean_hist * (1 - mean_hist) - EPSILON:
                common = (mean_hist * (1 - mean_hist) / var_hist) - 1; alpha_hist = mean_hist * common; beta_hist = (1 - mean_hist) * common
                current_strength = alpha_hist + beta_hist
                if current_strength > EPSILON:
                    scale_factor = PRIOR_STRENGTH / current_strength
                    global_prior_alpha = max(EPSILON, alpha_hist * scale_factor); global_prior_beta = max(EPSILON, beta_hist * scale_factor)
                else:
                    global_prior_alpha = max(EPSILON, mean_hist * PRIOR_STRENGTH); global_prior_beta = max(EPSILON, (1-mean_hist) * PRIOR_STRENGTH)
                report_f.write(f"Calculated Beta Prior: alpha={global_prior_alpha:.3f}, beta={global_prior_beta:.3f} (effective strength ~{PRIOR_STRENGTH:.1f})\n")
                print(f"Calculated Beta Prior: alpha={global_prior_alpha:.3f}, beta={global_prior_beta:.3f}")
            else:
                report_f.write("Historical variance issue or constant data, using default Beta prior (Uniform alpha=1, beta=1).\n")
                print("Historical variance issue or constant data, using default Beta prior (Uniform)")
        else:
            report_f.write(f"Not enough historical data (<2 points) for variance, using default Beta prior (Uniform alpha=1, beta=1).\n")
            print("Not enough historical data (<2 points) for variance, using default Beta prior (Uniform)")
        report_f.write("-" * 30 + "\n\n")


        # --- Phase 2: Process Trace Again, Learning Distributions ---
        print("\n--- Phase 2: Processing Trace and Learning Distributions ---")
        # (Phase 2 logic for matching and updating distributions remains the same)
        sessions = {}
        next_session_id = 0
        for i, request in enumerate(trace_data):
            req_hashes = request.get("hash_ids", []); req_len = len(req_hashes)
            matched_session_id, match_len = find_longest_prefix_match(req_hashes, sessions)
            session_id_for_update = None; match_perc_0_100 = 0.0; match_perc_0_1 = 0.0
            is_match_sufficient = False
            if matched_session_id is not None:
                potential_match_perc = calculate_match_percentage(match_len, req_len)
                if potential_match_perc >= MIN_MATCH_PERCENT:
                    is_match_sufficient = True
                    match_perc_0_100 = potential_match_perc
                    match_perc_0_1 = max(EPSILON, min(1 - EPSILON, match_perc_0_100 / 100.0))
                    session_id_for_update = matched_session_id
            if is_match_sufficient:
                sessions[session_id_for_update]['hashes'].extend(req_hashes[match_len:])
            else:
                new_id = next_session_id; session_id_for_update = new_id
                sessions[new_id] = { 'hashes': list(req_hashes), 'match_percentages_0_1': [],
                    'dist_state': {
                        'hist': {'counts': list(global_hist_counts), 'total': PRIOR_STRENGTH},
                        'kde_data': [],
                        'beta': {'alpha': global_prior_alpha, 'beta': global_prior_beta} } }
                next_session_id += 1
            if is_match_sufficient:
                session_state = sessions[session_id_for_update]['dist_state']
                sessions[session_id_for_update]['match_percentages_0_1'].append(match_perc_0_1)
                bin_index = min(int(match_perc_0_100 // (100 / HISTOGRAM_BINS)), HISTOGRAM_BINS - 1)
                session_state['hist']['counts'][bin_index] += 1; session_state['hist']['total'] += 1
                session_state['kde_data'].append(match_perc_0_1)
                current_alpha, current_beta = fit_beta_mom( sessions[session_id_for_update]['match_percentages_0_1'],
                    global_prior_alpha, global_prior_beta, PRIOR_STRENGTH, epsilon=EPSILON)
                session_state['beta']['alpha'] = current_alpha; session_state['beta']['beta'] = current_beta
        print(f"Phase 2 Processing complete. Created {next_session_id} sessions.")

        # --- Select Sessions to Plot ---
        session_obs_counts = []
        for sid, sdata in sessions.items():
            n_obs = len(sdata.get('match_percentages_0_1', []))
            session_obs_counts.append((sid, n_obs))

        # Sort sessions by number of observations (descending)
        session_obs_counts.sort(key=lambda item: item[1], reverse=True)

        # Select the IDs of the sessions to plot, up to MAX_PLOTS
        sessions_to_plot_ids = {item[0] for item in session_obs_counts[:MAX_PLOTS]}
        print(f"Total sessions: {len(sessions)}. Plotting distributions for top {len(sessions_to_plot_ids)} sessions (max {MAX_PLOTS}) based on observation count.")
        report_f.write("--- Evaluation Results per Session ---\n")
        if len(sessions) > MAX_PLOTS:
             report_f.write(f"--- NOTE: Plot images generated only for the top {len(sessions_to_plot_ids)} sessions (out of {len(sessions)}) based on observation count ---\n\n")


        # --- Evaluation and Reporting ---
        print("\n--- Generating Reports and Plots (up to limit) ---")

        x_eval = np.linspace(0, 1, 200) # X-axis for plots (0-1 scale)
        x_eval_plot = x_eval.reshape(-1, 1) # For sklearn KDE input
        plots_generated = 0

        for session_id, session_data in sessions.items():
            # --- Start Report Section for Session (Always) ---
            report_f.write(f"\n--- Session {session_id} ---\n")
            match_percs = session_data['match_percentages_0_1']
            n_obs = len(match_percs)
            report_f.write(f"  Number of requests assigned (generating observations): {n_obs}\n")
            report_f.write(f"  Observed match percentages (0-1): {[f'{p:.3f}' for p in match_percs]}\n")
            if n_obs == 0:
                report_f.write("  No match observations recorded for this session (only creation event). Distributions reflect prior.\n")

            state = session_data['dist_state']

            # --- Generate Plot ONLY for selected sessions ---
            should_plot = session_id in sessions_to_plot_ids
            if should_plot:
                plots_generated += 1
                if plots_generated % 20 == 0: # Progress update
                     print(f"  Generating plot {plots_generated}/{len(sessions_to_plot_ids)}...")

                fig, ax = plt.subplots(1, 1, figsize=(12, 7))
                ax.set_title(f'Session {session_id} - Learned Match % Distributions ({n_obs} obs)')
                ax.set_xlabel("Match Percentage / 100")
                ax.set_ylabel("Probability Density / Normalized Frequency", color='black')
                ax.set_xlim(0, 1)
                ax.tick_params(axis='y', labelcolor='black')

                ax2 = ax.twinx()
                ax2.set_ylabel("Survival Probability (P >= x)", color='black')
                ax2.set_ylim(0, 1.05)
                ax2.tick_params(axis='y', labelcolor='black')

                plot_path = os.path.join(output_subdir, f'session_{session_id}_distribution.png')
                lines, labels = [], []
                lines2, labels2 = [], []
            # --- End Plot Setup ---


            # --- Histogram Calculation (Always) ---
            report_f.write("\n  Histogram:\n")
            hist_counts = np.array(state['hist']['counts'])
            hist_total = state['hist']['total']
            report_f.write(f"    Total effective count (prior + obs): {hist_total:.1f}\n")
            bin_width_perc = 100.0 / HISTOGRAM_BINS
            bin_width_01 = 1.0 / HISTOGRAM_BINS
            bin_edges_01 = np.linspace(0, 1, HISTOGRAM_BINS + 1)
            hist_probs = hist_counts / hist_total if hist_total > EPSILON else np.zeros_like(hist_counts)
            bin_centers_01 = bin_edges_01[:-1] + bin_width_01 / 2
            try:
                sf_hist = np.cumsum(hist_probs[::-1])[::-1] # Cumulative sum from right
                sf_hist_plot = np.concatenate(([1.0], sf_hist))
            except Exception as e:
                sf_hist_plot = None
                report_f.write(f"    Error calculating Histogram SF: {e}\n")

            # Plot Histogram if should_plot
            if should_plot:
                bar_container = ax.bar(bin_centers_01, hist_probs / bin_width_01, width=bin_width_01 * 0.9, alpha=0.4, label=f'Hist Density (N={hist_total:.0f})', color='tab:grey')
                lines.append(bar_container.patches[0])
                labels.append(f'Hist Density (N={hist_total:.0f})')
                if sf_hist_plot is not None:
                    line, = ax2.step(bin_edges_01, sf_hist_plot, where='post', label='Hist Survival (P>=x)', color='red', linestyle=':')
                    lines2.append(line); labels2.append('Hist Survival (P>=x)')
                else:
                    line, = ax2.plot([],[], ':', label='Hist Survival - Error', color='red')
                    lines2.append(line); labels2.append('Hist Survival - Error')
            # Report probabilities (Always)
            for i, prob in enumerate(hist_probs):
                 report_f.write(f"    P({i*bin_width_perc:.0f}% - {(i+1)*bin_width_perc:.0f}%) = {prob:.3f}\n")
            #--- End Histogram ---


            # --- KDE Calculation (Always) ---
            report_f.write("\n  KDE:\n")
            kde_data = np.array(state['kde_data']).reshape(-1, 1)
            pdf_kde = None
            sf_kde_approx = None
            kde_error = None
            kde_status_label = ""

            if kde_data.shape[0] >= 1:
                try:
                    is_zero_variance = kde_data.shape[0] > 1 and np.var(kde_data) < EPSILON
                    if is_zero_variance:
                        report_f.write(f"    Skipping KDE calculations: Data variance near zero ({np.var(kde_data):.2e})\n")
                        kde_status_label = f'KDE (N={n_obs}) - Zero Var'
                    else:
                        kde = KernelDensity(kernel='gaussian', bandwidth=KDE_BANDWIDTH).fit(kde_data)
                        log_dens = kde.score_samples(x_eval_plot)
                        pdf_kde = np.exp(log_dens)
                        mid_prob_density = np.exp(kde.score_samples([[0.5]]))[0]
                        report_f.write(f"    Example Density at 50% ~ {mid_prob_density:.3f}\n")
                        # Calculate SF
                        dx = x_eval[1] - x_eval[0]
                        cdf_kde_approx = np.cumsum(pdf_kde) * dx
                        cdf_kde_approx = np.clip(cdf_kde_approx, 0, 1)
                        sf_kde_approx = 1.0 - cdf_kde_approx
                        kde_status_label = f'KDE (N={n_obs}, bw={KDE_BANDWIDTH:.3f})'

                except Exception as e:
                     kde_error = e
                     report_f.write(f"    Error fitting/evaluating KDE - {e}\n")
                     kde_status_label = f'KDE (N={n_obs}) - Error'
            else:
                report_f.write("    Not enough data points for KDE.\n")
                kde_status_label = f'KDE (N={n_obs}) - No Data'

            # Plot KDE if should_plot
            if should_plot:
                if pdf_kde is not None:
                     line, = ax.plot(x_eval, pdf_kde, '--', label=f'KDE Density', color='orange') # Shorter label
                     lines.append(line); labels.append(f'KDE Density')
                else:
                     line, = ax.plot([], [], '--', label=f'KDE Density - N/A', color='orange') # Indicate not plotted
                     lines.append(line); labels.append(f'KDE Density - N/A')

                if sf_kde_approx is not None:
                     line2, = ax2.plot(x_eval, sf_kde_approx, '--', label='KDE Survival (Approx)', color='magenta')
                     lines2.append(line2); labels2.append('KDE Survival (Approx)')
                else:
                     line2, = ax2.plot([],[], '--', label='KDE Survival - N/A', color='magenta')
                     lines2.append(line2); labels2.append('KDE Survival - N/A')
            # --- End KDE ---


            # --- Beta Calculation (Always) ---
            report_f.write("\n  Beta Distribution:\n")
            beta_alpha = state['beta']['alpha']
            beta_beta = state['beta']['beta']
            report_f.write(f"    Parameters: alpha={beta_alpha:.3f}, beta={beta_beta:.3f}\n")
            beta_pdf = None
            beta_sf = None
            beta_error = None
            beta_status_label = ""

            try:
                beta_dist = sp_stats.beta(beta_alpha, beta_beta)
                beta_pdf = beta_dist.pdf(x_eval)
                beta_sf = beta_dist.sf(x_eval)
                beta_status_label = f'Beta (α={beta_alpha:.2f}, β={beta_beta:.2f})'
                report_f.write(f"    Mean = {beta_dist.mean():.3f}, Median = {beta_dist.median():.3f}, Std Dev = {beta_dist.std():.3f}\n")
                report_f.write("    Survival Probabilities (P(match >= X%)) from Beta:\n")
                for threshold_perc in PROB_THRESHOLDS:
                    threshold_0_1 = threshold_perc / 100.0
                    prob_ge_thresh = beta_dist.sf(threshold_0_1)
                    report_f.write(f"      P(match >= {threshold_perc:.0f}%) = {prob_ge_thresh:.4f}\n")
            except Exception as e:
                 beta_error = e
                 report_f.write(f"    Error evaluating Beta PDF/CDF/SF - {e}\n")
                 beta_status_label = f'Beta - Error'

            # Plot Beta if should_plot
            if should_plot:
                if beta_pdf is not None:
                     line_pdf, = ax.plot(x_eval, beta_pdf, '-', label=f'Beta PDF', color='green') # Shorter label
                     lines.append(line_pdf); labels.append(f'Beta PDF')
                else:
                     line_pdf, = ax.plot([], [], '-', label=f'Beta PDF - N/A', color='green')
                     lines.append(line_pdf); labels.append(f'Beta PDF - N/A')

                if beta_sf is not None:
                     line_sf, = ax2.plot(x_eval, beta_sf, '-.', label=f'Beta Survival (P>=x)', color='blue')
                     lines2.append(line_sf); labels2.append(f'Beta Survival (P>=x)')
                else:
                     line_sf, = ax2.plot([],[], '-.', label='Beta Survival - N/A', color='blue')
                     lines2.append(line_sf); labels2.append('Beta Survival - N/A')

                # Add vertical lines and observations only if plotting
                for threshold_perc in PROB_THRESHOLDS:
                    threshold_0_1 = threshold_perc / 100.0
                    ax.axvline(x=threshold_0_1, color='grey', linestyle=':', linewidth=1.0, alpha=0.6)
                    ax.text(threshold_0_1 + 0.01, ax.get_ylim()[1]*0.85, f'{threshold_perc:.0f}%', color='grey', fontsize=8, ha='left')

                if n_obs > 0:
                    y_jitter = np.random.rand(n_obs) * ax.get_ylim()[1] * 0.02
                    line, = ax.plot(match_percs, y_jitter, 'x', color='black', markersize=3, alpha=0.7, label='Observations')
                    lines.append(line); labels.append('Observations')

                # Final Plot Adjustments only if plotting
                ax.grid(True, which='major', linestyle=':', linewidth=0.5, axis='x')
                ax.grid(True, which='major', linestyle=':', linewidth=0.5, axis='y', color='grey', alpha=0.7)
                ax2.grid(True, which='major', linestyle=':', linewidth=0.5, axis='y', color='grey', alpha=0.7)
                ax.set_ylim(bottom=0)
                ax2.set_ylim(0, 1.05)
                ax.legend(lines + lines2, labels + labels2, loc='best', fontsize=8)
                fig.tight_layout()
                plt.savefig(plot_path)
                plt.close(fig)
                report_f.write(f"  Plot saved to: {plot_path}\n")
            else:
                # If not plotting, add a note
                report_f.write("  Plot skipped (not in top sessions by observation count).\n")
            # --- End Beta ---


            report_f.write("-" * 30 + "\n") # End session report section

        print(f"\n--- Analysis Finished ---")
        print(f"Report saved to: {report_path}")
        print(f"Plots saved in directory: {output_subdir} (Generated {plots_generated} plots)")

if __name__ == "__main__":
    main()