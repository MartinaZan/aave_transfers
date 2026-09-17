from utils import *

###################################################################################

random.seed(42)

WINDOW_DAYS = 14          # Length (in days) of the pre/post windows used for the mean and trend tests
SPIKE_HALF_WINDOW = 1     # Tolerance margin (in days) around the event for the spike check
ROLL_WINDOW = 28          # Length (in days) of the rolling window used as the baseline for mean/std in the spike test
N_PLACEBO = 200           # Number of random pre/post placebo windows used to build the empirical null distribution
MIN_GAP_FROM_EVENTS = 14  # Minimum distance (in days) between a placebo window and any real event, to avoid contaminating the null distribution
THRESHOLD_SPIKE = 4       # Threshold for the spike test

###################################################################################
## Statistics (mean and trend)

def mean_diff_stat(pre, post):
    """mean_diff_stat: mean(post) - mean(pre), captures a shift in level."""
    return np.mean(post) - np.mean(pre)

def slope_diff_stat(pre, post):
    """slope_diff_stat: difference between the linear-fit slope of the post window and the pre window
    (captures a shift in trend)"""
    t1 = np.arange(len(pre))
    t2 = np.arange(len(post))
    b1 = np.polyfit(t1, pre, 1)[0]
    b2 = np.polyfit(t2, post, 1)[0]
    return b2 - b1

###################################################################################
## Placebo null distribution and empirical p-value

def build_placebo_null_distribution(series, stat_func, events_parsed, window_days=WINDOW_DAYS, n_placebo=N_PLACEBO,
                                    min_gap_from_events=MIN_GAP_FROM_EVENTS):
    """Build a null distribution of the statistic (mean_diff_stat or slope_diff_stat) by sampling random pre/post windows
    across the series, avoiding any real events."""
    # The valid range for placebo centers is the series minus the pre/post windows.
    valid_start = series.index.min() + pd.Timedelta(days=window_days)
    valid_end = series.index.max() - pd.Timedelta(days=window_days)
    all_days = pd.date_range(valid_start, valid_end, freq='D')

    event_set = set(events_parsed)
    candidate_days = [d for d in all_days if all(abs((d - ev).days) > min_gap_from_events for ev in event_set)]

    # Placebo centers are sampled at random days across the series
    sampled = random.sample(candidate_days, min(n_placebo, len(candidate_days)))

    # For each placebo center, the same pre/post windows and the same statistic (mean_diff_stat or slope_diff_stat) are
    # computed as would be computed for a real event. This produces an empirical null distribution.
    stats_list = []
    for center in sampled:
        pre = series.loc[center - pd.Timedelta(days=window_days):center - pd.Timedelta(days=1)].dropna().values
        post = series.loc[center + pd.Timedelta(days=1):center + pd.Timedelta(days=window_days)].dropna().values
        stats_list.append(stat_func(pre, post))
    return np.array(stats_list)

def empirical_pvalue(observed_stat, null_dist):
    """Compute the empirical p-value for an observed statistic against a null distribution."""
    # The p-value is the fraction of placebo statistics at least as extreme (in absolute value) as the observed one.
    if len(null_dist) == 0 or pd.isna(observed_stat):
        return np.nan
    return np.mean(np.abs(null_dist) >= np.abs(observed_stat))

###################################################################################
## Spike test

def spike_test(series, event_date, half_window_days=SPIKE_HALF_WINDOW, roll_window=ROLL_WINDOW):
    """Check for a spike in the series around the event date.
    Note: This test does not produce a p-value; it's a fixed-threshold flagging whether a spike is present or not."""

    # Day-to-day differences are computed, then standardized using a rolling mean/std (with a 1-day lag, so the
    # baseline never includes the current point or looks ahead).
    diff = series.diff()
    roll_mu = diff.rolling(roll_window).mean().shift(1)
    roll_sigma = diff.rolling(roll_window).std().shift(1)
    z = (diff - roll_mu) / roll_sigma

    # Within a ±1 day window around the event, any standardized value exceeding 4 in absolute value is flagged
    # as a spike.
    start = event_date - pd.Timedelta(days=half_window_days)
    end = event_date + pd.Timedelta(days=half_window_days)
    window_z = z.loc[start:end]

    spikes = window_z[np.abs(window_z) > THRESHOLD_SPIKE]
    return {'has_spike': len(spikes) > 0}

###################################################################################
## Multiple testing FDR Correction

def apply_fdr(results, field_names, events_parsed, test_key, alpha=0.05):
    """Apply the Benjamini-Hochberg FDR correction to the p-values in the results dictionary for a given test."""
    pvals, coords = [], []

    for field_name in field_names:
        for ev in events_parsed:
            res = results[field_name][ev]
            if 'error' in res:
                continue
            p = res[test_key]['p_value']
            if not pd.isna(p):
                pvals.append(p)
                coords.append((field_name, ev))

    if len(pvals) == 0:
        return
    
    _, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    for (field_name, ev), p_corr in zip(coords, pvals_corrected):
        results[field_name][ev][test_key]['p_value_fdr'] = p_corr

###################################################################################
## Build tables for plotting

def build_numeric_table(results, field_names, events_parsed, test_key, value_key='p_value_fdr'):
    """Build a DataFrame where rows are events, columns are metrics, and values are the p-values for a given test."""
    table = pd.DataFrame(index=[ev.date() for ev in events_parsed], columns=field_names, dtype=float)
    for field_name in field_names:
        for ev in events_parsed:
            res = results[field_name][ev]
            if 'error' in res or value_key not in res.get(test_key, {}):
                table.loc[ev.date(), field_name] = np.nan
                continue
            table.loc[ev.date(), field_name] = res[test_key].get(value_key, np.nan)
    return table

def build_spike_numeric_table(results, field_names, events_parsed):
    """Build a DataFrame where rows are events, columns are metrics, and values are 1 if a spike was detected, 0 otherwise."""
    table = pd.DataFrame(index=[ev.date() for ev in events_parsed], columns=field_names, dtype=float)
    for field_name in field_names:
        for ev in events_parsed:
            res = results[field_name][ev]
            if 'error' in res:
                table.loc[ev.date(), field_name] = np.nan
                continue
            table.loc[ev.date(), field_name] = 1 if res['spike']['has_spike'] else 0
    return table

def plot_significance_heatmap(p_table, title, alpha=0.05):
    """Plot a heatmap of significance results, with color indicating significance and labels indicating metrics and events."""

    labels = ["number of nodes", "number of edges", "node degree (mean)", "node degree (std)",
            "max in-degree", "max out-degree", "top-10 degree (mean)", "top-10 degree (std)",
            "top-10 degree (ratio)", "CV degree", "Gini coefficient", "density", "reciprocity",
            "clustering coefficient (mean)", "clustering coefficient (std)", "transitivity",
            "number WCC", "number SCC", "largest WCC relative size", "largest SCC relative size",
            "modularity score", "degree assortativity", "average coreness", "fraction nodes in core",
            "normalized first Betti number", "closeness centrality (mean)", "closeness centrality (std)"]
    
    p_table = p_table.T
    if title == "Spike test":
        sig_matrix = (p_table.astype(float) > alpha).astype(int)
    else:
        sig_matrix = (p_table.astype(float) < alpha).astype(int)
    has_sig_in_col = sig_matrix.any(axis=0)  # ora il check è per colonna, non riga

    plt.figure(figsize=(11,6.5))
    ax = sns.heatmap(
        sig_matrix,
        cmap=['white', 'firebrick'],
        cbar=False,
        linewidths=0.5,
        linecolor='lightgray',
        mask=p_table.isna()
    )
    ax.set_yticklabels(labels)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.xlabel('Event date',labelpad=5)
    plt.ylabel('Metric')
    plt.title(title,pad=10)

    for tick_label, is_sig in zip(ax.get_xticklabels(), has_sig_in_col):
        tick_label.set_color('gray' if not is_sig else 'black')
        tick_label.set_rotation(90)

    plt.tight_layout()
    plt.savefig(f"output/{title.lower().replace(' ','_')}.pdf", format="pdf", bbox_inches='tight')
    plt.close()

###################################################################################

def run_RQ3(metrics):
    print("Answering RQ3...")

    # Get list event dates
    df_events = get_df_events()
    events_dates = set(df_events['date'])
    timestamps = pd.DatetimeIndex(pd.to_datetime(metrics['timestamp']))
    events_parsed = sorted(pd.to_datetime(list(events_dates), dayfirst=True))

    # Limit events to the range of the metrics timestamps
    limit_start = pd.Timestamp(metrics["timestamp"][0])
    limit_end = pd.Timestamp(metrics["timestamp"][-1])    
    events_parsed = [ev for ev in events_parsed if limit_start <= ev <= limit_end]

    # Select the metrics to analyze
    field_names = ['n_nodes', 'n_edges', 'node_degree_avg', 'node_degree_std', 'max_in_degree', 'max_out_degree',
                   'top10_degree_avg', 'top10_degree_std', 'top10_degree_avg_ratio', 'CV_degree', 'gini_degree',
                   'density', 'reciprocity', 'clustering_avg', 'clustering_std', 'transitivity', 'num_wcc', 'num_scc',
                   'largest_wcc_rel_size', 'largest_scc_rel_size', 'modularity_score', 'degree_assortativity',
                   'k_core_decomposition', 'fraction_nodes_core', 'beta_1_norm', 'closeness_centrality_mean',
                   'closeness_centrality_std']

    results = {}
    for field_name in field_names:
        series = pd.Series(metrics[field_name], index=timestamps, dtype='float64').sort_index()

        # Placebo null distributions built one time for each metric
        null_mean = build_placebo_null_distribution(series, mean_diff_stat, events_parsed)
        null_slope = build_placebo_null_distribution(series, slope_diff_stat, events_parsed)

        # For each event, compute the mean and trend statistics, and compare them to the null distributions to get
        # empirical p-values
        results[field_name] = {}
        for ev in events_parsed:
            pre = series.loc[ev - pd.Timedelta(days=WINDOW_DAYS):ev - pd.Timedelta(days=1)].dropna().values
            post = series.loc[ev + pd.Timedelta(days=1):ev + pd.Timedelta(days=WINDOW_DAYS)].dropna().values

            obs_mean = mean_diff_stat(pre, post)
            obs_slope = slope_diff_stat(pre, post)

            results[field_name][ev] = {
                'mean': {'stat': obs_mean, 'p_value': empirical_pvalue(obs_mean, null_mean)},
                'trend': {'stat': obs_slope, 'p_value': empirical_pvalue(obs_slope, null_slope)},
                'spike': spike_test(series, ev)
            }

    # Apply FDR correction and build tables for plotting
    apply_fdr(results, field_names, events_parsed, 'mean')
    apply_fdr(results, field_names, events_parsed, 'trend')

    table_mean = build_numeric_table(results, field_names, events_parsed, 'mean')
    plot_significance_heatmap(table_mean, "Mean shift test")

    table_trend = build_numeric_table(results, field_names, events_parsed, 'trend')
    plot_significance_heatmap(table_trend, "Trend shift test")

    table_spike = build_spike_numeric_table(results, field_names, events_parsed)
    plot_significance_heatmap(table_spike, "Spike test")