from utils import *

###################################################################################

def plot_events(dates, ax, x_min, x_max, color):
    """Plot vertical lines for events."""
    for j in range(len(dates)):
        date = dates[j]
        if date >= x_min and date <= x_max:
            ax.axvline(x=date, color=color, linestyle="dotted")

def plot_metrics(metrics, show_events=False):
    """Plot graph metrics over time."""
    line_colors = (sns.color_palette("tab10", 8) + sns.color_palette("Dark2", 8) + sns.color_palette("Paired", 8) + sns.color_palette("Set2", 10))

    n_cols = 3
    n_rows = 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.6 * n_rows))
    axes = axes.flatten()

    x = [pd.Timestamp(date) for date in metrics['timestamp']]

    axes[0].plot(x, metrics['n_nodes'], label='number of nodes', color=line_colors[1])
    axes[0].plot(x, metrics['n_edges'], label='number of edges', color=line_colors[2])

    axes[1].plot(x, metrics['node_degree_avg'], label='node degree (mean)', color=line_colors[3])

    axes[2].plot(x, metrics['max_in_degree'], label='max in-degree', color=line_colors[4])
    axes[2].plot(x, metrics['max_out_degree'], label='max out-degree', color=line_colors[5])

    axes[3].plot(x, metrics['top10_degree_avg'], label='top-10 degree (mean)', color=line_colors[6])
    axes[3].plot(x, metrics['top10_degree_avg_ratio'], label='top-10 degree (ratio)', color=line_colors[7])

    ax4 = axes[4]
    ax4.plot(x, metrics['gini_degree'], label='Gini coefficient', color=line_colors[8])
    ax4b = ax4.twinx()
    ax4b.plot(x, metrics['CV_degree'], label='CV degree', color=line_colors[9])
    ax4b.legend(fontsize=10, loc="upper right")
    ax4b.tick_params(axis='both', labelsize=12.5)

    axes[5].plot(x, metrics['density'], label='density', color=line_colors[10])

    axes[6].plot(x, metrics['reciprocity'], label='reciprocity', color=line_colors[11])

    axes[7].plot(x, metrics['clustering_avg'], label='clustering coefficient (mean)', color=line_colors[14])
    axes[7].fill_between(x, 
                        np.array(metrics['clustering_avg']) - np.array(metrics['clustering_std']), 
                        np.array(metrics['clustering_avg']) + np.array(metrics['clustering_std']), 
                        alpha=0.25, label='clustering coefficient (std)', color=line_colors[14], edgecolor='none')

    axes[8].plot(x, metrics['transitivity'], label='transitivity', color=line_colors[15])

    axes[9].plot(x, metrics['num_wcc'], label='number WCC', color=line_colors[16])
    axes[9].plot(x, metrics['num_scc'], label='number SCC', color=line_colors[17])

    axes[10].plot(x, metrics['largest_wcc_rel_size'], label='largest WCC relative size', color=line_colors[18])
    axes[10].plot(x, metrics['largest_scc_rel_size'], label='largest SCC relative size', color=line_colors[19])

    axes[11].plot(x, metrics['modularity_score'], label='modularity score', color=line_colors[20])
    axes[12].plot(x, metrics['degree_assortativity'], label='degree assortativity', color=line_colors[21])

    ax13 = axes[13]
    ax13.plot(x, metrics['fraction_nodes_core'], label='fraction nodes in core', color=line_colors[22])
    ax13b = ax13.twinx()
    ax13b.plot(x, metrics['k_core_decomposition'], label='average coreness', color=line_colors[23])
    ax13b.legend(fontsize=10, loc="upper right")
    ax13b.tick_params(axis='both', labelsize=12.5)

    axes[14].plot(x, metrics['beta_1_norm'], label='normalized first Betti number', color=line_colors[24])

    axes[15].plot(x, metrics['closeness_centrality_mean'], label='closeness centrality (mean)', color=line_colors[13])
    axes[15].fill_between(x, 
                        np.array(metrics['closeness_centrality_mean']) - np.array(metrics['closeness_centrality_std']), 
                        np.array(metrics['closeness_centrality_mean']) + np.array(metrics['closeness_centrality_std']), 
                        alpha=0.25, label='closeness centrality (std)', color=line_colors[13], edgecolor='none')

    fig.delaxes(axes[-2])
    fig.delaxes(axes[-1])

    x_min, x_max = min(x), max(x)

    panel_labels = [f"{chr(97 + i)})" for i in range(len(axes))]
    for ax, lab in zip(axes, panel_labels):
        ax.set_title(lab, loc='left',fontsize=14)
        ax.legend(fontsize=10, loc="upper left")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        ax.tick_params(axis='both', labelsize=12.5)
        ax.grid(alpha=0.3)

    if show_events:
        df_events = get_df_events()
        colors = {"Aave News": "orange", "Global Conflict": "black", "Market / Regulatory": "green",
            "Security Incident": "red", "Technological": "royalblue"}

        for ax in axes:
            for cat, group in df_events.groupby("category"):
                dates = group["date"].tolist()
                plot_events(dates, ax, x_min, x_max, colors[cat],)
        
        legend_items = [(colors[cat], cat) for cat in colors]
        handles = [Line2D([0], [0], color=c, linestyle="dotted", label=l) for c, l in legend_items]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=5, frameon=False, fontsize=14)

        plt.tight_layout(w_pad=2)
        plt.savefig("output/events.pdf", format="pdf", bbox_inches='tight')
        plt.close(fig)
    else:
        plt.tight_layout(w_pad=2)
        plt.savefig("output/metrics.pdf", format="pdf", bbox_inches='tight')
        plt.close(fig)

def get_correlation(metrics, df_transfers):
    """Compute and plot the correlation matrix of graph metrics and transfer values."""
    df_daily_values = (
        df_transfers.groupby(df_transfers["timestamp"].dt.date)["value"]
        .agg(mean="mean", count="count")
    )

    df = pd.DataFrame(metrics).set_index("timestamp")
    df.insert(0,'mean_value',df_daily_values['mean'].values)
    df.insert(1,'total_transfers',df_daily_values['count'].values)

    df =  df[['mean_value', 'total_transfers', 'n_nodes', 'n_edges',
       'node_degree_avg', 'node_degree_std', 'max_in_degree', 'max_out_degree',
       'top10_degree_avg', 'top10_degree_std', 'top10_degree_avg_ratio', 'CV_degree',
       'gini_degree', 'density', 'reciprocity', 'clustering_avg',
       'clustering_std', 'transitivity', 'num_wcc', 'num_scc',
       'largest_wcc_rel_size', 'largest_scc_rel_size', 'modularity_score', 'degree_assortativity',
       'k_core_decomposition', 'fraction_nodes_core', 'beta_1_norm', 'closeness_centrality_mean',
       'closeness_centrality_std']]

    labels = ["average transfer value", "total transfers", "number of nodes", "number of edges",
            "node degree (mean)", "node degree (std)", "max in-degree", "max out-degree",
            "top-10 degree (mean)", "top-10 degree (std)", "top-10 degree (ratio)", "CV degree",
            "Gini coefficient", "density", "reciprocity", "clustering coefficient (mean)",
            "clustering coefficient (std)", "transitivity", "number WCC", "number SCC",
            "largest WCC relative size", "largest SCC relative size", "modularity score", "degree assortativity",
            "average coreness", "fraction nodes in core", "normalized first Betti number", "closeness centrality (mean)",
            "closeness centrality (std)"]

    corr_matrix = df.corr(method='pearson')
    mask_low = corr_matrix.abs() < 0.75

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_facecolor('whitesmoke')
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, square=True, annot_kws={"size": 7},
                mask=mask_low, ax=ax, cbar=False)

    ax.axhline(2, color="black", lw=0.5, linestyle="dashed")
    ax.axvline(2, color="black", lw=0.5, linestyle="dashed")
    ax.set_xticklabels(labels,rotation=90,fontsize=11)
    ax.set_yticklabels(labels,fontsize=11)

    plt.tight_layout()
    plt.savefig("output/correlation.pdf", format="pdf", bbox_inches='tight')
    plt.close(fig)

###################################################################################

def run_RQ1(metrics, df_transfers):
    print("Answering RQ1...")
    get_correlation(metrics, df_transfers)
    plot_metrics(metrics)
    plot_metrics(metrics, show_events=True)