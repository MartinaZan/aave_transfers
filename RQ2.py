from utils import *

###################################################################################

def plot_overview_activity(df):
    """Plot the overall activity of addresses over time."""

    # Stack sender-events and receiver-events into a single dataframe, so each address (whether sending or receiving)
    # has its own activity rows (truncating each timestamp to midnight, keeping only the day)
    df_from = df[['from', 'timestamp']].rename(columns={'from': 'address'})
    df_to = df[['to', 'timestamp']].rename(columns={'to': 'address'})
    activity = pd.concat([df_from, df_to], ignore_index=True)
    activity['date'] = activity['timestamp'].dt.normalize()
    activity = activity.dropna(subset=['address'])

    # For each address, compute the first and last date it appeared in the data, and the total activity duration
    # (how many days passed between an address's first and last observed activity)
    range_address = activity.groupby('address')['date'].agg(['min', 'max']).reset_index()
    range_address.columns = ['address', 'first_date', 'last_date']
    range_address['duration'] = (range_address['last_date'] - range_address['first_date']).dt.days

    first_activity_per_day = range_address.groupby('first_date').size().sort_index().cumsum()

    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    # Left subplot: Cumulative unique addresses over time (based on first activity date)
    ax = axes[0]
    ax.plot(first_activity_per_day.index, first_activity_per_day.values)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Cumulative number of addresses (ever active)', fontsize=11)
    ax.set_title('Cumulative growth of addresses over time',fontsize=14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    ax.tick_params(axis='both', labelsize=12)
    ax.set_yscale('log')
    ax.grid(alpha=0.3)

    # Right subplot: Distribution of address activity duration
    ax = axes[1]
    ax.hist(range_address['duration'], bins=30, edgecolor='black')
    ax.set_xlabel('Activity duration (days)', fontsize=11)
    ax.set_ylabel('Number of addresses', fontsize=11)
    ax.set_title('Distribution of address activity duration',fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/address_activity.pdf", format="pdf", bbox_inches='tight')
    plt.close(fig)

def plot_frequency_transfers(df):
    # Compute how many times each address appears as sender / receiver
    vc_from = df["from"].value_counts()
    vc_to = df["to"].value_counts()

    # Build the set of ALL unique addresses (sender OR receiver), one row per address
    all_addresses = vc_from.index.union(vc_to.index)
    freq_from = all_addresses.map(vc_from).to_series(index=all_addresses).fillna(0)
    freq_to = all_addresses.map(vc_to).to_series(index=all_addresses).fillna(0)

    # Build log-spaced bins for the frequency axis
    n_bins = 60
    max_freq = max(vc_from.max(), vc_to.max())
    bin_edges = np.unique(np.logspace(0, np.log10(max_freq), n_bins + 1))

    # Assign each address's sender/receiver frequency to a bin (one row per address now)
    from_binned = pd.cut(freq_from, bins=bin_edges, include_lowest=True)
    to_binned = pd.cut(freq_to, bins=bin_edges, include_lowest=True)
    pivot = pd.crosstab(from_binned, to_binned)

    # Rebuild the bin edges directly from pivot's actual categories for the plot
    row_edges = np.array([interval.left for interval in pivot.index] + [pivot.index[-1].right])
    col_edges = np.array([interval.left for interval in pivot.columns] + [pivot.columns[-1].right])

    # Add bins for 0 counts
    mask_only_sender = freq_to == 0
    left_counts, _ = np.histogram(freq_from[mask_only_sender], bins=row_edges)
    mask_only_receiver = freq_from == 0
    bottom_counts, _ = np.histogram(freq_to[mask_only_receiver], bins=col_edges)

    # Heatmap layout
    fig, axes = plt.subplots(2, 2, figsize=(7, 6),
        gridspec_kw={'width_ratios': [1, 20], 'height_ratios': [20, 1], 'wspace': 0.05, 'hspace': 0.05},
        sharex='col', sharey='row')
    ax_left, ax_main = axes[0, 0], axes[0, 1]
    ax_corner, ax_bottom = axes[1, 0], axes[1, 1]
    ax_corner.axis('off')
    ax_corner.text(0.5, 0.5, "0", ha='center', va='center', fontsize=12)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="gainsboro")
    norm = LogNorm(vmin=1, vmax=max(pivot.values.max(), left_counts.max(), bottom_counts.max()))

    # Main heatmap (log-log)
    X, Y = np.meshgrid(col_edges, row_edges)
    masked = np.ma.masked_where(pivot.values == 0, pivot.values)
    mesh = ax_main.pcolormesh(X, Y, masked, cmap=cmap, norm=norm)
    ax_main.set_xscale("log")
    ax_main.set_yscale("log")

    # Left panel: no transfer received
    Xl, Yl = np.meshgrid([0, 1], row_edges)
    masked_left = np.ma.masked_where(left_counts == 0, left_counts).reshape(-1, 1)
    ax_left.pcolormesh(Xl, Yl, masked_left, cmap=cmap, norm=norm)
    ax_left.set_xlim(0, 1)
    ax_left.set_xticks([0.5]); ax_left.set_xticklabels(["0"])

    # Right panel: no transfer sent
    Xb, Yb = np.meshgrid(col_edges, [0, 1])
    masked_bottom = np.ma.masked_where(bottom_counts == 0, bottom_counts).reshape(1, -1)
    ax_bottom.pcolormesh(Xb, Yb, masked_bottom, cmap=cmap, norm=norm)
    ax_bottom.set_ylim(0, 1)
    ax_bottom.set_yticks([0.5]); ax_bottom.set_yticklabels(["0"])

    ax_bottom.set_xlabel("Absolute frequency as receiver",fontsize=12)
    ax_left.set_ylabel("Absolute frequency as sender", fontsize=12)
    plt.setp(ax_main.get_xticklabels(), visible=False)
    plt.setp(ax_main.get_yticklabels(), visible=False)

    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist())
    cbar.set_label("Count of addresses", fontsize=12)
    plt.savefig("output/heatmap_activity.pdf", format="pdf", bbox_inches='tight')
    plt.close(fig)

def get_top10_addresses(df):
    """Compute the top-10 addresses by total activity, top-10 sender-dominant addresses, and
    top-10 receiver-dominant addresses."""
    vc_from = df["from"].value_counts()
    vc_to = df["to"].value_counts()

    # Build the set of all unique addresses that appear either as sender OR receiver, and compute send-frequency and
    # receive-frequency.
    all_values = vc_from.index.union(vc_to.index)
    freq_from = all_values.map(vc_from).to_series(index=all_values).fillna(0)
    freq_to = all_values.map(vc_to).to_series(index=all_values).fillna(0)

    # Compute the log-scale difference between sending and receiving activity:
    #   - diff > 0  => the address sends much more than it receives (sender-dominant)
    #   - diff < 0  => the address receives much more than it sends (receiver-dominant)
    diff = np.log10(freq_from + 1) - np.log10(freq_to + 1)

    # Find the 10 addresses with the highest TOTAL activity (sent + received combined)
    top10_sum = (freq_from + freq_to).nlargest(10)

    df_all = pd.DataFrame({"from": freq_from, "to": freq_to}).loc[top10_sum.index]
    df_senders = pd.DataFrame({"from": freq_from, "to": freq_to})[["from", "to"]].loc[diff.nlargest(10).index]
    df_receivers = pd.DataFrame({"from": freq_from, "to": freq_to})[["from", "to"]].loc[diff.nsmallest(10).index]

    return df_all, df_senders, df_receivers

def plot_activity_over_time(df, list_users):
    """Plot the activity of a list of addresses over time, showing when they sent or received transfers."""
    groups = [list_users[:10], list_users[10:20], list_users[20:30]]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["Top activity", "Top senders", "Top receivers"]
    prefixes = ["T", "S", "R"]
    groups = [g for g in groups if g]

    fig, axes = plt.subplots(
        len(groups), 1, sharex=True,
        figsize=(10, len(list_users) * 0.25),
        gridspec_kw={"hspace": 0, "height_ratios": [len(g) for g in groups]},
    )

    for ax, users, color, label, prefix in zip(axes, groups, colors, labels, prefixes):
        n = len(users)
        idx = {u: i for i, u in enumerate(users)}

        mask_from = df["from"].isin(idx)
        mask_to = df["to"].isin(idx)
        times = np.concatenate([df["timestamp"].values[mask_from], df["timestamp"].values[mask_to]])
        codes = pd.Series(np.concatenate([df["from"].values[mask_from], df["to"].values[mask_to]])).map(idx).values

        counts = np.bincount(codes, minlength=n)
        # ytick_labels = [f"{short_address(u)} ({c})" for u, c in zip(users, counts)]
        ytick_labels = [f"{prefix}{i+1} ({c})" for i, c in enumerate(counts)]
        
        ax.scatter(times, codes, c=color, s=12, rasterized=True)
        ax.set_yticks(range(n))
        ax.set_yticklabels(ytick_labels, fontsize=11)
        ax.tick_params(axis='both', labelsize=11)
        margin = 0.75
        ax.set_ylim(n - 1 + margin, -margin)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylabel(label, fontsize=11)

    plt.tight_layout()
    plt.savefig("output/activity_top_addresses.pdf", format="pdf", bbox_inches='tight', dpi=150)
    plt.close(fig)

def compute_snapshot_local_metrics(G_snap):
    """Compute local graph metrics for a given snapshot graph."""
    G_und = G_snap.to_undirected()

    clustering = nx.clustering(G_und)
    degree_centrality = nx.degree_centrality(G_snap)
    hubs, authorities = nx.hits(G_snap)

    return {
        "clustering": clustering,
        "degree_centrality": degree_centrality,
        "hubs": hubs,
        "authorities": authorities,
    }

def build_node_evolution_table(node_list, snapshots):
    """Build a DataFrame containing the evolution of local metrics for a list of nodes across multiple snapshots."""
    snapshot_metrics = [compute_snapshot_local_metrics(G) for G in snapshots]

    all_records = []
    for t, (G_snap, metrics) in enumerate(zip(snapshots, snapshot_metrics), start=1):
        date = G_snap.graph.get("date", None)

        for node in node_list:
            if node not in G_snap.nodes:
                continue

            in_deg = G_snap.in_degree(node)
            out_deg = G_snap.out_degree(node)

            all_records.append({
                "node": node,
                "date": date,
                "in_degree": in_deg,
                "out_degree": out_deg,
                "clustering": metrics["clustering"].get(node),
                "degree_centrality": metrics["degree_centrality"].get(node),
                "hub_score": metrics["hubs"].get(node),
                "authority_score": metrics["authorities"].get(node),
            })

    return pd.DataFrame(all_records)

def plot_group_evolution(df, metrics, cfg):
    """Plot the evolution of specified metrics for a group of nodes over time."""
    highlight_nodes = set(cfg['highlight_nodes'] or [])
    yscale_config = cfg['yscale_config'] or {}

    fig, axes = plt.subplots(3, 2, figsize=(2*5, 3*2), sharex=False)
    fig.suptitle(cfg['title'])
    axes_flat = axes.flatten()
    cmap = plt.get_cmap("tab10")

    for idx, (ax, (col, ylabel)) in enumerate(zip(axes_flat, metrics)):
        for i, node in enumerate(cfg['nodes']):
            df_node = df[df['node'] == node].sort_values("date")
            if df_node.empty:
                continue
            is_highlight = node in highlight_nodes
            ax.plot(df_node['date'], df_node[col].clip(lower=0), lw=1, color=cmap(i % 10),
                    # label=short_address(node),
                    label=f"{cfg['prefix']}{i+1}",
                    alpha=1 if is_highlight else 0.3, ls="-" if is_highlight else ":")

        y_cfg = yscale_config.get(idx, {"scale": "linear"})
        if y_cfg['scale'] == "symlog":
            linthresh = y_cfg.get("linthresh", 2)
            linscale = y_cfg.get("linscale", 1.0)
            ax.set_yscale("symlog", linthresh=linthresh, linscale=linscale)
            ax.axhline(linthresh, color="black", lw=1, ls="--", alpha=0.5)
        else:
            ax.set_yscale("linear")

        ax.set_ylabel(ylabel, fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
        ax.tick_params(axis='both', labelsize=11)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=10)

    plt.tight_layout()
    plt.savefig(f"output/{cfg['file_name']}.pdf", format="pdf", bbox_inches='tight')
    plt.close(fig)

###################################################################################

def run_RQ2(df_transfers, snapshots):
    print("Answering RQ2...")

    plot_overview_activity(df_transfers)
    df_all, df_senders, df_receivers = get_top10_addresses(df_transfers)
    list_addresses = list(df_all.index) + list(df_senders.index) + list(df_receivers.index)
    plot_activity_over_time(df_transfers, list_addresses)
    plot_frequency_transfers(df_transfers)

    df_nodes = build_node_evolution_table(list_addresses, snapshots)

    metrics = [
        ("in_degree", "in-degree"),
        ("out_degree", "out-degree"),
        ("clustering", "clustering"),
        ("degree_centrality", "degree centrality"),
        ("authority_score", "authority score"),
        ("hub_score", "hub score"),
    ]

    groups_config = [
        {
            "nodes": list_addresses[0:10],
            "title": "Top-10 addresses",
            "prefix": "A",
            "yscale_config": {
                0: {"scale": "symlog", "linthresh": 1, "linscale": 0.4},
                1: {"scale": "symlog", "linthresh": 1, "linscale": 0.4},
                2: {"scale": "symlog", "linthresh": 0.01, "linscale": 0.4},
                3: {"scale": "symlog", "linthresh": 0.001, "linscale": 0.4},
                4: {"scale": "symlog", "linthresh": 0.01, "linscale": 0.4},
                5: {"scale": "symlog", "linthresh": 0.01, "linscale": 0.4},
                },
            "highlight_nodes": None,
            "file_name": "local_activity_top_all"
        },
        {
            "nodes": list_addresses[10:20],
            "title": "Top-10 sender-dominant addresses",
            "prefix": "S",
            "yscale_config": {
                0: {"scale": "linear"},
                1: {"scale": "symlog", "linthresh": 1, "linscale": 0.4},
                2: {"scale": "symlog", "linthresh": 0.0001, "linscale": 0.4},
                3: {"scale": "symlog", "linthresh": 0.001, "linscale": 0.4},
                4: {"scale": "linear"},
                5: {"scale": "symlog", "linthresh": 0.00001, "linscale": 0.4},
                },
            "highlight_nodes": ['0x301d9bc22d66f7bc49329a9d9eb16d3ecc4a12b4',
                '0x5afe0000727040681d781fb23e8c82b468076042',
                '0x42e1bf5f3d4b17cccf31e30febdf417ff3726b88',
                '0x7cfdeb2def0fa6810eda0ac17c2a4e2f9e12db4f',
                '0x5bb06d84c7d2b9094edf129ad1bbfbca7d4edff5',
                '0x455bf23ea7575a537b6374953fa71b5f3653272c',
                '0xc5fd3ca1042aff36a9fa4b66c7b789846562ebb6'],
            "file_name": "local_activity_top_senders"
        },
        {
            "nodes": list_addresses[20:30],
            "title": "Top-10 receiver-dominant addresses",
            "prefix": "R",
            "yscale_config": {
                0: {"scale": "symlog", "linthresh": 1, "linscale": 0.4},
                1: {"scale": "linear"},
                2: {"scale": "symlog", "linthresh": 0.0001, "linscale": 0.4},
                3: {"scale": "symlog", "linthresh": 0.001, "linscale": 0.4},
                4: {"scale": "symlog", "linthresh": 0.00001, "linscale": 0.4},
                5: {"scale": "linear"},
            },
            "highlight_nodes": None,
            "file_name": "local_activity_top_receivers"
        },
    ]

    for cfg in groups_config:
        plot_group_evolution(df_nodes, metrics, cfg)