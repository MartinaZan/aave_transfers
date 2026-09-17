import io
import json
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import seaborn as sns
import random
from statsmodels.stats.multitest import multipletests

###################################################################################

def import_data(file_name):
    """Import data from a single JSON file and return a DataFrame."""
    with open(file_name) as f:
        d = json.load(f)

    df = pd.DataFrame(d)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def import_data_range(start_id, end_id, prefix="data/aave_transfers"):
    """Import data from multiple JSON files (start_id to end_id) and append into one DataFrame."""
    dfs = []
    for id in range(start_id, end_id + 1):
        file_name = f"{prefix}_{id}.json"
        df = import_data(file_name)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

###################################################################################

# Events data
def get_df_events():
    data = """date	category	description
    14/10/2020	Technological	Staking Deposit Contract Ethereum fork
    21/10/2020	Market / Regulatory	PayPal Accepts Cryptocurrencies
    26/10/2020	Security Incident	Harvest Finance Attack
    14/11/2020	Security Incident	Value DeFi Exploit
    01/12/2020	Technological	Beacon Chain Genesis Ethereum fork
    03/12/2020	Aave News	Aave V2 Launch
    04/02/2021	Security Incident	Yearn Finance Attack
    08/02/2021	Market / Regulatory	Tesla Buys Bitcoin
    19/02/2021	Market / Regulatory	Bitcoin $1 Trillion Hit
    15/04/2021	Technological	Berlin Ethereum fork
    18/05/2021	Aave News	Aave All-Time High
    05/08/2021	Technological	London Ethereum fork
    10/08/2021	Security Incident	Poly Network Hack
    27/10/2021	Security Incident	Cream Finance Exploit
    27/10/2021	Technological	Altair Ethereum fork
    15/11/2021	Market / Regulatory	US Infrastructure Bill and Crypto Tax
    09/12/2021	Technological	Arrow Glacier Ethereum fork
    02/02/2022	Security Incident	Wormhole Bridge Hack
    24/02/2022	Global Conflict	Russo-Ukrainian War
    16/03/2022	Aave News	Aave V3 Launch
    29/03/2022	Security Incident	Ronin Bridge Hack
    18/04/2022	Security Incident	Beanstalk Governance Attack
    08/05/2022	Security Incident	Terra Classic Collapse
    30/06/2022	Technological	Gray Glacier Ethereum fork
    01/08/2022	Security Incident	Nomad Bridge Exploit
    06/09/2022	Technological	Bellatrix Ethereum fork
    15/09/2022	Technological	The Merge (Paris) Ethereum fork
    06/10/2022	Security Incident	BSC Token Hub Exploit
    11/11/2022	Market / Regulatory	FTX Bankruptcy Filing
    19/01/2023	Market / Regulatory	Genesis Files for Bankruptcy
    13/03/2023	Security Incident	Euler Finance Attack
    12/04/2023	Technological	Shapella Ethereum fork
    23/06/2023	Market / Regulatory	Bitcoin $31k Peak Price
    07/10/2023	Global Conflict	Gaza War
    23/11/2023	Security Incident	KyberSwap Elastic Exploit
    13/03/2024	Technological	Dencun Ethereum fork
    10/08/2024	Aave News	Large Aave Liquidations
    12/12/2024	Aave News	Aave Crypto Whales
    21/02/2025	Security Incident	Bybit Attack
    07/05/2025	Technological	Pectra Ethereum fork
    22/05/2025	Security Incident	Cetus Protocol Exploit
    27/08/2025	Aave News	Aave Horizon Release
    15/10/2025	Aave News	Aave TVL Peak
    03/11/2025	Security Incident	Balancer Exploit
    03/12/2025	Technological	Fusaka Ethereum fork
    22/12/2025	Aave News	Significant Aave Whale Sell-Off
    25/02/2026	Aave News	Aave $1 Trillion Hit
    28/02/2026	Global Conflict	Iran War
    30/03/2026	Aave News	Aave V4 Launch
    18/04/2026	Security Incident	Kelp DAO Exploit
    """

    df_events = pd.read_csv(io.StringIO(data), sep="\t")
    df_events["date"] = pd.to_datetime(df_events["date"])
    return df_events

###################################################################################

def short_address(addr):
    """Shorten an address for display purposes."""
    return addr[:6] + "..." + addr[-4:]

def cut_dates(metrics_dict, start_date_str, end_date_str):
    """Cut the metrics dictionary to only include data between start_date and end_date."""
    start_date = datetime.strptime(start_date_str, "%d/%m/%Y")
    end_date = datetime.strptime(end_date_str, "%d/%m/%Y")

    idx = [i for i, ts in enumerate(metrics_dict["timestamp"])
        if start_date <= datetime.fromisoformat(ts) <= end_date]

    return {k: [v[i] for i in idx] for k, v in metrics_dict.items()}

###################################################################################

def gini(values):
    """Compute the Gini coefficient."""
    v = np.array(sorted(values), dtype=float)
    n = len(v)
    return (2 * np.dot(np.arange(1, n + 1), v) - (n + 1) * v.sum()) / (n * v.sum())

def compute_metrics(df):
    """Compute various graph metrics for each day in the transfers DataFrame."""
    
    # Get daily groups
    df["date"] = df["timestamp"].dt.date
    df_groups = df.groupby("date")

    metrics = defaultdict(list)
    snapshots = []

    for day_idx, (date, day_df) in enumerate(sorted(df_groups), start=1):
        print(f"{date}...")
        G_day = nx.DiGraph()   
        G_day.graph["date"] = date 
        for _, row in day_df.iterrows():
            u, v = row["from"], row["to"]
            G_day.add_edge(u, v, value=float(row["value"]))
        snapshots.append(G_day)

        metrics['timestamp'].append(date)

        #######################
        ##  Size and degree  ##
        #######################
        metrics['n_nodes'].append(G_day.number_of_nodes())
        metrics['n_edges'].append(G_day.number_of_edges())

        metrics['node_degree_avg'].append(np.mean([d for _, d in G_day.degree()]))
        metrics['node_degree_std'].append(np.std([d for _, d in G_day.degree()]))

        metrics['max_in_degree'].append(max(d for _, d in G_day.in_degree()))
        metrics['max_out_degree'].append(max(d for _, d in G_day.out_degree()))

        degrees = sorted([d for _, d in G_day.degree()], reverse=True)
        top10 = degrees[:10]
        metrics['top10_degree_avg'].append(np.mean(top10))
        metrics['top10_degree_std'].append(np.std(top10))
        metrics['top10_degree_avg_ratio'].append(np.mean(top10) / (sum(degrees) / len(degrees)))

        metrics['CV_degree'].append(np.std([d for _, d in G_day.degree()]) / np.mean([d for _, d in G_day.degree()]) if np.mean([d for _, d in G_day.degree()]) != 0 else 0)
        metrics['gini_degree'].append(gini([d for _, d in G_day.degree()]))

        metrics['density'].append(nx.density(G_day))
            
        ##################################
        ##  Clustering and reciprocity  ##
        ##################################

        metrics['reciprocity'].append(nx.reciprocity(G_day))

        clustering = nx.clustering(G_day.to_undirected())
        metrics['clustering_avg'].append(np.mean(list(clustering.values())))
        metrics['clustering_std'].append(np.std(list(clustering.values())))

        metrics['transitivity'].append(nx.transitivity(G_day))

        ####################################
        ##  Community and core structure  ##
        ####################################
        wcc = list(nx.weakly_connected_components(G_day))
        scc = list(nx.strongly_connected_components(G_day))
        metrics['num_wcc'].append(len(wcc))
        metrics['num_scc'].append(len(scc))

        metrics['largest_wcc_rel_size'].append(max(len(c) for c in wcc) / G_day.number_of_nodes())
        metrics['largest_scc_rel_size'].append(max(len(c) for c in scc) / G_day.number_of_nodes())

        G_undir = G_day.to_undirected()
        G_undir = nx.convert_node_labels_to_integers(G_undir)
        communities = tuple(nx.community.greedy_modularity_communities(G_undir))
        metrics['modularity_score'].append(nx.community.modularity(G_undir, communities))

        metrics['degree_assortativity'].append(nx.degree_assortativity_coefficient(G_day))

        G_undir = G_day.to_undirected()
        G_undir.remove_edges_from(nx.selfloop_edges(G_undir))
        G_undir = nx.convert_node_labels_to_integers(G_undir)
        kcore = nx.core_number(G_undir)
        metrics['k_core_decomposition'].append(np.mean(list(kcore.values())))
        metrics['fraction_nodes_core'].append(sum(1 for v in kcore.values() if v == max(kcore.values())) / G_day.number_of_nodes())

        metrics['beta_1_norm'].append((G_day.number_of_edges() - G_day.number_of_nodes() + nx.number_weakly_connected_components(G_day)) / G_day.number_of_edges())

        closeness = nx.closeness_centrality(G_day)
        metrics['closeness_centrality_mean'].append(np.mean(list(closeness.values())))
        metrics['closeness_centrality_std'].append(np.std(list(closeness.values())))

    return metrics, snapshots