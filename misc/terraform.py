import pandas as pd
import networkx as nx
import time
from datetime import timedelta
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import random
import os
import json
import string

# We previously use eligible_mask and modified_mask but this gets complicated with merging and splitting so now we add columns to laundering_df instead

def controller(config_path=None):
    if not config_path or not os.path.exists(config_path):
        config_path = input("Config file not found: Enter full path to config file: ").strip()
        while not os.path.exists(config_path):
            config_path = input("File not found. Please re-enter path to JSON config file: ").strip()



    with open(config_path) as f:
        config = json.load(f)
    
    # seed_value = int(time.time())
    # seed_value = 50 # Reproducable and consistent perturbations
    seed = config.get("seed", int(time.time()))
    random.seed(seed)
    print(f"Using seed: {seed}")

    print("\n--- Parameterized Actions ---")
    for i, action in enumerate(config.get("actions", []), 1):
        print(f"{i}. {action['type']} | Settings: {action.get('settings', {})}")
    print("--------------------------\n")



    inputfile = config.get("inputfile")
    if not inputfile or not os.path.exists(inputfile):
        inputfile = input("Input path not found in config file. Enter path to input CSV file: ").strip()
        while not os.path.exists(inputfile):
            inputfile = input("File not found. Please re-enter input file path: ").strip()

    outputfile = config.get("outputfile")
    if not outputfile:
        outputfile = input("Output path not found in config file. Enter desired path to save output CSV file: ").strip()

    laundering_df, laundering_adjacent_df, remaining_non_laundering_df = prepare_graph(inputfile, config.get("visualization", 0))
    
    laundering_df["is_modified"] = 0
    laundering_adjacent_df["is_modified"] = 0

    mod_filter = config.get("modification_filter", "any")

    first_action = True

    for action in config.get("actions", []):
        action_type = action.get("type")
        settings = action.get("settings", {})

        if not first_action:
            if mod_filter == "modified":
                laundering_df["eligible_laundering"] &= laundering_df["is_modified"] == 1
                eligible_rows = laundering_df[laundering_df["eligible_laundering"] == True]

            elif mod_filter == "unmodified":
                laundering_df["eligible_laundering"] &= laundering_df["is_modified"] == 0
        
        if action_type == "INTERMEDIARY":
            laundering_df = intermediary_nodes(laundering_df, laundering_adjacent_df, settings)
        elif action_type == "SPLIT":
            laundering_df, laundering_adjacent_df = split_nodes(laundering_df, laundering_adjacent_df, settings)
        elif action_type == "MERGE":
            laundering_df, laundering_adjacent_df = merge_nodes(laundering_df, laundering_adjacent_df, settings)
        else:
            print(f"Unknown action: {action_type}")
            exit()
        first_action = False

        # combined_df["is_modified"] = combined_df.apply(lambda row: 1 if row["is_modified"] != 0 else 0, axis=1)

        # laundering_df = combined_df[combined_df['Is Laundering'] == 1].copy()
        # non_laundering_df = combined_df[combined_df['Is Laundering'] != 1].copy()




    num_laundering_edges = len(laundering_df)
    print(f"Total Laundering Edges: {num_laundering_edges}")
    hardcoded_original = 1611
    laundering_percentage_diff = ((num_laundering_edges - hardcoded_original) / hardcoded_original) * 100
    laundering_percentage_diff = round(laundering_percentage_diff, 2)

    print(f"Total Laundering Edges: {num_laundering_edges}")
    print(f"Percentage difference: {laundering_percentage_diff}%")


    num_modified_laundering_edges = laundering_df['is_modified'].sum()

    if num_laundering_edges > 0:
        percent_edges_modified = (num_modified_laundering_edges / num_laundering_edges) * 100
        percent_edges_modified = round(percent_edges_modified, 2)
        print(f"Percentage of Total Laundering Edges that have been modified or newly created: {num_modified_laundering_edges} / {num_laundering_edges}  ({percent_edges_modified}%)") # For inject it makes sense that for 50% this will return 66%, since if you have A -> B and C -> D and A->B is selected you end up with A->X->B and C->d so 2 edges are modified for each 1 that isnt
    else:
        print("No laundering edges found to compute modification percentage.")

    if 'cluster_id' in laundering_df.columns:
        total_clusters = laundering_df['cluster_id'].nunique()
        modified_clusters = laundering_df[laundering_df['is_modified'] == 1]['cluster_id'].nunique()

        if total_clusters > 0:
            percent_modified_clusters = (modified_clusters / total_clusters) * 100
            percent_modified_clusters = round(percent_modified_clusters, 2)

            print(f"Clusters with at least one modified edge: {modified_clusters} / {total_clusters}  ({percent_modified_clusters}%)")
        else:
            print("No clusters found in laundering data.")
    else:
        print("cluster_id column not found in laundering_df — cannot compute cluster-level modification stats.")

    print("Preparing final DF and saving to CSV...")
    final_df = pd.concat([laundering_df, laundering_adjacent_df, remaining_non_laundering_df], ignore_index=True)
    final_df = final_df.rename(columns={'Account.1': 'Account'})
    columns_to_drop = ['cluster_id', 'eligible_laundering', 'is_modified']
    existing_columns = [col for col in columns_to_drop if col in final_df.columns]
    final_df = final_df.drop(existing_columns, axis=1)

    final_df.to_csv(outputfile, index=False)


    prepare_graph(outputfile, config.get("visualization", []))
    
    return (percent_edges_modified, percent_modified_clusters, laundering_percentage_diff)
    

def cluster_laundering_graph(laundering_df): # This can be changed / made more efficient
    print("Clustering Graph...")
    G = nx.MultiDiGraph()
    for _, row in laundering_df.iterrows():
        G.add_edge(str(row['Account']), str(row['Account.1']), **row.to_dict())

    # Find clusters
    clusters = list(nx.weakly_connected_components(G))
    clusters = sorted(clusters, key=len, reverse=True)

    node_to_cluster = {}
    valid_cluster_ids = set()
    for idx, cluster in enumerate(clusters):
        if len(cluster) > 1: # Maye 2 XXX
            valid_cluster_ids.add(idx)
            for node in cluster:
                node_to_cluster[node] = idx

    def get_cluster_id(row):
        src = str(row['Account'])
        dst = str(row['Account.1'])
        # Only if both in the same cluster
        if node_to_cluster.get(src) == node_to_cluster.get(dst):
            return node_to_cluster.get(src)
        return None

    laundering_df['cluster_id'] = laundering_df.apply(get_cluster_id, axis=1)
    laundering_df['eligible_laundering'] = laundering_df['cluster_id'].notna()

    return laundering_df, valid_cluster_ids, clusters, G

def prepare_graph(inputfile, visualize=0): 
    print("\n--- Graph Stats ---")
    print(inputfile)
    print("--------------------------")
    print("Preparing Graph...")

    start_time = time.time()
    df = pd.read_csv(inputfile, delimiter=',', dtype={
        'Timestamp': str,
        'From Bank': str,
        'Account': str,
        'To Bank': str,
        'Amount Received': str,
        'Receiving Currency': str,
        'Amount Paid': str,
        'Payment Currency': str,
        'Payment Format': str
    })
    df.columns = df.columns.str.strip()

    laundering_df = df[df['Is Laundering'] == 1].copy()

    laundering_df, valid_cluster_ids, clusters, G = cluster_laundering_graph(laundering_df)

    laundering_accounts = set(laundering_df['Account']).union(set(laundering_df['Account.1']))

    laundering_adjacent_df = df[
        (df['Is Laundering'] != 1) &
        (df['Account'].isin(laundering_accounts) | df['Account.1'].isin(laundering_accounts))
    ].copy()

    # CLUSTER ADJACENTS, this seems like it should have been done earlier lol
    node_to_cluster = {}
    for _, row in laundering_df.iterrows():
        node_to_cluster[str(row['Account'])] = row['cluster_id']
        node_to_cluster[str(row['Account.1'])] = row['cluster_id']

    def assign_cluster(row):
        src = str(row['Account'])
        dst = str(row['Account.1'])
        if src in node_to_cluster:
            return node_to_cluster[src]
        if dst in node_to_cluster:
            return node_to_cluster[dst]
        return None

    laundering_adjacent_df['cluster_id'] = laundering_adjacent_df.apply(assign_cluster, axis=1)



    remaining_non_laundering_df = df[
        (~df.index.isin(laundering_df.index)) &
        (~df.index.isin(laundering_adjacent_df.index))
    ].copy()

    total_nodes = len(set(df['Account']).union(set(df['Account.1'])))
    laundering_nodes = len(set(laundering_df['Account']).union(set(laundering_df['Account.1'])))
    total_edges = len(df)
    laundering_edges = len(laundering_df)

    print(f"Total bank accounts (nodes) in dataset: {total_nodes}")
    print(f"Total bank accounts involved in laundering edges: {laundering_nodes}")
    print(f"Total transactions (edges) in dataset: {total_edges}")
    print(f"Total laundering transactions (edges): {laundering_edges}")
    print("")
    print(f"Clusters found: {len(clusters)}")
    # print(f"Clusters with > 2 nodes: {len(valid_cluster_ids)}")
    # print(f"Edges in clusters with > 2 nodes: {laundering_df['eligible_laundering'].sum()}")
    print("")

    cluster_size_counts = Counter(len(c) for c in clusters)
    print("Graph Summary:")
    for size, count in sorted(cluster_size_counts.items(), reverse=True):
        print(f"{count} cluster(s) with {size} nodes")
    print("--------------------------\n")
    print(f"Built and analyzed graph in: {time.time() - start_time:.2f} seconds\n")

    if visualize in (1, 2):
        if visualize == 2: # Visualize both laundering_df and laundering_adjacent_df
            # print(laundering_adjacent_df)
            for _, row in laundering_adjacent_df.iterrows():
                G.add_edge(row['Account'], row['Account.1'])

        print("Visualizing Graph")
        plt.figure(figsize=(16, 12))
        base_offset = 5
        cluster_positions = {}
        x_offset = 0

        # Position clustered nodes
        for i, cluster_nodes in enumerate(clusters):
            subgraph = G.subgraph(cluster_nodes).copy()
            pos = nx.spring_layout(subgraph, seed=42)
            offset_pos = {node: (x + x_offset, y) for node, (x, y) in pos.items()}
            cluster_positions.update(offset_pos)
            x_offset += base_offset

        # Position non-clustered nodes
        all_involved_nodes = set(G.nodes)
        for node in all_involved_nodes:
            if node not in cluster_positions:
                neighbors = [n for n in G.neighbors(node) if n in cluster_positions]
                if neighbors:
                    ref_node = neighbors[0]
                    ref_x, ref_y = cluster_positions[ref_node]
                    offset_scale = 2

                    offset_x = random.uniform(-offset_scale, offset_scale)
                    offset_y = random.uniform(-offset_scale, offset_scale)
                    cluster_positions[node] = (ref_x + offset_x, ref_y + offset_y)
                else:
                    cluster_positions[node] = (x_offset, 0)
                    x_offset += 1

        nodes_to_draw = list(G.nodes)
        node_colors = ['red' if node in laundering_accounts else 'gray' for node in nodes_to_draw]

        nx.draw_networkx_nodes(G, pos=cluster_positions, nodelist=nodes_to_draw,
                               node_size=100, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_labels(G, pos=cluster_positions, font_size=8)

        # Draw laundering edges in red
        laundering_edges = list(zip(laundering_df['Account'], laundering_df['Account.1']))
        nx.draw_networkx_edges(
            G,
            pos=cluster_positions,
            edgelist=laundering_edges,
            connectionstyle="arc3,rad=0.1",
            arrows=True,
            arrowstyle='-|>',
            edge_color='red',
            alpha=0.7,
            width=1.5,
            arrowsize=10
        )

        # Draw adjacent edges in gray if visualize option 2 used
        if visualize == 2:
            adjacent_edges = list(zip(laundering_adjacent_df['Account'], laundering_adjacent_df['Account.1']))
            nx.draw_networkx_edges(
                G,
                pos=cluster_positions,
                edgelist=adjacent_edges,
                connectionstyle="arc3,rad=0.2",
                arrows=True,
                arrowstyle='-|>',
                edge_color='gray',
                alpha=0.5,
                width=1,
                arrowsize=10
            )

        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return laundering_df, laundering_adjacent_df, remaining_non_laundering_df

def gen_node_name():    
    # To generate the same names if given the same seed
    node_name = f"10{''.join(random.choices('0123456789ABCDEF', k=7))}"
    
    return node_name


def intermediary_nodes(laundering_df, laundering_adjacent_df, settings):
    percent = settings.get("percent", 1.0)
    intermediary_depth = settings.get("intermediary_depth", 1)
    perturb_by_edges = settings.get("perturb_by_edges", True)

    laundering_df['Timestamp'] = pd.to_datetime(laundering_df['Timestamp'])

    # Temporal tracking, hopefuly doesnt take too long
    dfs_to_concat = [df for df in [laundering_df, laundering_adjacent_df] if not df.empty and not df.isna().all().all()]
    combined_df = pd.concat(dfs_to_concat, ignore_index=True)
    combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'])
    combined_df_sorted = combined_df.sort_values('Timestamp')
    all_outgoing_by_account = combined_df_sorted.groupby('Account')

    print(f"Running INTERMEDIARY with {percent*100:.1f}% of eligible {'edges' if perturb_by_edges else 'connections'}")
    print(f"Mode: {'Perturb by edges' if perturb_by_edges else 'Persist intermediaries by connections'}")
    print(f"Number of intermediaries per selected edge: {intermediary_depth}")

    start_time = time.time()
    eligible_rows = laundering_df[laundering_df["eligible_laundering"] == True]

    new_rows = []
    intermediary_map = {}

    if perturb_by_edges:
        eligible_indices = eligible_rows.index
        num_to_perturb = int(len(eligible_indices) * percent)
        if num_to_perturb == 0 and len(eligible_indices) > 0:
            atleast1 = random.random()
            if atleast1 <= percent:
                num_to_perturb = 1
        perturbed_indices = set(random.sample(list(eligible_indices), num_to_perturb))
    else:
        all_connections = laundering_df.apply(lambda r: (str(r['Account']).strip(), str(r['Account.1']).strip()), axis=1)
        eligible_connections = all_connections[laundering_df["eligible_laundering"] == True]
        unique_connections = list(set(eligible_connections))
        num_to_perturb = int(len(unique_connections) * percent)
        if num_to_perturb == 0 and len(unique_connections) > 0:
            atleast1 = random.random()
            if atleast1 <= percent:
                num_to_perturb = 1
        perturbed_keys = set(random.sample(unique_connections, num_to_perturb))

    for idx, row in laundering_df.iterrows():
        src = str(row['Account']).strip()
        dst = str(row['Account.1']).strip()
        orig_time = row['Timestamp']
        key = (src, dst)

        if perturb_by_edges:
            perturb = idx in perturbed_indices
        else:
            perturb = key in perturbed_keys

        if perturb:
            if not perturb_by_edges and key in intermediary_map:
                intermediaries = intermediary_map[key]
            else:
                intermediaries = [gen_node_name() for _ in range(intermediary_depth)]
                if not perturb_by_edges:
                    intermediary_map[key] = intermediaries
        else:
            intermediaries = []

        if intermediaries:
            try:
                dst_outputs = all_outgoing_by_account.get_group(dst)
                future_outputs = dst_outputs[dst_outputs['Timestamp'] > orig_time]
                dst_next_time = future_outputs['Timestamp'].iloc[0] if not future_outputs.empty else orig_time + timedelta(hours=1)
            except KeyError:
                dst_next_time = orig_time + timedelta(hours=1)

            hops = [src] + intermediaries + [dst]
            times = [orig_time]
            if len(hops) > 2:
                increments = (dst_next_time - orig_time) / (len(hops) - 1)
                for i in range(1, len(hops) - 1):
                    times.append(orig_time + i * increments)


            for i in range(len(hops) - 1):
                new_row = row.copy()
                new_row['Account'] = hops[i]
                new_row['Account.1'] = hops[i + 1]
                new_row['From Bank'] = row['From Bank']
                new_row['To Bank'] = row['To Bank']
                new_row['Timestamp'] = times[i]
                new_row['is_modified'] = 1
                new_row = new_row.drop(labels='cluster_id', errors='ignore')

                if i != len(hops) - 2:
                    new_row['Amount Paid'] = row['Amount Paid']
                    new_row['Amount Received'] = row['Amount Paid']
                else:
                    new_row['Amount Paid'] = row['Amount Paid']
                    new_row['Amount Received'] = row['Amount Received']

                new_rows.append(new_row)

        else:
            original_row = row.copy()
            original_row['is_modified'] = 0
            original_row = original_row.drop(labels='cluster_id', errors='ignore')
            new_rows.append(original_row)

    laundering_df = pd.DataFrame(new_rows)
    laundering_df['Timestamp'] = pd.to_datetime(laundering_df['Timestamp'])
    laundering_df['Timestamp'] = laundering_df['Timestamp'].dt.strftime('%Y/%m/%d %H:%M')

    if (perturb_by_edges):
        print(f"Perturbed {num_to_perturb} edges in eligible clusters.")
    else:
        print(f"Perturbed {num_to_perturb} connections in eligible clusters.")
        
    print(f"Intermediary Node Injection completed in: {time.time() - start_time:.2f} seconds")

    laundering_df, valid_cluster_ids, clusters, G = cluster_laundering_graph(laundering_df)

    return laundering_df 

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        x = str(x)
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x

    def get_sets(self):
        result = {}
        for node in self.parent:
            root = self.find(node)
            result.setdefault(root, set()).add(node)
        return result

def merge_nodes(laundering_df, laundering_adjacent_df, settings):
    percent = settings.get("percent", 1.0)
    perturb_by_edges = settings.get("perturb_by_edges", True)

    uf = UnionFind()
    merged_nodes = set()

    start_time = time.time()


    eligible_rows = laundering_df[laundering_df["eligible_laundering"] == True]

    num_to_merge = 0
    total_merge = 0

    if perturb_by_edges:
        for cluster_id, cluster_df in eligible_rows.groupby('cluster_id'):
            cluster_size = len(set(cluster_df['Account'].astype(str)).union(set(cluster_df['Account.1'].astype(str))))
            if cluster_size < 2:
                continue

            src_to_dsts = defaultdict(set)
            dst_to_srcs = defaultdict(set)
            for _, row in cluster_df.iterrows():
                src = str(row['Account']).strip()
                dst = str(row['Account.1']).strip()
                src_to_dsts[src].add(dst)
                dst_to_srcs[dst].add(src)

            connections = list((src, dst) for src, dsts in src_to_dsts.items() for dst in dsts)
            
            max_merges = max((cluster_size - 2), 0)

            num_to_merge = int(len(connections) * percent)
            num_to_merge = min(num_to_merge, max_merges)

            if num_to_merge == 0:
                continue

            total_merge+=num_to_merge
            connections_to_merge = random.sample(connections, num_to_merge)

            for src, dst in connections_to_merge:
                # Mark edges between src and dst to be removed
                laundering_df.loc[
                    ((laundering_df['Account'] == src) & (laundering_df['Account.1'] == dst)) |
                    ((laundering_df['Account'] == dst) & (laundering_df['Account.1'] == src)),
                    ['Account', 'Account.1']
                ] = "TO_DROP", "TO_DROP"

                uf.union(src, dst)
                merged_nodes.add(dst)

                src_to_dsts[src].discard(dst)
                dst_to_srcs[dst].discard(src)

                # Redirect incoming edges
                for src2 in list(dst_to_srcs.get(dst, [])):
                    src_to_dsts[src2].discard(dst)
                    src_to_dsts[src2].add(src)
                    dst_to_srcs[src].add(src2)
                    dst_to_srcs[dst].discard(src2)
                dst_to_srcs.pop(dst, None)

                # Redirect outgoing edges 
                for dst2 in list(src_to_dsts.get(dst, [])):
                    dst_to_srcs[dst2].discard(dst)
                    dst_to_srcs[dst2].add(src)
                    src_to_dsts[src].add(dst2)
                src_to_dsts.pop(dst, None)

                # if dst in src_to_dsts[src]:
                #     src_to_dsts[src].remove(dst)
                # if src in dst_to_srcs[dst]:
                #     dst_to_srcs[dst].remove(src)

                # # Redirect incoming edges
                # for src2 in list(dst_to_srcs.get(dst, [])):
                #     if dst in src_to_dsts[src2]:
                #         src_to_dsts[src2].remove(dst)
                #     src_to_dsts[src2].add(src)
                #     dst_to_srcs[src].add(src2)
                #     dst_to_srcs[dst].remove(src2)
                # dst_to_srcs.pop(dst, None)

                # # Redirect outgoing edges 
                # for dst2 in list(src_to_dsts.get(dst, [])):
                #     dst_to_srcs[dst2].remove(dst)
                #     dst_to_srcs[dst2].add(src)
                #     src_to_dsts[src].add(dst2)
                # src_to_dsts.pop(dst, None)

    else: # Here we maintain all edges. if we combine 2 radial nodes we cannot combine the edge into one and use the earlier time because the node may not have the full amount of both by then, and we also cannot combine them and use the later time because the next node may have an earlier outgoing edge
        for cluster_id, cluster_df in eligible_rows.groupby('cluster_id'):
            if len(cluster_df) < 2:
                continue

            outgoing = defaultdict(list)
            for _, row in cluster_df.iterrows():
                src = str(row['Account']).strip()
                dst = str(row['Account.1']).strip()
                outgoing[src].append(dst)

            for src, destinations in outgoing.items():
                if len(destinations) > 1:
                    print(f"Node '{src}' has outgoing edges to: {destinations}")

                    destinations = list(set(destinations)) # We want this to be unique so when we merge a node we do so for all edges in that node 

                    if len(destinations) < 2:
                        continue

                    num_to_merge = int(len(destinations) * percent)
                    if num_to_merge == 0:
                        continue

                    total_merge+=num_to_merge

                    base = uf.find(destinations[0])
                    merge_these = random.sample(destinations[1:], min(num_to_merge, len(destinations) - 1))
                    for dst in merge_these:
                        uf.union(base, dst)
                        merged_nodes.update([dst])

    # # Save for comparison
    # original_from = eligible_rows['Account'].astype(str).copy()
    # original_to = eligible_rows['Account.1'].astype(str).copy()

    # Remap using union-find results
    # laundering_df['Account'] = laundering_df['Account'].astype(str).map(uf.find)
    # laundering_df['Account.1'] = laundering_df['Account.1'].astype(str).map(uf.find)
    # laundering_df['From Bank'] = laundering_df['From Bank'].astype(str).map(uf.find)
    # laundering_df['To Bank'] = laundering_df['To Bank'].astype(str).map(uf.find)
    # laundering_df['is_modified'] = laundering_df['is_modified'].astype(str).map(uf.find) # Get original one first

    # original_from_aligned = original_from.reindex(laundering_df.index)
    # original_to_aligned = original_to.reindex(laundering_df.index)

    # laundering_df['is_modified'] = laundering_df['is_modified'].astype(int) | (
    #     (laundering_df['Account'] != original_from_aligned) | 
    #     (laundering_df['Account.1'] != original_to_aligned)
    # ).astype(int)

    original_account = laundering_df['Account'].astype(str)
    original_account1 = laundering_df['Account.1'].astype(str)

    laundering_df['Account'] = original_account.map(uf.find)
    laundering_df['Account.1'] = original_account1.map(uf.find)

    # Update modified
    changed_mask = (laundering_df['Account'] != original_account) | (laundering_df['Account.1'] != original_account1)
    laundering_df['is_modified'] = laundering_df['is_modified'].astype(int) | changed_mask.astype(int)

    laundering_df = laundering_df[(laundering_df['Account'] != "TO_DROP") & (laundering_df['Account.1'] != "TO_DROP")].copy()

    laundering_adjacent_df['Account'] = laundering_adjacent_df['Account'].astype(str).apply(
        lambda x: uf.find(x) if x in merged_nodes else x
    )
    laundering_adjacent_df['Account.1'] = laundering_adjacent_df['Account.1'].astype(str).apply(
        lambda x: uf.find(x) if x in merged_nodes else x
    )

    laundering_df, valid_cluster_ids, clusters, G = cluster_laundering_graph(laundering_df)

    print(f"Merged {total_merge} nodes in eligible clusters.")
    print(f"Node Merging completed in: {time.time() - start_time:.2f} seconds")

    return laundering_df, laundering_adjacent_df



def split_nodes(laundering_df, laundering_adjacent_df, settings):
    percent = settings.get("percent", 1.0)
    perturb_by_edges = settings.get("perturb_by_edges", False)
    split_depth = settings.get("split_depth", 1)
    start_time = time.time()


    laundering_df = laundering_df.copy()
    eligible_rows = laundering_df[laundering_df["eligible_laundering"] == True]

    if perturb_by_edges: # Sucks please fix
        new_rows = []
        split_map = {} 
        edge_transactions = {}

        eligible_edges = set((str(row['Account']).strip(), str(row['Account.1']).strip()) for _, row in eligible_rows.iterrows())
        num_to_perturb = int(len(eligible_edges) * percent)
        selected_edges = set(random.sample(list(eligible_edges), num_to_perturb))
        print(f"Splitting {len(selected_edges)} edges")

        for idx, row in laundering_df.iterrows():
            row = row.copy()
            src = str(row['Account']).strip()
            dst = str(row['Account.1']).strip()
            amount_received = float(row['Amount Received'])
            if (src, dst) in selected_edges:
                if (src, dst) not in split_map:
                    split_map[(src, dst)] = [gen_node_name()]

                split_amount = amount_received / (split_depth+1)
                for i in range(split_depth):
                    new_dst = gen_node_name()
                    split_map[(src, dst)].append(new_dst)

                    if src not in edge_transactions:
                        edge_transactions[src] = []
                    edge_transactions[src].append({'Timestamp': row['Timestamp'], 'dst': new_dst, 'Amount': split_amount})

                    if i == 0: # Original connection
                        original_row = row.copy()
                        original_row['is_modified'] = 0
                        original_row['Amount Received'] =  f"{split_amount:.2f}"
                        original_row['Amount Paid'] =  f"{split_amount:.2f}"
                        original_row = original_row.drop(labels='cluster_id', errors='ignore')
                        new_rows.append(original_row)
                    else: # Subsequent splits
                        split_row = row.copy()
                        split_row['Account.1'] = new_dst
                        split_row['is_modified'] = 1
                        split_row['Amount Received'] = f"{split_amount:.2f}"
                        split_row['Amount Paid'] = f"{split_amount:.2f}"
                        split_row = split_row.drop(labels='cluster_id', errors='ignore')
                        new_rows.append(split_row)

            else:
                row['is_modified'] = 0
                new_rows.append(row)

        laundering_df = pd.DataFrame(new_rows)

    else:
        original_laundering_df = laundering_df.copy()
        all_updated_rows = []
        all_new_rows = []
        clusternum = 0
        total_split = 0
        for cluster_id, cluster_df in eligible_rows.groupby('cluster_id'):
            print(f"Cluster: {clusternum}")
            clusternum+=1
            node_set = set(cluster_df['Account'].astype(str)) | set(cluster_df['Account.1'].astype(str))
            num_nodes = len(node_set)

            num_to_split = int(num_nodes * percent)
            # if num_to_split == 0:
            #     num_to_split = 1 # This is not an appropriate solution
            if num_to_split == 0 and num_nodes > 0:
                atleast1 = random.random()
                if atleast1 <= percent:
                    num_to_split = 1 # Hopefully we now no longer lose those edges entirely

            total_split += num_to_split

            if num_to_split == 0:
                continue  

            split_map = {}
            selected_nodes = random.sample(list(node_set), min(num_to_split, len(node_set)))
            print(f"Splitting {selected_nodes}")

            for node in selected_nodes:
                new_nodes = [gen_node_name() for _ in range(split_depth)]
                split_map[node] = new_nodes

            # Per Cluster
            new_rows = []
            updated_original_rows = []

            for original_node, new_nodes in split_map.items():
                src_edges = cluster_df[cluster_df['Account'] == original_node]
                dst_edges = cluster_df[cluster_df['Account.1'] == original_node]

                handled_nodes = set(split_map.keys())
                for _, row in cluster_df.iterrows():
                    src = str(row['Account'])
                    dst = str(row['Account.1'])

                    if src not in handled_nodes and dst not in handled_nodes:
                        updated_original_rows.append(row)

                # Outgoing edges
                for _, row in src_edges.iterrows():
                    target = str(row['Account.1'])
                    target_clones = split_map.get(target, [])
                    source_clones = split_map.get(original_node, [])

                    total_edges = 1 + len(target_clones) + len(source_clones) + len(source_clones) * len(target_clones)
                    if total_edges == 0:
                        continue

                    split_amount_paid = float(row['Amount Paid']) / total_edges
                    split_amount_received = float(row['Amount Received']) / total_edges
                    formatted_paid = f"{split_amount_paid:.2f}"
                    formatted_received = f"{split_amount_received:.2f}"

                    # A to B (original)
                    updated_row = row.copy()
                    updated_row['Amount Paid'] = formatted_paid
                    updated_row['Amount Received'] = formatted_received
                    updated_row["is_modified"] = 1
                    updated_original_rows.append(updated_row)

                    # A to Yi (target clones)
                    for tgt in target_clones:
                        new_row = row.copy()
                        new_row['Account.1'] = tgt
                        new_row['Amount Paid'] = formatted_paid
                        new_row['Amount Received'] = formatted_received
                        new_row["is_modified"] = 1
                        new_rows.append(new_row)

                    # Xi to B (source clones)
                    for src in source_clones:
                        new_row = row.copy()
                        new_row['Account'] = src
                        new_row['Amount Paid'] = formatted_paid
                        new_row['Amount Received'] = formatted_received
                        new_row["is_modified"] = 1
                        new_rows.append(new_row)

                    # Xi to Yi (clones to clones)
                    for src in source_clones:
                        for tgt in target_clones:
                            new_row = row.copy()
                            new_row['Account'] = src
                            new_row['Account.1'] = tgt
                            new_row['Amount Paid'] = formatted_paid
                            new_row['Amount Received'] = formatted_received
                            new_row["is_modified"] = 1
                            new_rows.append(new_row)

                # Incoming edges
                for _, row in dst_edges.iterrows():
                    source = str(row['Account'])
                    source_clones = split_map.get(source, [])
                    target_clones = split_map.get(original_node, [])

                    total_edges = 1 + len(source_clones) + len(target_clones) + len(source_clones) * len(target_clones)
                    if total_edges == 0:
                        continue

                    split_amount_paid = float(row['Amount Paid']) / total_edges
                    split_amount_received = float(row['Amount Received']) / total_edges
                    formatted_paid = f"{split_amount_paid:.2f}"
                    formatted_received = f"{split_amount_received:.2f}"

                    # A to B (original)
                    updated_row = row.copy()
                    updated_row['Amount Paid'] = formatted_paid
                    updated_row['Amount Received'] = formatted_received
                    updated_row["is_modified"] = 1
                    updated_original_rows.append(updated_row)

                    # XSi to B (source clones)
                    for src in source_clones:
                        new_row = row.copy()
                        new_row['Account'] = src
                        new_row['Amount Paid'] = formatted_paid
                        new_row['Amount Received'] = formatted_received
                        new_row["is_modified"] = 1
                        new_rows.append(new_row)

                    # A to XTi (target clones)
                    for tgt in target_clones:
                        new_row = row.copy()
                        new_row['Account.1'] = tgt
                        new_row['Amount Paid'] = formatted_paid
                        new_row['Amount Received'] = formatted_received
                        new_row["is_modified"] = 1
                        new_rows.append(new_row)

                    # XSi to XTi (clones to clones)
                    for src in source_clones:
                        for tgt in target_clones:
                            new_row = row.copy()
                            new_row['Account'] = src
                            new_row['Account.1'] = tgt
                            new_row['Amount Paid'] = formatted_paid
                            new_row['Amount Received'] = formatted_received
                            new_row["is_modified"] = 1
                            new_rows.append(new_row)

                all_updated_rows.extend(updated_original_rows)
                all_new_rows.extend(new_rows)

        # Combine and drop duplicates (we get duplicates from src and tgt)
        combined_df = pd.DataFrame(all_updated_rows + all_new_rows)
        combined_df.drop_duplicates(subset=["Timestamp", "Account", "Account.1", "Amount Received"], inplace=True)

        if combined_df.empty:
            print("Nothing modified, returning original")
            return original_laundering_df, laundering_adjacent_df

        modified_keys = set(tuple(row) for row in combined_df[["Timestamp", "Account", "Account.1", "Receiving Currency", "Payment Format"]].values)
        # This approach is actually somewhat problematic becuase there are rare instances in that dataset where a non-duplicate transaction has all these columns in common, but its infrequent enough in laundering transactions that its acceptable 

        def row_key(row):
            return (row["Timestamp"], row["Account"], row["Account.1"], row["Receiving Currency"], row["Payment Format"])

        # Remove only original rows that were replaced
        unmodified_rows = original_laundering_df[~original_laundering_df.apply(row_key, axis=1).isin(modified_keys)]

        laundering_df = pd.concat([unmodified_rows, combined_df], ignore_index=True)


    laundering_df, valid_cluster_ids, clusters, G = cluster_laundering_graph(laundering_df)
    
    print(f"Split {total_split} nodes in eligible clusters.")

    print(f"Node Splitting completed in: {time.time() - start_time:.2f} seconds")


    return laundering_df, laundering_adjacent_df


# configfile = input("Enter config file name: ")
# controller(configfile)