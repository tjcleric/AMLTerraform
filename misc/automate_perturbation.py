import os
from terraform import controller
from myformat import format_transactions
from combine_splits import combine_csv_files
import time
import shutil
import statistics


def update_percentage(preset_filename, multiplyer):
    if os.path.exists(preset_filename):
        with open(preset_filename, 'r') as f:
            lines = f.readlines()

        updated_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith('"percent":') or '"percent":' in stripped:
                indent = line[:len(line) - len(stripped)]
                new_value = round(0.2 * multiplyer, 5)
                # new_value = round((0.2 * multiplyer)-0.1, 5)
                updated_line = f'{indent}"percent": {new_value},\n'
                updated_lines.append(updated_line)
                print(f"Updated percent in {preset_filename} to {new_value}")
            else:
                updated_lines.append(line)

        with open(preset_filename, 'w') as f:
            f.writelines(updated_lines)
    else:
        print(f"File not found: {preset_filename}")

def update_seed(preset_filename, multiplyer):
    if os.path.exists(preset_filename):
        with open(preset_filename, 'r') as f:
            lines = f.readlines()

        updated_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith('"seed":') or '"seed":' in stripped:
                indent = line[:len(line) - len(stripped)]
                new_value = 50 * multiplyer
                updated_line = f'{indent}"seed": {new_value},\n'
                updated_lines.append(updated_line)
                print(f"Updated seed in {preset_filename} to {new_value}")
            else:
                updated_lines.append(line)

        with open(preset_filename, 'w') as f:
            f.writelines(updated_lines)
    else:
        print(f"File not found: {preset_filename}")


def mean_stddev(lst):
    return (statistics.mean(lst), statistics.stdev(lst) if len(lst) > 1 else 0.0)

def handle_everything():
    datadir = "data"
    with open("perturbation_metrics.txt", "w") as f:
        f.write("")  

    for i in range(1, 4):
        for j in range(1, 5):
            update_percentage(f"preset{i}.json", j)
            edges_list = []
            clusters_list = []
            laundering_list = []

            for k in range(1, 7):
                update_seed(f"preset{i}.json", k)
                preset_dir = os.path.join(datadir, f"preset{i}_{j*20}_{k}")
                os.makedirs(preset_dir, exist_ok=True)
                print(f"Created: {preset_dir}")

                configfile = f"preset{i}.json"
                percent_edges_modified, percent_modified_clusters, laundering_percentage_diff = controller(configfile)

                edges_list.append(percent_edges_modified)
                clusters_list.append(percent_modified_clusters)
                laundering_list.append(laundering_percentage_diff)

                time.sleep(1)

                combine_csv_files([f"{datadir}/unformatted_remaining_split.csv", f"{datadir}/augmented.csv"], f"{datadir}/unformatted.csv")

                time.sleep(1)

                format_transactions(f"{datadir}/unformatted.csv")

                unformatted_path = f"{datadir}/unformatted.csv"
                if os.path.exists(unformatted_path):
                    os.remove(unformatted_path)
                    print(f"Deleted {unformatted_path}")

                formatted_path = f"{datadir}/formatted_transactions.csv"
                if os.path.exists(formatted_path):
                    destination_path = os.path.join(preset_dir, "formatted_transactions.csv")
                    shutil.move(formatted_path, destination_path)
                    print(f"Moved {formatted_path} to {destination_path}")

            edges_mean, edges_std = mean_stddev(edges_list)
            clusters_mean, clusters_std = mean_stddev(clusters_list)
            laundering_mean, laundering_std = mean_stddev(laundering_list)

            line = (f"preset_{i}_{j*20}: "
                    f"{edges_mean:.5f}+-{edges_std:.5f}, "
                    f"{clusters_mean:.5f}+-{clusters_std:.5f}, "
                    f"{laundering_mean:.5f}+-{laundering_std:.5f}\n")

            with open("perturbation_metrics.txt", "a") as f:
                f.write(line)



handle_everything()
