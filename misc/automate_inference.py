import os
import subprocess
import re
import statistics
import sys

def extract_test_metrics(output):
    # Extract Test F1, Precision, Recall
    f1_match = re.search(r"Test F1:\s+([0-9.]+)", output)
    prec_match = re.search(r"Test Precision:\s+([0-9.]+)", output)
    recall_match = re.search(r"Test Recall:\s+([0-9.]+)", output)

    if f1_match and prec_match and recall_match:
        f1 = float(f1_match.group(1)) * 100
        precision = float(prec_match.group(1)) * 100
        recall = float(recall_match.group(1)) * 100
        return f1, precision, recall
    else:
        print("NO MATCH FOUND for one or more Test metrics:")
        return None, None, None

PYTHON_PATH = os.path.expanduser("~/.conda/envs/megagnn/bin/python")

def run_model(data_dir, model_name, unique_name):
    cmd = [
        PYTHON_PATH, "main.py",
        "--inference",
        "--data", data_dir,
        "--model", model_name,
        "--emlps",
        "--reverse_mp",
        "--ego",
        "--flatten_edges",
        "--edge_agg_type", model_name,
        "--task", "edge_class",
        "--unique_name", unique_name
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=os.environ)
        print(result.stdout)
        return extract_test_metrics(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running model for {data_dir} with {model_name}")
        print(e.stderr)
        return None, None, None

def main():
    os.chdir(os.path.expanduser("~/RP/MEGA-GNN")) # Change to your MEGA-GNN location
    data_root = "../data" # Change to your data location
    results_file = "../results.txt" #Change to wherever you want to store the results
    
    preset_groups = {}
    for dirname in os.listdir(data_root):
        if dirname.startswith("preset") and "_" in dirname:
            parts = dirname.split("_")
            if len(parts) == 3:
                i = parts[0].replace("preset", "")
                j = parts[1]
                k = parts[2]
                key = (i, j)
                preset_groups.setdefault(key, []).append(dirname)

    with open(results_file, "w") as f_out:
        for model in ["gin", "pna"]:
        # for model in ["pna"]:
        # for model in ["gin"]:
            for (i, j), dirnames in sorted(preset_groups.items()):
                f1_scores = []
                precision_scores = []
                recall_scores = []
                for dirname in sorted(dirnames):
                    f1, precision, recall = run_model(dirname, model, "22" if model == "gin" else "newpna_15")
                    if None not in (f1, precision, recall):
                        f1_scores.append(f1)
                        precision_scores.append(precision)
                        recall_scores.append(recall)

                if f1_scores:
                    mean_f1 = statistics.mean(f1_scores)
                    std_f1 = statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0

                    mean_prec = statistics.mean(precision_scores)
                    std_prec = statistics.stdev(precision_scores) if len(precision_scores) > 1 else 0.0

                    mean_recall = statistics.mean(recall_scores)
                    std_recall = statistics.stdev(recall_scores) if len(recall_scores) > 1 else 0.0

                    label = f"preset{i}_{int(j)}_{model}"
                    f_out.write(f"{label}:\n")
                    f_out.write(f"  Test F1: {mean_f1:.2f}% ± {std_f1:.2f}%\n")
                    f_out.write(f"  Test Precision: {mean_prec:.2f}% ± {std_prec:.2f}%\n")
                    f_out.write(f"  Test Recall: {mean_recall:.2f}% ± {std_recall:.2f}%\n")
                    print(f"Recorded: {label}:")
                    print(f"  Test F1: {mean_f1:.2f}% ± {std_f1:.2f}%")
                    print(f"  Test Precision: {mean_prec:.2f}% ± {std_prec:.2f}%")
                    print(f"  Test Recall: {mean_recall:.2f}% ± {std_recall:.2f}%")
                else:
                    print(f"No valid scores found for {model}_preset{i}_{j}")

main()
