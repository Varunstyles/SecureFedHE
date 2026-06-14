import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(log_dir):
    dfs = []
    for f in os.listdir(log_dir):
        if f.endswith("_metrics.csv"):
            path = os.path.join(log_dir, f)
            try:
                df = pd.read_csv(path)
                df['source_file'] = f
                # Determine phase/epsilon from filename
                if 'baseline' in f:
                    df['label'] = 'Baseline FL'
                elif 'ring' in f:
                    df['label'] = 'Ring Topology'
                elif 'eps10' in f:
                    df['label'] = 'Selective HE (ε=10)'
                elif 'eps20' in f:
                    df['label'] = 'Selective HE (ε=20)'
                elif 'eps50' in f:
                    df['label'] = 'Selective HE (ε=50)'
                else:
                    df['label'] = 'Unknown'
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)

def plot_accuracy(df, out_dir):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='round_num', y='eval_acc', hue='label', marker='o')
    plt.title('Test Accuracy over Rounds', fontsize=16)
    plt.xlabel('Communication Round', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Method')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'accuracy_comparison.png'), dpi=300)
    plt.savefig(os.path.join(out_dir, 'accuracy_comparison.pdf'), dpi=300)
    plt.close()

def plot_time(df, out_dir):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='label', y='wall_time_s', errorbar='sd')
    plt.title('Average Wall Time per Round', fontsize=16)
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Wall Time (seconds)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'time_comparison.png'), dpi=300)
    plt.savefig(os.path.join(out_dir, 'time_comparison.pdf'), dpi=300)
    plt.close()

def plot_overhead(df, out_dir):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='label', y='enc_overhead_s')
    plt.title('Encryption Overhead per Round', fontsize=16)
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Encryption Time (seconds)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'encryption_overhead.png'), dpi=300)
    plt.savefig(os.path.join(out_dir, 'encryption_overhead.pdf'), dpi=300)
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, 'logs')
    out_dir = os.path.join(base_dir, 'plots')
    
    os.makedirs(out_dir, exist_ok=True)
    
    df = load_data(log_dir)
    if df is None:
        print("No CSV files found in logs directory.")
        return
    
    print("Generating Accuracy vs Round graph...")
    plot_accuracy(df, out_dir)
    
    print("Generating Wall Time comparison graph...")
    plot_time(df, out_dir)
    
    print("Generating Encryption Overhead graph...")
    plot_overhead(df, out_dir)
    
    print(f"All graphs generated successfully in {out_dir}.")

if __name__ == '__main__':
    main()
