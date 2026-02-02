import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit

# Load the dataset
# Using 2025 dataset which should contain the most recent chips
import os
csv_path = os.path.join('..', 'data', 'peak_accelerators_ieee_hpec_2025.csv')
df = pd.read_csv(csv_path)

# Clean up column names (strip whitespace)
df.columns = df.columns.str.strip()

# Define a mapping function to assign process nodes based on Product Name
def get_process_node(row):
    product = str(row['Product']).lower()
    company = str(row['Company']).lower()
    
    # NVIDIA
    if 'nvidia' in company:
        if 'h100' in product: return 4
        if 'b200' in product: return 4 # 4NP
        if 'a100' in product: return 7
        if 'v100' in product: return 12
        if 'p100' in product: return 16
        if 'k80' in product: return 28
        if 't4' in product: return 12
        if 'l40' in product: return 5 # Ada Lovelace is 4N (5nm class)
        if 'a10' in product and 'a100' not in product: return 8 # Ampere consumer/ws is Samsung 8nm usually, but A10 is Server. Check? A10 is 8nm Samsung.
        
    # Cerebras
    if 'cerebras' in company:
        if 'wse-3' in product or 'cs-3' in product: return 5
        if 'wse-2' in product or 'cs-2' in product: return 7
        if 'wse-1' in product or 'cs-1' in product: return 16
        
    # Groq
    if 'groq' in company:
        return 14 # TSP v1
        
    # Tenstorrent
    if 'tenstorrent' in company:
        if 'blackhole' in product: return 6
        if 'wormhole' in product: return 12 # Early wormhole was 12nm, some sources say 6nm intended. Sticking to safer 12nm or excluding if unsure.
        if 'greyskull' in product: return 12

    # Graphcore
    if 'graphcore' in company:
        if 'mk1' in product: return 16
        if 'mk2' in product: return 7
        if 'bow' in product: return 7 # Wafer on wafer TSMC 7nm

    # Google TPU
    if 'google' in company:
        if 'tpu v1' in product: return 28
        if 'tpu v2' in product: return 16
        if 'tpu v3' in product: return 16
        if 'tpu v4' in product: return 7
        if 'tpu v5' in product or 'tpuv5' in product: return 5 # TPU v5p

    return None

# Apply the mapping
df['ProcessNode_nm'] = df.apply(get_process_node, axis=1)

# Filter out rows where ProcessNode is NaN or Power/Perf is missing/zero
df_filtered = df.dropna(subset=['ProcessNode_nm', 'Power', 'PeakPerformance']).copy()
df_filtered = df_filtered[df_filtered['Power'] > 0]
df_filtered = df_filtered[df_filtered['PeakPerformance'] > 0]

# Calculate Performance per Watt (TFLOPS/W for easier reading)
# PeakPerformance is usually in pure FLOPS. Divide by 1e12 for Tera.
df_filtered['Perf_TFLOPS'] = df_filtered['PeakPerformance'] / 1e12
df_filtered['PerfPerWatt'] = df_filtered['Perf_TFLOPS'] / df_filtered['Power']

# Normalize Precision for grouping
df_filtered['Precision_Group'] = df_filtered['Precision'].apply(lambda x: 'Half (FP16/BF16)' if x in ['fp16', 'bf16'] else ('Int8' if x == 'int8' else 'Other'))

# Select relevant columns for inspection
print("Data used for analysis:")
print(df_filtered[['Company', 'Product', 'ProcessNode_nm', 'Precision', 'PerfPerWatt']].sort_values('ProcessNode_nm', ascending=False))

# Normalizing Company names for easier filtering
df_filtered['Company'] = df_filtered['Company'].astype(str).str.strip()

# --- Modeling (NVIDIA only for consistent architecture/precision scaling) ---
# Filter for NVIDIA and typically FP16/BF16 to ensure we compare apples-to-apples.
# (Most NVIDIA datacenter GPUs listed will use FP16/BF16 for AI Peak stats)
df_nvidia = df_filtered[
    (df_filtered['Company'].str.contains('NVIDIA', case=False)) & 
    (df_filtered['Precision_Group'] == 'Half (FP16/BF16)')
].copy()

if len(df_nvidia) < 3:
    # If strictly filtering for Half precision removes too many (older might be fp32), 
    # lets loosen precision but keep company strict if needed, or rely on available data.
    # K80/P100 might be listed as different precision or not mapped?
    # Let's fallback to just NVIDIA if count is low, but print warning.
    print(f"Warning: Only {len(df_nvidia)} NVIDIA (FP16) points found. Using all NVIDIA points.")
    df_nvidia = df_filtered[df_filtered['Company'].str.contains('NVIDIA', case=False)].copy()

print("NVIDIA Data used for model:")
print(df_nvidia[['Product', 'ProcessNode_nm', 'Precision', 'PerfPerWatt']].sort_values('ProcessNode_nm', ascending=False))

# Use NVIDIA data for the model
X_model = df_nvidia['ProcessNode_nm'].values
Y_model = df_nvidia['PerfPerWatt'].values

# Fit curve
def power_law(x, a, b):
    # a * x^b
    return a * np.power(x, b)

try:
    if len(X_model) >= 3:
        popt, pcov = curve_fit(power_law, X_model, Y_model, maxfev=2000)
        a_fit, b_fit = popt
        print(f"\nFitted Model (NVIDIA Only): Efficiency (TFLOPS/W) = {a_fit:.4f} * (Node_nm)^{b_fit:.4f}")
    else:
        print("Not enough NVIDIA data points for curve fitting.")
        a_fit, b_fit = 0, 0
except Exception as e:
    print(f"Could not fit power law model: {e}")
    a_fit, b_fit = 0, 0

# --- Predictive Analysis Calculation (Restored) ---
# Format: (Company, Product Substring, CurrentNode, TargetNodes, ReferencePrecision)
print("\n--- Predictive Analysis: What if others use finer nodes? ---")
print(f"Using Scaling Model: Efficiency ~ Node^({b_fit:.4f})")

predictions = []

# Helper to find baseline
def get_baseline_efficiency(company_name, product_substring, precision_group):
    # Try to find specific product first
    subset = df_filtered[
        (df_filtered['Company'].str.contains(company_name, case=False)) & 
        (df_filtered['Product'].str.contains(product_substring, case=False)) &
        (df_filtered['Precision_Group'] == precision_group)
    ]
    if len(subset) == 0:
        # Fallback to just company and precision
        subset = df_filtered[
            (df_filtered['Company'].str.contains(company_name, case=False)) & 
            (df_filtered['Precision_Group'] == precision_group)
        ]
    
    if len(subset) > 0:
        # Take the best performing one as baseline
        best = subset.loc[subset['PerfPerWatt'].idxmax()]
        return best['Product'], best['ProcessNode_nm'], best['PerfPerWatt'], best['Precision']
    return None, None, None, None

vendors_to_predict = [
    {'name': 'Groq', 'prod': 'Tensor', 'prec_group': 'Half (FP16/BF16)', 'targets': [4, 3]}, 
    {'name': 'Tenstorrent', 'prod': 'Wormhole', 'prec_group': 'Other', 'targets': [4, 3]},
    {'name': 'Cerebras', 'prod': 'CS-3', 'prec_group': 'Half (FP16/BF16)', 'targets': [3]}, 
]

prediction_plot_data = []

for v in vendors_to_predict:
    prod, node, eff, prec = get_baseline_efficiency(v['name'], v['prod'], v['prec_group'])
    if prod:
        print(f"\nBaseline: {v['name']} {prod} @ {node}nm ({prec}) = {eff:.2f} TFLOPS/W")
        prediction_plot_data.append({
            'label': f"{v['name']} (Current)",
            'node': node,
            'eff': eff,
            'type': 'Actual',
            'color': 'black'
        })

        for target_node in v['targets']:
            if target_node < node:
                predicted_eff = eff * np.power((target_node / node), b_fit)
                prediction_plot_data.append({
                    'label': f"{v['name']} ({target_node}nm Pred)",
                    'node': target_node,
                    'eff': predicted_eff,
                    'type': 'Prediction',
                    'color': 'red'
                })

# --- Narrative Chart Generation ---
plt.rcParams.update({'font.size': 12})

# Chart 1: The Status Quo (The Illusion)
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=df_filtered, 
    x='ProcessNode_nm', y='PerfPerWatt', 
    hue='Company', style='Precision_Group', 
    s=150, alpha=0.9, palette='deep'
)
plt.title('Stage 1: Current Landscape (Status Quo)\nApparent Gap due to Manufacturing Disparity')
plt.xlabel('Process Node (nm) [Lower is Better]')
plt.ylabel('Efficiency (TFLOPS / Watt)')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig(os.path.join('..', 'figures', 'step1_status_quo.png'))
print("Saved step1_status_quo.png")

# Chart 2: The Physics Law (NVIDIA's Path)
plt.figure(figsize=(12, 7))
# Background points
sns.scatterplot(
    data=df_filtered, 
    x='ProcessNode_nm', y='PerfPerWatt', 
    color='grey', alpha=0.2, s=50, label='Industry Context'
)
# NVIDIA Path
sns.scatterplot(
    data=df_nvidia, 
    x='ProcessNode_nm', y='PerfPerWatt', 
    color='green', s=200, marker='o', label='NVIDIA Lineage'
)
# Trend Line
if a_fit != 0:
    x_trend = np.linspace(3, 28, 100)
    y_trend = power_law(x_trend, a_fit, b_fit)
    plt.plot(x_trend, y_trend, 'g--', linewidth=2, label=f'Physics Law: $E \\propto Node^{{{b_fit:.2f}}}$')

plt.title('Stage 2: The Physics of Scale\nNVIDIA\'s Efficiency is Driven by Node Shrinking')
plt.xlabel('Process Node (nm) [Lower is Better]')
plt.ylabel('Efficiency (TFLOPS / Watt)')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend()
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig(os.path.join('..', 'figures', 'step2_physics_law.png'))
print("Saved step2_physics_law.png")

# Chart 3: The Future (Simulation / Twist)
plt.figure(figsize=(12, 7))
# Background & Trend
plt.plot(x_trend, y_trend, 'g--', linewidth=1, alpha=0.5)
sns.scatterplot(
    data=df_nvidia[df_nvidia['Product'].str.contains('H100|B200')], 
    x='ProcessNode_nm', y='PerfPerWatt', 
    color='green', s=200, label='NVIDIA Current (H100/B200)'
)

# Plot Predictions (The Twist)
for p in prediction_plot_data:
    if p['type'] == 'Prediction':
        plt.plot(p['node'], p['eff'], marker='*', color='red', markersize=25, markeredgecolor='black')
        plt.text(p['node'], p['eff']+0.5, f"{p['label']}\n{p['eff']:.1f}", 
                 color='red', fontsize=10, ha='center', fontweight='bold',
                 bbox=dict(facecolor='white', edgecolor='red', alpha=0.8))
    elif p['type'] == 'Actual' and ('Groq' in p['label'] or 'Tenstorrent' in p['label']):
         plt.scatter(p['node'], p['eff'], marker='s', color='black', s=100, label='Competitor Current')
         plt.arrow(p['node'], p['eff'], prediction_plot_data[0]['node'] - p['node'] if 'Groq' in p['label'] else 0, 
                   prediction_plot_data[2]['eff'] - p['eff'] if 'Groq' in p['label'] else 0, 
                   color='red', alpha=0.3, width=0.1) # Simple visual cue

plt.title('Stage 3: The "What If" Simulation\nLatent Architectural Efficiency Unlocked at 3nm/4nm')
plt.xlabel('Process Node (nm) [Lower is Better]')
plt.ylabel('Efficiency (TFLOPS / Watt)')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(loc='upper right')
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig(os.path.join('..', 'figures', 'step3_simulation.png'))
print("Saved step3_simulation.png")

# Keep the original output as well
output_file = os.path.join('..', 'figures', 'process_node_efficiency_model.png')
plt.savefig(output_file)
print(f"\nPlot saved to {output_file}")
plt.xlabel('Process Node (nm) [Lower is Better]')
plt.ylabel('Efficiency (TFLOPS / Watt)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().invert_xaxis() 
plt.tight_layout()

plt.savefig(output_file)
print(f"\nPlot saved to {output_file}")
