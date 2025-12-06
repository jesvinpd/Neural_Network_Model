import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('weights_log.csv')

# Filter for Node 1
node_data = df[df['Node'] == 1].copy()

# Create a single plot
plt.figure(figsize=(12, 6))

# Plot each weight type with different colors
weight_types = ['input', 'output', 'bias']
colors = ['blue', 'green', 'red']
markers = ['o', 's', '^']

for weight_type, color, marker in zip(weight_types, colors, markers):
    type_data = node_data[node_data['WeightType'] == weight_type]
    
    if not type_data.empty:
        # Plot each weight index separately
        for weight_idx in type_data['WeightIndex'].unique():
            weight_data = type_data[type_data['WeightIndex'] == weight_idx]
            label = f'{weight_type} W{int(weight_idx)}' if weight_type != 'bias' else 'Bias'
            plt.plot(weight_data['Epoch'], weight_data['WeightValue'], 
                    marker=marker, markersize=4, linewidth=1.5, 
                    color=color, label=label, alpha=0.7)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Weight Value', fontsize=12)
plt.title('Node 1 Weight Values Across Epochs', fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('node1_weights_simple.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'node1_weights_simple.png'")
plt.show()