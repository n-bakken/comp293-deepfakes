import pandas as pd
import matplotlib.pyplot as plt

#filename = 'for-rerec.csv'
filename = 'for-original.csv'
#filename = 'for-norm.csv'
#filename = 'for-2sec.csv'

df = pd.read_csv(filename)

# Step 2: Add new columns for average precision, recall, and F1-score
df['Precision'] = (df['Precision (Real)'] + df['Precision (Fake)']) / 2
df['Recall'] = (df['Recall (Real)'] + df['Recall (Fake)']) / 2
df['F1'] = (df['F1-Score (Real)'] + df['F1-Score (Fake)']) / 2

# Step 3: Plot the data
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
x = range(len(df['Algorithm']))  # Positions for algorithms

fig, ax = plt.subplots(figsize=(10, 6))

# Create bars for each metric
for i, metric in enumerate(metrics):
    ax.bar(
        [pos + 0.2 * i for pos in x],  # Offset each bar group
        df[metric],
        width=0.2,
        label=metric
    )

# Customize the plot
ax.set_xticks([pos + 0.3 for pos in x])
ax.set_xticklabels(df['Algorithm'])
ax.set_ylabel('Score')
newname = filename[:-4]
ax.set_title(f'Performance Metrics for {newname}')
ax.legend(title='Metric')

# Set the y-axis limits to zoom in
ax.set_ylim(0.95, 1.00001)  # Adjust the range as needed

plt.tight_layout()
# Save the plot to a file instead of showing it
plt.savefig(f'performance_metrics_{filename}.png', dpi=300, bbox_inches='tight')
