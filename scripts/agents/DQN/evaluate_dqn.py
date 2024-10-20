import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV files
data_baseline = pd.read_csv("logs/baseline_dqn_evaluation_results.csv")
data_improvement_1 = pd.read_csv("logs/improvement_1_dqn_evaluation_results.csv")

# Filter the data where 'Run' column is 'Average'
average_baseline = data_baseline[data_baseline['Run'] == 'Average']
average_improvement_1 = data_improvement_1[data_improvement_1['Run'] == 'Average']

# Create a DataFrame with the values for plotting
plot_data = pd.DataFrame({
    'Implementation': ['Baseline', 'Baseline', 'Improvement 1', 'Improvement 1'],
    'Metric': ['Steps', 'Total Reward', 'Steps', 'Total Reward'],
    'Value': [
        average_baseline['Steps'].values[0], 
        average_baseline['Total Reward'].values[0], 
        average_improvement_1['Steps'].values[0], 
        average_improvement_1['Total Reward'].values[0]
    ]
})

# Plot the bar chart
plt.figure(figsize=(8, 6))
sns.barplot(x='Implementation', y='Value', data=plot_data, palette='viridis', hue='Metric')
plt.title('Average Steps and Total Reward for DQN Implementations over 5 runs')
plt.ylabel('Value')
plt.xlabel('Implementation')
plt.legend(title='Metric')
plt.tight_layout()
plt.savefig('data/visualizations/dqn_evaluation.pdf')
plt.show()
