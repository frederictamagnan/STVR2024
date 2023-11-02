import pandas as pd
import matplotlib.pyplot as plt

# Load the JSON lines dataset into a pandas DataFrame
df = pd.read_json('./results/jsonl/all_data_26092.jsonl', lines=True)

# list_group_name=['TF+POH|Kmeans|BUC','DAAE+POH|Kmeans|RS','TermFreq+POH|Kmeans|BUC','-|NoClustering|RS','AE|Kmeans|RS']
# Group the DataFrame by coverage_freq and dataset_name
grouped = df.groupby(['coverage_freq', 'dataset_name'])

# Loop through all unique combinations and create a plot for each
for (freq, name), group in grouped:
    # Sort the filtered DataFrame by cluster_nb in ascending order
    group = group.sort_values(by='cluster_nb')

    # Group the DataFrame by clustering_pipeline and sample_heuristic
    # subgrouped = group.groupby(['clustering_pipeline', 'sample_heuristic'])
    subgrouped=group.groupby(['name_exp'])

    # Create the plot and save it with a name that includes the coverage_freq and dataset_name
    fig, ax = plt.subplots()
    i=0
    for group_name, subgroup in subgrouped:
        # if group_name[0] not in list_group_name:
        #     continue

        group_name_=group_name[0].replace('NoClustering','Baseline')
        x = subgroup['cluster_nb']
        y_mean = subgroup['pattern_coverage'].apply(lambda x: x[0])
        y_std = subgroup['pattern_coverage'].apply(lambda x: x[1])
        ax.plot(x, y_mean, label=group_name_,marker='o')
        ax.fill_between(x, y_mean-y_std, y_mean+y_std, alpha=0.2)
        i+=1
    plt.xlabel('Nb of Clusters : k')
    plt.ylabel('Coverage')
    plt.title('Test Selection Coverage Function')

    ax.legend(loc='lower right')
    figname = f"./results/plot/fig_coverage_freq_{freq}_dataset_{name}_1010.png"
    plt.savefig(figname)
    plt.show()
