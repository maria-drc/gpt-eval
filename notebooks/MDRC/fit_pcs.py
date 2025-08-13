import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('../../results/tables/df_model_test_scores.csv')
df['Publication date'] = pd.to_datetime(df['Publication date'])
df = df.sort_values('Publication date')

# df = df[df['model'] != 'claude-3-5-sonnet-202410']

# Create task x time matrix
print("Creating task x time matrix...")
task_time_matrix = df.pivot_table(
    values='score', 
    index='task_id', 
    columns='Publication date', 
    aggfunc='mean'  # In case of duplicates
)

print(f"Matrix shape: {task_time_matrix.shape} (tasks x time)")
print(f"Missing values: {task_time_matrix.isna().sum().sum()}")

# Handle missing values - replace with zero
task_time_matrix = task_time_matrix.fillna(0)
print(f"After filling NaN with 0: {task_time_matrix.shape}")

# Use the cleaned matrix
X = task_time_matrix.values  # Tasks x Time
task_ids = task_time_matrix.index.values
time_points = task_time_matrix.columns

print(f"Final matrix: {X.shape} tasks x time points")

# Plot average performance across all tasks over time
plt.figure(figsize=(12, 6))

# Calculate mean and std across tasks for each time point (keeping zeros as zeros)
mean_scores = task_time_matrix.mean(axis=0)
std_scores = task_time_matrix.std(axis=0)
n_tasks_per_time = [len(task_ids)] * len(time_points)

# Plot mean with error bars
plt.errorbar(range(len(time_points)), mean_scores, yerr=std_scores, 
             fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8,
             color='steelblue', label='Mean ± Std')

# Add individual task trajectories (very faint)
for i in range(min(20, len(task_ids))):  # Show max 20 tasks to avoid clutter
    task_series = task_time_matrix.iloc[i]
    plt.plot(range(len(time_points)), task_series, 
            alpha=0.1, color='gray', linewidth=0.5)

plt.xlabel('Time Point')
plt.ylabel('Score')
plt.title('Average Performance Across All Tasks Over Time')
plt.xticks(range(len(time_points)), 
           [d.strftime('%Y-%m') for d in time_points], 
           rotation=45, ha='right')
plt.grid(alpha=0.3)
plt.legend()

# Add text showing number of tasks per time point
for i, n in enumerate(n_tasks_per_time):
    plt.text(i, plt.ylim()[0] + 0.02*(plt.ylim()[1]-plt.ylim()[0]), 
             f'n={n}', ha='center', va='bottom', fontsize=8, alpha=0.7)

plt.tight_layout()
plt.show()

print(f"Average performance over time (including zeros):")
for i, (date, score) in enumerate(zip(time_points, mean_scores)):
    print(f"{date.strftime('%Y-%m')}: {score:.1f} (all {len(task_ids)} tasks)")
print()

# Standardize the time series (across time for each task)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.T).T  # Standardize each task's time series

# PCA Analysis
print("\nRunning PCA...")
pca = PCA()
pca_scores = pca.fit_transform(X_scaled)  # Tasks in PC space

# Explained variance
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"PC1 explains {explained_var[0]:.1%} of variance")
print(f"PC2 explains {explained_var[1]:.1%} of variance") 
print(f"PC3 explains {explained_var[2]:.1%} of variance")
print(f"First 3 PCs explain {cumulative_var[2]:.1%} of variance")

# Plot explained variance
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(explained_var)+1), explained_var, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_var)+1), cumulative_var, 'ro-')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Variance Explained')
plt.axhline(y=0.8, color='k', linestyle='--', alpha=0.5, label='80%')
plt.axhline(y=0.9, color='k', linestyle='--', alpha=0.5, label='90%')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Clustering based on first few PCs
n_components = 3  # Use first 3 PCs
pca_subset = pca_scores[:, :n_components]

# Try different numbers of clusters
max_clusters = min(10, len(task_ids)//2)
inertias = []
silhouette_scores = []

from sklearn.metrics import silhouette_score

for k in range(2, max_clusters+1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_subset)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(pca_subset, cluster_labels)
    silhouette_scores.append(sil_score)

# Plot elbow curve
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(2, max_clusters+1), inertias, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(2, max_clusters+1), silhouette_scores, 'go-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Choose optimal number of clusters (highest silhouette score)
optimal_k = range(2, max_clusters+1)[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters: {optimal_k}")

# Final clustering
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans_final.fit_predict(pca_subset)

# Visualize clusters in PC space
plt.figure(figsize=(15, 5))

# PC1 vs PC2
plt.subplot(1, 3, 1)
scatter = plt.scatter(pca_scores[:, 0], pca_scores[:, 1], c=cluster_labels, cmap='tab10', s=60, alpha=0.7)
plt.xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
plt.ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
plt.title('Tasks in PC1-PC2 Space')
plt.colorbar(scatter, label='Cluster')
plt.grid(alpha=0.3)

# PC1 vs PC3  
plt.subplot(1, 3, 2)
scatter = plt.scatter(pca_scores[:, 0], pca_scores[:, 2], c=cluster_labels, cmap='tab10', s=60, alpha=0.7)
plt.xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
plt.ylabel(f'PC3 ({explained_var[2]:.1%} variance)')
plt.title('Tasks in PC1-PC3 Space')
plt.colorbar(scatter, label='Cluster')
plt.grid(alpha=0.3)

# PC2 vs PC3
plt.subplot(1, 3, 3)
scatter = plt.scatter(pca_scores[:, 1], pca_scores[:, 2], c=cluster_labels, cmap='tab10', s=60, alpha=0.7)
plt.xlabel(f'PC2 ({explained_var[1]:.1%} variance)')
plt.ylabel(f'PC3 ({explained_var[2]:.1%} variance)')
plt.title('Tasks in PC2-PC3 Space')
plt.colorbar(scatter, label='Cluster')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Show cluster assignments
print(f"\nCluster assignments:")
cluster_df = pd.DataFrame({
    'task_id': task_ids,
    'cluster': cluster_labels,
    'PC1': pca_scores[:, 0],
    'PC2': pca_scores[:, 1], 
    'PC3': pca_scores[:, 2]
})

for cluster_id in range(optimal_k):
    tasks_in_cluster = cluster_df[cluster_df['cluster'] == cluster_id]['task_id'].values
    print(f"Cluster {cluster_id}: {len(tasks_in_cluster)} tasks")
    print(f"  Tasks: {tasks_in_cluster[:10]}{'...' if len(tasks_in_cluster) > 10 else ''}")

# Plot average time series by cluster
plt.figure(figsize=(12, 8))
colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))

for cluster_id in range(optimal_k):
    cluster_tasks = task_ids[cluster_labels == cluster_id]
    cluster_data = task_time_matrix.loc[cluster_tasks]
    
    # Plot individual tasks (thin lines)
    for _, task_series in cluster_data.iterrows():
        plt.plot(range(len(time_points)), task_series.values, 
                color=colors[cluster_id], alpha=0.2, linewidth=0.5)
    
    # Plot cluster average (thick line)  
    cluster_mean = cluster_data.mean(axis=0)
    plt.plot(range(len(time_points)), cluster_mean.values,
            color=colors[cluster_id], linewidth=3, 
            label=f'Cluster {cluster_id} (n={len(cluster_tasks)})')

plt.xlabel('Time Point')
plt.ylabel('Score')
plt.title('Task Clusters: Time Series Patterns')
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(range(len(time_points)), 
           [d.strftime('%Y-%m') for d in time_points], 
           rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Principal components interpretation
print(f"\nPrincipal Component Loadings (first 3 PCs):")
pc_loadings = pca.components_[:3, :].T  # Time x PC
loadings_df = pd.DataFrame(pc_loadings, 
                          index=[d.strftime('%Y-%m') for d in time_points],
                          columns=['PC1', 'PC2', 'PC3'])
print(loadings_df.round(3))

# Plot PC loadings
plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(range(len(time_points)), pc_loadings[:, i], 'o-', linewidth=2)
    plt.xlabel('Time Point')
    plt.ylabel(f'PC{i+1} Loading')
    plt.title(f'PC{i+1} ({explained_var[i]:.1%} variance)')
    plt.grid(alpha=0.3)
    plt.xticks(range(len(time_points)), 
               [d.strftime('%Y-%m') for d in time_points], 
               rotation=45, ha='right')

plt.tight_layout()
plt.show()

print(f"\nSummary:")
print(f"- Analyzed {len(task_ids)} tasks across {len(time_points)} time points")
print(f"- First 3 PCs explain {cumulative_var[2]:.1%} of variance")
print(f"- Optimal clustering: {optimal_k} clusters")
print(f"- Cluster assignments saved in cluster_df DataFrame")


# Show cluster assignments
print(f"\nCluster assignments:")
cluster_df = pd.DataFrame({
    'task_id': task_ids,
    'cluster': cluster_labels,
    'PC1': pca_scores[:, 0],
    'PC2': pca_scores[:, 1], 
    'PC3': pca_scores[:, 2]
})

for cluster_id in range(optimal_k):
    tasks_in_cluster = cluster_df[cluster_df['cluster'] == cluster_id]['task_id'].values
    print(f"Cluster {cluster_id}: {len(tasks_in_cluster)} tasks")
    print(f"  Tasks: {tasks_in_cluster[:10]}{'...' if len(tasks_in_cluster) > 10 else ''}")

# Create word clouds for each cluster
print(f"\nCreating word clouds for cluster task descriptions...")

# Get unique task descriptions (avoid duplicates from same task_id)
task_descriptions = df.groupby('task_id')['task_description'].first().reset_index()

# Merge with cluster assignments
cluster_descriptions = cluster_df.merge(task_descriptions, on='task_id', how='left')

# Check for missing descriptions
missing_desc = cluster_descriptions['task_description'].isna().sum()
if missing_desc > 0:
    print(f"Warning: {missing_desc} tasks missing descriptions")
    cluster_descriptions = cluster_descriptions.dropna(subset=['task_description'])

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create subplots for word clouds
fig, axes = plt.subplots(2, (optimal_k + 1) // 2, figsize=(15, 8))
if optimal_k == 1:
    axes = [axes]
elif optimal_k <= 2:
    axes = axes.flatten()
else:
    axes = axes.flatten()

for cluster_id in range(optimal_k):
    # Get descriptions for this cluster
    cluster_desc = cluster_descriptions[cluster_descriptions['cluster'] == cluster_id]
    
    if len(cluster_desc) == 0:
        continue
        
    # Combine all descriptions into one text
    combined_text = ' '.join(cluster_desc['task_description'].fillna(''))
    
    if len(combined_text.strip()) == 0:
        continue
    
    # Create word cloud
    wordcloud = WordCloud(
        width=400, height=300,
        background_color='white',
        max_words=50,
        colormap='viridis',
        stopwords={'and', 'or', 'the', 'to', 'of', 'for', 'in', 'on', 'with', 'by', 'from', 'such', 'as', 'that', 'this', 'is', 'are', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'other', 'others', 'all', 'any', 'some', 'each', 'every', 'their', 'them', 'they', 'its', 'it', 'his', 'her', 'him', 'she', 'he', 'we', 'us', 'our', 'you', 'your'}
    ).generate(combined_text)
    
    # Plot
    ax = axes[cluster_id] if optimal_k > 1 else axes[0]
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(f'Cluster {cluster_id}\n({len(cluster_desc)} tasks)', fontsize=12)
    ax.axis('off')
