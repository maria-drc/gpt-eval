import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pytensor.tensor as pt
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
from scipy.stats import pearsonr
COLOR_PALETTE = ['#FFBD59', '#38B6FF', '#8E3B46', '#E0777D', '#739E82']


df = pd.read_csv('../../results/tables/df_model_test_scores.csv')
df.head()


df_clean = df.dropna(subset=['exam_length', 'score'])
corr, p_value = pearsonr(df_clean['exam_length'], df_clean['score'])
print("Correlation coefficient with outliers:", corr)
print("p-value:", p_value)

df_clean = df.dropna(subset=['exam_length', 'score'])
corr, p_value = pearsonr(df_clean['exam_length'][df_clean['exam_length'] > 25000], df_clean['score'][df_clean['exam_length'] > 25000])
print("Correlation coefficient without outliers:", corr)
print("p-value:", p_value)


df_task = df.groupby('task_id').agg({
    'exam_length': 'first',
    'score': ['mean', 'std'],
    'occupation_group': 'first'
}).reset_index()
df_task.columns = ['task_id', 'exam_length', 'score_mean', 'score_std', 'occupation_group']

# Define a colors mapping using your provided color palette (assume only three groups)
occ_groups = sorted(df_task['occupation_group'].unique())
COLOR_PALETTE = ['#FFBD59', '#38B6FF', '#8E3B46', '#E0777D', '#739E82']
color_mapping = dict(zip(occ_groups, COLOR_PALETTE[:len(occ_groups)]))

df_task = df.groupby('task_id').agg({
    'exam_length': 'first',
    'score': ['mean', 'std'],
    'occupation_group': 'first'
}).reset_index()
df_task.columns = ['task_id', 'exam_length', 'score_mean', 'score_std', 'occupation_group']

# Define a colors mapping using your provided color palette (assume only three groups)
occ_groups = sorted(df_task['occupation_group'].unique())
COLOR_PALETTE = ['#FFBD59', '#38B6FF', '#8E3B46', '#E0777D', '#739E82']
color_mapping = dict(zip(occ_groups, COLOR_PALETTE[:len(occ_groups)]))

plt.figure(figsize=(10, 6))
ax = plt.gca()
# Plot each occupation group with error bars
for occ in occ_groups:
    sub_df = df_task[df_task['occupation_group'] == occ]
    plt.errorbar(sub_df['exam_length'], sub_df['score_mean'],
                 yerr=sub_df['score_std'],
                 fmt='o', label=occ,
                 ecolor='gray', 
                 color=color_mapping[occ],
                 capsize=2, markersize=10, alpha=0.7)

plt.xlabel('Exam Length (characters)', fontsize=16)
plt.ylabel('Average Score', fontsize=16)
plt.ylabel('Mean task score across models', fontsize=16)
plt.ylim([0,100])
# Despine: remove top and right borders.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('../../results/figures/exam_length_vs_score_errorbars.png', bbox_inches='tight')
plt.show()



plt.figure(figsize=(10, 6))
ax = plt.gca()
# Plot each occupation group with error bars
for occ in occ_groups:
    sub_df = df_task[df_task['occupation_group'] == occ]
    plt.scatter(sub_df['exam_length'], sub_df['score_mean'],
            label=occ,
            color=color_mapping[occ],
            s=150,    # adjust marker size as needed
            alpha=0.7)

plt.xlabel('Exam length (characters)', fontsize=16)
plt.ylabel('Mean task score across models', fontsize=16)
# plt.title('Correlation between Exam Length and Average Score per Task', fontsize=18)
plt.ylim([0,100])
# Despine: remove top and right borders.
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('../../results/figures/exam_length_vs_score.png', bbox_inches='tight')
plt.show()