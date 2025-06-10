import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_test_data = '../../data/exam_approach/test_results/claude-3-7-sonnet-20250219/'

path_epoch = '../../data/external/epoch_ai/'

files_score = {
    "business_and_financial_operations": "scores_only_business_and_financial_operations_occupations.csv",
    "computer_and_mathematical": "scores_only_computer_and_mathematical_occupations.csv",
    "management": "scores_only_management_occupations.csv"
}

file_full_exams = {
    "business_and_financial_operations": 'test_answers_business_and_financial_operations_occupations.csv',
    "computer_and_mathematical": 'test_answers_computer_and_mathematical_occupations.csv',
    "management": 'test_answers_management_occupations.csv'
}

df_test = pd.read_csv(path_test_data + file_full_exams['business_and_financial_operations'])
df_test.columns
df_test[['task_id', 'task_description', 'exam', 'instructions']].iloc[2]

# Initialize an empty list to store DataFrames
# NOTE the code below is not used, all_exams re written by exams in lines below
df_exams = []
# Loop through the dictionary to process each file
for category, file_name in files_score.items():
    df = pd.read_csv(path_test_data + file_name)
    # Remove the 'Unnamed: 0' column
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Add the 'occupation_category' column
    df['occupation_category'] = category
    # Fill since the begingin NA with 0
    # df = df.fillna(0)
    # Now read the exam to get exam length
    df_full = pd.read_csv(path_test_data + file_full_exams[category])
    df_full['exam_length'] = df_full['exam'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    dict_task_examlength = dict(zip(df_full['task_id'], df_full['exam_length']))
    df['exam_length'] = df['task_id'].map(dict_task_examlength)
     # Append the processed DataFrame to the list
    df_exams.append(df)


df_exams[0].head()

# Concatenate all DataFrames into one
all_exams = pd.concat(df_exams, ignore_index=True)
all_exams.head()
# all_exams = all_exams.fillna(0)


occupations =['Business and Financial Operations Occupations',
'Computer and Mathematical Occupations',
'Management Occupations']

occupations_file_names = [occ.lower().replace(' ', '_') for occ in occupations]

exam_list = pd.DataFrame()
for occ in occupations_file_names:
    results = pd.read_csv(f'../../data/exam_approach/test_results/claude-3-7-sonnet-20250219/test_results_{occ}.csv',index_col=0)
    results = results.loc[:, ~results.columns.str.startswith('Unnamed')]
    results['occupation_group'] = occ
    full_results = pd.read_csv(f'../../data/exam_approach/test_results/claude-3-7-sonnet-20250219/test_answers_{occ}.csv', index_col=0)
    full_results['exam_length'] = full_results['exam'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    dict_task_examlength = dict(zip(full_results['task_id'], full_results['exam_length']))
    results['exam_length'] = results['task_id'].map(dict_task_examlength)
     # Append the processed DataFrame to the list

    exam_list = pd.concat([exam_list, results], axis=0, ignore_index=True)

# mark exams with empty entry, nan entry or key grade scores over 100 as invalid
exam_list.loc[exam_list['exam']=='','exam'] = 'Exam not valid'
exam_list['exam'] = exam_list['exam'].fillna('Exam not valid')
exam_list.loc[exam_list['key_grade']>100,'exam'] = 'Exam not valid'
exam_list.loc[exam_list['check_overall_makes_sense']==False, 'exam'] = 'Exam not valid'

exams = exam_list[exam_list['exam'] !='Exam not valid']

exams[['task_id', 'score_chatgpt4o','score_chatgpt35']][exams['task_id'] ==  21522]

columns_to_plot = ['score_chatgpt_o3', 'score_claude_sonnet', 'score_gemini_25']
columns_to_plot = ['score_gemini_25', 'score_claude_sonnet',
       'score_chatgpt_o3', 'score_deepseek']

exams.head()
# exams['exam_length']
all_exams=exams

# Create a histogram for each column
for column in columns_to_plot:
    plt.figure(figsize=(8, 6))
    # all_exams = all_exams.fillna(0)
    plt.hist(all_exams[column], bins=10, color='blue', edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of {column}', fontsize=16)
    plt.xlabel('Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

exams[['task_id', 'score_chatgpt4o','score_chatgpt35']][exams['task_id'] ==  21522]



model_family = ["GPT", "Claude",  "Gemini", "DeepSeek"]
df_model_info = pd.read_csv(path_epoch + 'notable_ai_models.csv')
df_model_info = df_model_info[df_model_info['Model'].str.contains('|'.join(model_family), case=False, na=False)]

print(df_model_info['Model'].to_list())

model_dict = {
    "claude-3-7-sonnet-20250219": ["claude-3-7-sonnet-20250219", "Claude 3.7 Sonnet"],
    "gpt-4o": ["gpt-4o-2024-08-06", "GPT-4o"],
    "deepseek-chat": ["DeepSeek-V3", "DeepSeek-V3"],
    "gemini-1.5-flash": ["gemini-1.5-flash-002", "Gemini 1.5 Pro"],
    "gemini-2.0-flash": ["gemini-2.0-flash-001", "NA"],
    "claude-3-5-sonnet-202410": ["claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet"],
    "gpt-3.5-turbo-0125": ['gpt-3.5-turbo-0125', "GPT-3.5 Turbo"],
    "gemini-2.5-pro-preview-03-25": ["gemini-2.5-flash-preview-04-17", "NA"],
    "o3-2025-04-16": ["o3-2025-04-16_high", "NA"],
    "claude-3-sonnet": ['claude-3-sonnet-20240229', 'Claude 3 Sonnet']
}

data = []
for model_key, values in model_dict.items():
    # Extract the second value from the dict list for df_model_info lookup
    model_info_key = values[1]
    # Get the row from df_model_info matching the model_info_key
    info_row = df_model_info[df_model_info['Model'] == model_info_key]

    # Extract the required values from df_model_info and df_model_benchmark
    row = {
        "model": model_key,
        "Publication date": info_row['Publication date'].values[0] if not info_row.empty else None,
        "Organization": info_row['Organization'].values[0] if not info_row.empty else None,
        "Organization categorization": info_row['Organization categorization'].values[0] if not info_row.empty else None,
        "Parameters": info_row['Parameters'].values[0] if not info_row.empty else None,
        "Training compute (FLOP)": info_row['Training compute (FLOP)'].values[0] if not info_row.empty else None,
        "Training time (hours)": info_row['Training time (hours)'].values[0] if not info_row.empty else None,
        "Training compute cost (2023 USD)": info_row['Training compute cost (2023 USD)'].values[0] if not info_row.empty else None,
         }
    # Append the row to the data list
    data.append(row)

columns = [
    "model", "Publication date", "Organization", "Organization categorization",
    "Parameters", "Training compute (FLOP)", "Training time (hours)",
    "Training compute cost (2023 USD)", "Model accessibility"
]
df_model_bench = pd.DataFrame(data, columns=columns)
df_model_bench.loc[df_model_bench['model'] == "gemini-2.0-flash", "Publication date"] = "2025-02-05"
df_model_bench.loc[df_model_bench['model'] == "gemini-2.0-flash", "Training compute (FLOP)"] = 2.43e+25
df_model_bench.loc[df_model_bench['model'] == "gpt-3.5-turbo-0125", "Training compute (FLOP)"] = 2.58e+24
df_model_bench.loc[df_model_bench['model'] == "gpt-3.5-turbo-0125", "Training compute (FLOP)"] = 2.58e+24
df_model_bench.loc[df_model_bench['model'] == "gemini-2.5-pro-preview-03-25", "Publication date"] = "2025-03-01"
df_model_bench.loc[df_model_bench['model'] == "o3-2025-04-16", "Publication date"] = "2025-01-31"
df_model_bench.loc[df_model_bench['model'] == "gemini-2.5-pro-preview-03-25", "Training compute (FLOP)"] = 5.6e+25
df_model_bench.loc[df_model_bench['model'] == "o3-2025-04-16", "Training compute (FLOP)"] = 8e+25 # taken from https://www.lesswrong.com/posts/NXTkEiaLA4JdS5vSZ/what-o3-becomes-by-2028

df_model_bench.loc[df_model_bench['model'] == "gemini-2.0-flash", "Organization"] = 'Google DeepMind'
df_model_bench.loc[df_model_bench['model'] == "gemini-2.5-pro-preview-03-25", "Organization"] = 'Google DeepMind'
df_model_bench.loc[df_model_bench['model'] == "o3-2025-04-16", "Organization"] = 'OpenAI'


full_mapping = {
    "claude-3-7-sonnet-20250219": "score_claude_sonnet",
    "gpt-4o": "score_chatgpt4o",
    "deepseek-chat": "score_deepseek",
    "gemini-1.5-flash": "score_gemini_flash_15",
    "gemini-2.0-flash": "score_gemini_flash",
    "claude-3-5-sonnet-202410": "score_claude_sonnet_35",
    "gpt-3.5-turbo-0125": "score_chatgpt35",
    "gemini-2.5-pro-preview-03-25": "score_gemini_25",
    "o3-2025-04-16": "score_chatgpt_o3",
    "claude-3-sonnet": 'score_sonnet30'
}

df_model_data_score = df_model_bench[['model', 'Publication date', 'Organization']]
df_model_data_score.head()
all_exams.columns


# Prepare a list to collect individual model test score DataFrames
model_score_rows = []

# Loop through each model and corresponding score column
for model_key, score_col in full_mapping.items():
    if score_col not in all_exams.columns:
        continue

    # Extract just the relevant columns from all_exams for this model
    temp = all_exams[['task_id', 'occupation', 'occupation_group', 'task_description', 'exam_length', score_col]].copy()
    temp = temp.rename(columns={score_col: 'score'})

    # Add model metadata
    model_info = df_model_data_score[df_model_data_score['model'] == model_key].iloc[0]
    temp['model'] = model_key
    temp['Publication date'] = model_info['Publication date']
    temp['Organization'] = model_info['Organization']

    # Append to list
    model_score_rows.append(temp)



# Combine all model-specific rows into one DataFrame
df_model_test_scores = pd.concat(model_score_rows, ignore_index=True)

df_model_test_scores['Publication date'] = pd.to_datetime(df_model_test_scores['Publication date'], errors='coerce')

# Sort the DataFrame by 'Publication date'
df_model_test_scores = df_model_test_scores.sort_values(by='Publication date').reset_index(drop=True)


df_model_test_scores.head()

df_model_test_scores.to_csv( '../../results/tables/df_model_test_scores.csv', index=False)

df_model_test_scores

df_model_test_scores[df_model_test_scores['task_id'] == 21522]
df_model_test_scores[df_model_test_scores['task_id'] == 21315]
len(df_model_test_scores[df_model_test_scores['task_id'] == 21522])