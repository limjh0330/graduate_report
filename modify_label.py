import pandas as pd

# Load the two files
data_entry_df = pd.read_csv('/mnt/data/Data_Entry_2017_v2020.csv')
nih_labels_df = pd.read_csv('/mnt/data/nih_labels.csv')

# Checking the first few rows of both dataframes to understand their structure
data_entry_df.head(), nih_labels_df.head()

# Extract unique disease labels from 'Finding Labels' in the data_entry_df
disease_labels = set()
for labels in data_entry_df['Finding Labels']:
    for label in labels.split('|'):
        disease_labels.add(label.strip())

# Add the 'No Finding' label to the set of disease labels
disease_labels.add('No Finding')

# Initialize new columns for each disease in the data_entry_df
for label in disease_labels:
    data_entry_df[label] = 0

# Populate the columns based on the 'Finding Labels'
for index, row in data_entry_df.iterrows():
    labels = row['Finding Labels'].split('|')
    if 'No Finding' in labels:
        data_entry_df.at[index, 'No Finding'] = 1
    else:
        for label in labels:
            label = label.strip()
            data_entry_df.at[index, label] = 1

# Display the first few rows of the updated dataframe to confirm the changes
data_entry_df.head()
