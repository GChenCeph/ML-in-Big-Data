import pandas as pd

# Define state codes and month codes
state_codes = {
    "Alabama": 1, "Arizona": 2, "Arkansas": 3, "California": 4, "Colorado": 5,
    "Connecticut": 6, "Delaware": 7, "Florida": 8, "Georgia": 9, "Idaho": 10,
    "Illinois": 11, "Indiana": 12, "Iowa": 13, "Kansas": 14, "Kentucky": 15,
    "Louisiana": 16, "Maine": 17, "Maryland": 18, "Massachusetts": 19, "Michigan": 20,
    "Minnesota": 21, "Mississippi": 22, "Missouri": 23, "Montana": 24, "Nebraska": 25,
    "Nevada": 26, "New Hampshire": 27, "New Jersey": 28, "New Mexico": 29, "New York": 30,
    "North Carolina": 31, "North Dakota": 32, "Ohio": 33, "Oklahoma": 34, "Oregon": 35,
    "Pennsylvania": 36, "Rhode Island": 37, "South Carolina": 38, "South Dakota": 39,
    "Tennessee": 40, "Texas": 41, "Utah": 42, "Vermont": 43, "Virginia": 44,
    "Washington": 45, "West Virginia": 46, "Wisconsin": 47, "Wyoming": 48, "Alaska": 50
}

month_codes = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, 
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}

# Function to categorize fire count
def categorize_fire_count(fire_count):
    if fire_count <= 3:
        return 1
    elif fire_count <= 29:
        return 2
    elif fire_count <= 141:
        return 3
    elif fire_count <= 348:
        return 4
    else:
        return 5

# Load the wildfire data
wilds = pd.read_csv('data.csv')

# Preprocess to create a summary of fire counts per state and year for each month
fire_summary = wilds.groupby(['state', 'year', 'month']).size().reset_index(name='count')
fire_summary_dict = {(row['state'], row['year'], row['month']): row['count'] for _, row in fire_summary.iterrows()}

# File paths of datasets to be processed
file_paths = [
    'climdiv-pcpnst-v1.0.0-20231106.txt', 'climdiv-pdsist-v1.0.0-20231106.txt',
    'climdiv-phdist-v1.0.0-20231106.txt', 'climdiv-sp01st-v1.0.0-20231106.txt',
    'climdiv-sp02st-v1.0.0-20231106.txt', 'climdiv-sp06st-v1.0.0-20231106.txt',
    'climdiv-sp12st-v1.0.0-20231106.txt', 'climdiv-sp24st-v1.0.0-20231106.txt',
    'climdiv-tmaxst-v1.0.0-20231106.txt', 'climdiv-tminst-v1.0.0-20231106.txt',
    'climdiv-tmpcst-v1.0.0-20231106.txt', 'climdiv-zndxst-v1.0.0-20231106.txt'
]

# Initialize DataFrame to store the combined data
combined_df = None

# Process each file and merge into combined_df
for file_path in file_paths:
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            words = line.split()
            state_id = int(words[0][:3])
            year_val = int(words[0][-4:])
            monthly_data = [float(word) for word in words[1:13]]
            data.append([state_id, year_val] + monthly_data)

    name = file_path.split('/')[-1].split('-')[1]
    columns = ['state', 'year'] + [f'{name}_m{i}' for i in range(1, 13)]
    df = pd.DataFrame(data, columns=columns)

    if combined_df is None:
        combined_df = df
    else:
        combined_df = pd.merge(combined_df, df, on=['state', 'year'])

# Categorize fire counts for each month
for month in range(1, 13):
    combined_df[f'f{month}'] = combined_df.apply(
        lambda row: categorize_fire_count(
            fire_summary_dict.get((row['state'], row['year'], month), 0)
        ), axis=1
    )

# Split the dataset into training, reference, and testing datasets
training_df = combined_df[(combined_df['year'] >= 1992) & (combined_df['year'] <= 2013) & (combined_df['state'] < 49)]
reference = combined_df[(combined_df['year'] >= 2014) & (combined_df['year'] <= 2015) & (combined_df['state'] < 49)]
testing_df = reference.copy()

# Set fire counts to 0 in the testing dataset
for i in range(1, 13):
    testing_df[f'f{i}'] = 0

# Save the datasets
training_df.to_csv('training.csv', index=False)
reference.to_csv('reference.csv', index=False)
testing_df.to_csv('testing.csv', index=False)

print("Datasets saved successfully.")
