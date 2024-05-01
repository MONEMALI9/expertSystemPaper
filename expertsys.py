import pandas as pd

# Step 1: Knowledge Base
def read_data(file_path):
    """
    Read data from CSV file and extract age and minutes played columns for the first 15 observations.
    """
    data = pd.read_csv(file_path)
    # Extracting age and minutes played for the first 15 observations
    subset_data = data[['name', 'Age', 'Min']].head(15)
    return subset_data

# Step 2: Facts
# Facts will be derived from the knowledge base

# Step 3: Rules
# For illustration purposes, let's define a simple rule:
# If age is less than 25 and minutes played is greater than 1000, then the player is considered a key player.
def is_key_player(age, minutes_played):
    if age < 25 and minutes_played > 90:
        return True
    else:
        return False

# Step 4: Inference Engine
def infer(data):
    conclusions = []
    for index, row in data.iterrows():
        player = row['name']
        age = row['Age']
        minutes_played = row['Min']
        if is_key_player(age, minutes_played):
            conclusions.append(f"{player} is a key player.")
    return conclusions

def classify_players(row):
    if row['Age'] < 25 and row['Min'] > 50:
        return "Promising"
    elif row['Age'] >= 30 and row['Min'] < 91:
        return "Less Impactful"
    elif 91 < row['Min'] < 50:
        return "Regular Performer"
    else:
        return "Unclassified"


'''
# Main function to execute the expert system
def main():
    file_path = "playerstats.csv"
    data = read_data(file_path)
    conclusions = infer(data)
    for conclusion in conclusions:
        print(conclusion)

    # Apply the classification function to each row in the knowledge base
    knowledge_base['Classification'] = knowledge_base.apply(classify_players, axis=1)

    # Display the updated knowledge base with classifications
    print(knowledge_base[['name', 'Age', 'Min' , 'Classification']])



if __name__ == "__main__":
    main()
'''
