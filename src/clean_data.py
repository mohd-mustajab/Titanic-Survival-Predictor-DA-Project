# src/clean_data.py
import pandas as pd

def clean_titanic_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Drop irrelevant columns
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')
    
    # Fill missing Age with median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Fill missing Embarked with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Fill missing Fare with median
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    return df

# Example usage
if __name__ == "__main__":
    df_cleaned = clean_titanic_data("data/titanic.csv")
    df_cleaned.to_csv("data/titanic_cleaned.csv", index=False)
    print("✅ Dataset cleaned and saved as titanic_cleaned.csv")
