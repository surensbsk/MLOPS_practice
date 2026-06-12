import pandas as pd


def preprocess_data(df):
    # Handle missing values
    df = df.dropna()

    # Encode categorical variables
    df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

    # Scale numerical features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[['Age', 'EstimatedSalary']] = scaler.fit_transform(df[['Age', 'EstimatedSalary']])

    return df

def split_data(df, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    X = df.drop('Purchased', axis=1)
    y = df['Purchased']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def load_and_preprocess_data(file_path):
    import pandas as pd
    df = pd.read_csv(file_path)
    df = preprocess_data(df)
    return df

# Example usage:
# df = load_and_preprocess_data('data/raw/Social_Network_Ads.csv')
# X_train, X_test, y_train, y_test = split_data(df)