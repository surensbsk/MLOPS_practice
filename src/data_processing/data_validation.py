def validate_data(df):
    # Check for missing values
    if df.isnull().values.any():
        raise ValueError("Data contains missing values.")
    
    # Check for correct data types
    expected_types = {
        'User ID': 'int64',
        'Gender': 'object',
        'Age': 'int64',
        'EstimatedSalary': 'float64',
        'Purchased': 'int64'
    }
    
    for column, expected_type in expected_types.items():
        if df[column].dtype != expected_type:
            raise TypeError(f"Column '{column}' does not have the expected type '{expected_type}'.")

    # Additional integrity checks can be added here
    return True

def check_data_integrity(df):
    # Example integrity check: Ensure 'Purchased' column only contains 0 or 1
    if not df['Purchased'].isin([0, 1]).all():
        raise ValueError("Invalid values found in 'Purchased' column. Only 0 or 1 are allowed.")
    
    return True

def validate_and_process_data(df):
    validate_data(df)
    check_data_integrity(df)
    return df