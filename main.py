import pandas as pd
import numpy as np

def main():
    print("Hello from etsquality!")
    pd.set_option('display.max_columns', None)
    our_data = read_xl_skip_rows("our_data.xlsm", number_of_rows=5, range='C:DP')
    alex_data = read_xl_skip_rows("alex_data.xlsb", number_of_rows=4, range='A:DO')

    our_data.columns = our_data.columns.str.lower().str.replace(' ', '_')
    alex_data.columns = alex_data.columns.str.lower().str.replace(' ', '_')


    #print(our_data.head())
    print("\n", "-"*80, "\n")
    #print(alex_data.head())

    compare_datasets(our_data, alex_data)

    return 0

def read_xl_skip_rows(file_path, number_of_rows=5, range='A:DO'):
    df = pd.read_excel(file_path, skiprows=number_of_rows, usecols=f"{range}")
    return df

def compare_datasets(df_original, df_new):
    # Placeholder for comparing 
    id_col = 'oaw_id'

    df_original.sort_values(by=id_col, inplace=True)
    df_new.sort_values(by=id_col, inplace=True)

    # Compare dataframes col by col then row by row
    print("Original shape:", df_original.shape)
    print("New shape:", df_new.shape)

    origin_column = df_original.columns.tolist()
    new_column = df_new.columns.tolist()

    # compare column lists
    different_columns = list(set(origin_column) - set(new_column))

    print("Different Columns:", different_columns)

    #df_new.drop(columns='x.1', inplace=True)

    print((df_original == df_new).all(axis=1))


    pass

if __name__ == "__main__":
    main()
