import pandas as pd
import numpy as np

def main():
    print("Hello from etsquality!")
    #pd.set_option('display.max_columns', None)
    alex_data = read_xl_skip_rows("alex_data.xlsb", number_of_rows=4, range='B:DN')
    our_data = read_xl_skip_rows("our_data.xlsm", number_of_rows=5, range='C:DO')

    alex_data.columns = our_data.columns.str.lower().str.replace(' ', '_')
    our_data.columns = alex_data.columns.str.lower().str.replace(' ', '_')

    compare_datasets(alex_data, our_data)

    return 0

def read_xl_skip_rows(file_path, number_of_rows=5, range='A:DO'):
    df = pd.read_excel(file_path, skiprows=number_of_rows, usecols=f"{range}")
    return df

def compare_datasets(df_original, df_new):
    # Placeholder for comparing 
    id_col = 'oaw_id'

    df_original.sort_values(by=id_col, inplace=True)
    df_new.sort_values(by=id_col, inplace=True)

    df_original.sort_index(axis=1, inplace=True)
    df_new.sort_index(axis=1, inplace=True)

    # Compare dataframes col by col then row by row
    print("Original shape:", df_original.shape)
    print("New shape:", df_new.shape)

    df_original.rename(columns={"x.1": "x"}, inplace=True)
    df_new.rename(columns={"x.1": "x"}, inplace=True)


    origin_columns = df_original.columns.tolist()
    new_columns = df_new.columns.tolist()

    print(origin_columns)
    print("\n", "-"*40, "\n")
    print(new_columns)

    # compare column lists
    different_columns = list(set(origin_columns) - set(new_columns))
    different_columns_other = list(set(new_columns) - set(origin_columns))

    print("Different Columns:", different_columns, different_columns_other)

    #df_new.drop(columns='x.1', inplace=True)

    print((df_original == df_new).all(axis=1))

    pass

if __name__ == "__main__":
    main()
