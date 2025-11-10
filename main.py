import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
    """Compare two dataframes and report numeric differences per id.

    Inputs:
      - df_original: original dataframe
      - df_new: new dataframe to compare

    Behavior:
      - Aligns both frames on `id_col`, finds numeric columns present in both,
        compares numeric values per id using an isclose tolerance, and prints a
        brief summary plus a sample of differences.
    """
    id_col = 'oaw_id'

    # Ensure a stable order
    df_original.sort_values(by=id_col, inplace=True)
    df_new.sort_values(by=id_col, inplace=True)

    df_original.sort_index(axis=1, inplace=True)
    df_new.sort_index(axis=1, inplace=True)

    # Basic shapes
    print("Original shape:", df_original.shape)
    print("New shape:", df_new.shape)

    # Normalise an odd column name if present
    df_original.rename(columns={"x.1": "x"}, inplace=True)
    df_new.rename(columns={"x.1": "x"}, inplace=True)

    origin_columns = df_original.columns.tolist()
    new_columns = df_new.columns.tolist()

    #print(origin_columns)
    print("\n", "-"*40, "\n")
    #print(new_columns)

    # compare column lists
    different_columns = list(set(origin_columns) - set(new_columns))
    different_columns_other = list(set(new_columns) - set(origin_columns))

    print("Different Columns:", different_columns, different_columns_other)

    # Define the flag columns we want to compare
    flag_columns = [
        'gcd+95', 'eu_ets_(old)', 'swiss', 'corsia_(old)', 'dom', 'circular', 
        'outer', 'roue_plan_vs_actual', 'domestic_offset_flag-uk', 'uk_ets_flag', 
        'eu_ets_flag', 'swiss_ets_flag', 'eu_outer_regions_ets_flag', 'combined_ets_flag'
    ]

    # Get the flag columns that exist in both dataframes
    common_flag_columns = list(set(flag_columns).intersection(set(origin_columns)).intersection(set(new_columns)))
    print("\nFlag columns found in both datasets:", common_flag_columns)

    # Compare flags
    if common_flag_columns:
        # Define which columns should use '0' vs 'N' for blanks
        zero_columns = ['gcd+95', 'dom', 'circular', 'outer', 'roue_plan_vs_actual']
        n_columns = ['eu_ets_(old)', 'swiss', 'corsia_(old)', 'domestic_offset_flag-uk', 
                    'uk_ets_flag', 'eu_ets_flag', 'swiss_ets_flag', 
                    'eu_outer_regions_ets_flag', 'combined_ets_flag']

        # Process each flag column to ensure Y/N consistency
        for col in common_flag_columns:
            # Step 1: Convert to strings and handle NaN/empty values
            for df in [df_original, df_new]:
                # Convert to string but preserve NaN
                df[col] = df[col].astype(str).replace('nan', 'N')
                df[col] = df[col].astype(str).replace('NaN', 'N')
                df[col] = df[col].astype(str).replace(np.nan, 'N')

                # Normalize various forms of true/false values to Y/N
                value_map = {
                    '1': 'Y', '1.0': 'Y', 'yes': 'Y', 'true': 'Y', 'True': 'Y',
                    '0': 'N', '0.0': 'N', 'no': 'N', 'false': 'N', 'False': 'N',
                    '': 'N', ' ': 'N'  # Empty strings and spaces become N
                }
                df[col] = df[col].map(lambda x: value_map.get(str(x), x))
                
                # Fill any remaining NaN with N
                df[col] = df[col].fillna('N')
            
            # Validation: check for unexpected values
            for df, name in [(df_original, 'original'), (df_new, 'new')]:
                invalid_vals = df[col][~df[col].isin(['Y', 'N'])]
                if not invalid_vals.empty:
                    print(f"\nWarning: Found invalid values in {name} dataset, column {col}:")
                    print(invalid_vals.unique())

        # Merge the dataframes for flag comparison
        flag_comparison = pd.merge(
            df_original[[id_col] + common_flag_columns],
            df_new[[id_col] + common_flag_columns],
            on=id_col,
            how='outer',
            suffixes=('_orig', '_new')
        )

        # Create a list to store flag differences
        flag_diffs = []

        # Compare each flag column
        for col in common_flag_columns:
            orig_col = f"{col}_orig"
            new_col = f"{col}_new"
            
            # Compare values
            mismatches = flag_comparison[flag_comparison[orig_col] != flag_comparison[new_col]]
            
            for _, row in mismatches.iterrows():
                flag_diffs.append({
                    'id': row[id_col],
                    'flag': col,
                    'original': row[orig_col],
                    'new': row[new_col]
                })

        # Create DataFrame of flag differences
        flag_diffs_df = pd.DataFrame(flag_diffs)
        
        if flag_diffs_df.empty:
            print("\nNo flag differences found!")
        else:
            # Clean up any remaining NaN values in the differences DataFrame
            flag_diffs_df['original'] = flag_diffs_df['original'].fillna('N')
            flag_diffs_df['new'] = flag_diffs_df['new'].fillna('N')
            
            # Remove rows where values actually match after NaN cleanup
            flag_diffs_df = flag_diffs_df[flag_diffs_df['original'] != flag_diffs_df['new']]
            
            # Summary of flag differences
            total_flag_diffs = len(flag_diffs_df)
            by_flag = flag_diffs_df.groupby('flag').size().to_dict()
            print(f"\nFound {total_flag_diffs} flag differences across ids.")
            print("Differences by flag:", by_flag)
            
            # Show sample of differences
            print("\nSample flag differences:")

            # Attach both original and new flag values for each id
            try:
                # Get and rename original flags
                orig_flags = df_original[[id_col, 'cs_applied'] + common_flag_columns].copy()
                orig_flags.rename(columns={c: f"orig_{c}" for c in common_flag_columns}, inplace=True)
                
                # Get and rename new flags
                new_flags = df_new[[id_col, 'cs_applied'] + common_flag_columns].copy()
                new_flags.rename(columns={c: f"new_{c}" for c in common_flag_columns}, inplace=True)
                
                # Merge original flags
                flag_diffs_df = flag_diffs_df.merge(orig_flags, left_on='id', right_on=id_col, how='left')
                # Merge new flags
                flag_diffs_df = flag_diffs_df.merge(new_flags, left_on='id', right_on=id_col, how='left')
                
                # Drop any duplicate id columns from the merges
                id_cols = [col for col in flag_diffs_df.columns if col == id_col]
                if len(id_cols) > 1:  # If we have duplicate id columns
                    flag_diffs_df = flag_diffs_df.loc[:,~flag_diffs_df.columns.duplicated()]
            except Exception as e:
                # If merge fails for any reason, continue without the additional flags
                print(f"Warning: could not attach all flag columns to differences dataset: {str(e)}")

            # Get type of each individual value (for debugging)
            print(flag_diffs_df.head(20).to_string(index=False))
            flag_diffs_df.to_csv("flag_differences.csv", index=False)

            # Visualize flag differences
            plt.figure(figsize=(15, 5))
            
            # Create bar plot of flag differences
            flag_counts = flag_diffs_df['flag'].value_counts().reset_index()
            flag_counts.columns = ['flag', 'count']
            ax = sns.barplot(data=flag_counts, x='flag', y='count')
            
            # Add value labels on top of bars
            for i in ax.containers:
                ax.bar_label(i)
                
            plt.xticks(rotation=45, ha='right')
            plt.title('Number of Differences by Flag')
            plt.xlabel('Flag')
            plt.ylabel('Count of Differences')
            plt.tight_layout()
            plt.show()

    # Quick boolean per-row equality check (object-wise)
    try:
        print("\nRows with all values matching:", (df_original == df_new).all(axis=1).sum())
    except Exception:
        # If shapes/columns differ this may raise; ignore for now
        pass

    # # Identify numeric columns present in both dataframes (excluding the id_col)
    # common_columns = list(set(origin_columns).intersection(set(new_columns)))
    # if id_col in common_columns:
    #     common_columns.remove(id_col)

    # # Determine numeric columns from the intersection
    # numeric_cols = []
    # for col in common_columns:
    #     if pd.api.types.is_numeric_dtype(df_original[col]) and pd.api.types.is_numeric_dtype(df_new[col]):
    #         numeric_cols.append(col)

    # print("Numeric columns to compare:", numeric_cols)

    # if not numeric_cols:
    #     print("No numeric columns found to compare.")
    #     return

    # Merge the two dataframes on the id column to align rows
    # merged = pd.merge(df_original[[id_col] + numeric_cols],
    #                   df_new[[id_col] + numeric_cols],
    #                   on=id_col,
    #                   how='outer',
    #                   suffixes=("_orig", "_new"))

    # # Build a long-form differences list
    # diffs = []
    # # tolerance for numeric equality
    # rtol = 1e-5
    # atol = 1e-8

    # for col in numeric_cols:
    #     orig_col = f"{col}_orig"
    #     new_col = f"{col}_new"

    #     # If the merged frame lacks either column (shouldn't happen) skip
    #     if orig_col not in merged.columns or new_col not in merged.columns:
    #         continue

    #     orig_vals = merged[orig_col].to_numpy()
    #     new_vals = merged[new_col].to_numpy()

    #     # Compare elementwise, treating NaNs specially: NaN vs NaN considered equal
    #     for oid, oval, nval in zip(merged[id_col].to_numpy(), orig_vals, new_vals):
    #         both_nan = pd.isna(oval) and pd.isna(nval)
    #         if both_nan:
    #             continue

    #         # If either is NaN and the other is not, that's a difference
    #         if pd.isna(oval) != pd.isna(nval):
    #             diffs.append({"id": oid, "column": col, "orig": oval, "new": nval, "diff": None})
    #             continue

    #         # Both not NaN: numeric compare
    #         try:
    #             are_close = np.isclose(oval, nval, rtol=rtol, atol=atol)
    #         except Exception:
    #             # If comparison fails for any reason, record as difference
    #             are_close = False

    #         if not are_close:
    #             diffs.append({"id": oid, "column": col, "orig": oval, "new": nval, "diff": (nval - oval)})

    # diffs_df = pd.DataFrame(diffs)

    # if diffs_df.empty:
    #     print("No numeric differences found for any id and numeric column.")
    # else:
    #     # Summary: counts per column and total
    #     total_diffs = len(diffs_df)
    #     by_column = diffs_df.groupby('column').size().to_dict()
    #     print(f"Found {total_diffs} numeric differences across ids.")
    #     print("Differences by column:", by_column)
    #     # Show a sample of differences (up to 20 rows)
    #     print("Sample differences:\n", diffs_df.head(20).to_string(index=False))

    #     # Create visualizations of the differences
    #     plt.figure(figsize=(15, 10))
        
    #     # 1. Create a bar plot of difference counts by column
    #     diff_counts = diffs_df['column'].value_counts().reset_index()
    #     diff_counts.columns = ['column', 'count']
    #     plt.subplot(2, 1, 1)
    #     ax = sns.barplot(data=diff_counts, x='column', y='count')
    #     # Add value labels on top of bars
    #     for i in ax.containers:
    #         ax.bar_label(i)
    #     plt.xticks(rotation=45)
    #     plt.title('Number of Differences by Column')
    #     plt.xlabel('Column')
    #     plt.ylabel('Count of Differences')
        
    #     # 2. Create boxplots of difference distributions
    #     plt.subplot(2, 1, 2)
    #     # Filter out None values from diff column
    #     plot_diffs = diffs_df[diffs_df['diff'].notna()]
    #     if not plot_diffs.empty:
    #         sns.boxplot(data=plot_diffs, x='column', y='diff')
    #         plt.xticks(rotation=45)
    #         plt.title('Distribution of Differences by Column')
    #         plt.xlabel('Column')
    #         plt.ylabel('Difference (new - original)')
        
    #     plt.tight_layout()
    #     plt.show()

if __name__ == "__main__":
    main()
