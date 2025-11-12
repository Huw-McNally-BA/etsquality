import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def main():
    print("Hello from etsquality!")
    #pd.set_option('display.max_columns', None)
    alex_data = read_xl_skip_rows("alex_v3.xlsb", number_of_rows=4, range='B:DN')
    our_data = read_xl_skip_rows("our_v3.xlsm", number_of_rows=5, range='C:DO')

    alex_data.columns = our_data.columns.str.lower().str.replace(' ', '_')
    our_data.columns = alex_data.columns.str.lower().str.replace(' ', '_')

    compare_datasets(alex_data, our_data)
    #check_cols(our_data)

    #numeric_descriptions = numeric_col_check(our_data, df_compare=alex_data, plot=False, plot_all=False)

    #numeric_col_plot_comparison(our_data, alex_data)
    #categoric_col_plot_comparison(our_data, alex_data)
    #print("\nNumeric column descriptions:\n", numeric_descriptions)

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
    numeric_flag_columns = [
        'planned_burn', 'expected_fuel_burn', 'max_tanks', 'subs_flight_no', 'master_fuel_burn', 'master_total_burn'
    ]
     
    # Define which columns should use numeric comparison with tolerance
    flag_columns = [
        'planned_burn', 'expected_fuel_burn', 'max_tanks',
        "ref",
        "x",
        "cs",
        "cs_applied",
        "operational_product_code",
        "d",
        "a",
        "d_icao",
        "a_icao",
        "route",
        "planned_route",
        "gcd+95",
        "eu_ets_(old)",
        "swiss",
        "corsia_(old)",
        "dom",
        "circular",
        "outer",
        "fly_time_min",
        "block_time_min",
        "subs_flytime_min",
        "subs_route",
        "aircraft_reg",
        "iata_type",
        "icao_type",
        "subs_oaw_id",
        "flight_no",
        "sfx",
        "subs_flight_no",
        "base_fuel",
        "master_total_burn",
        "x",
        "concat_to_cityflyer_raw",
        "subs_base_uplift_(where_applied)",
        "depart_fuel_source",
        "subs_depart_fuel_source",
        "subs_fuel_uplift_tonne"
    ]

    # Get the flag columns that exist in both dataframes
    common_flag_columns = list(set(flag_columns).intersection(set(origin_columns)).intersection(set(new_columns)))
    print("\nFlag columns found in both datasets:", common_flag_columns)

    # Compare flags
    if common_flag_columns:
        # Process each flag column to ensure Y/N consistency
        for col in common_flag_columns:
            # Step 1: Process values based on column type
            for df in [df_original, df_new]:
                if col in numeric_flag_columns:
                    # For numeric columns, just handle NaN consistently
                    df[col] = df[col].replace('nan', np.nan)
                    df[col] = df[col].replace('NaN', np.nan)
                else:
                    # For non-numeric columns: convert to string, strip whitespace, and normalize
                    df[col] = df[col].astype(str).str.strip().str.lower()
                    
                    # Convert NaN variants to N
                    df[col] = df[col].replace('nan', 'N')
                    df[col] = df[col].replace('NaN', 'N')
                    df[col] = df[col].replace(np.nan, 'N')

                    # Normalize various forms of true/false values to Y/N
                    value_map = {
                        '1': 'Y', '1.0': 'Y', 'yes': 'Y', 'true': 'Y', 'True': 'Y',
                        '0': 'N', '0.0': 'N', 'no': 'N', 'false': 'N', 'False': 'N',
                        '': 'N', ' ': 'N'  # Empty strings and spaces become N
                    }
                    df[col] = df[col].map(lambda x: value_map.get(str(x).strip(), x))
                    
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
            
            if col in numeric_flag_columns:
                # For numeric columns, convert to float and use isclose
                orig_vals = pd.to_numeric(flag_comparison[orig_col], errors='coerce')
                new_vals = pd.to_numeric(flag_comparison[new_col], errors='coerce')
                
                # Find mismatches using numpy's isclose with 0.1 tolerance
                matches = np.isclose(orig_vals, new_vals, rtol=0.1, atol=0.1, equal_nan=True)
                mismatches = flag_comparison[~matches]
            else:
                # For non-numeric columns, use exact comparison
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

def check_cols(df: pd.DataFrame):
    numeric_cols = 0
    string_cols = 0
    bool_cols = 0
    other_cols = 0

    for col, dtype in df.dtypes.items():
        if pd.api.types.is_numeric_dtype(dtype):
            numeric_cols += 1
        elif pd.api.types.is_string_dtype(dtype):
            string_cols += 1
        elif pd.api.types.is_bool_dtype(dtype):
            bool_cols += 1
        else:
            print(f"{col} has dtype {dtype}. Custom handling may be needed.")
            other_cols += 1

    print(f"\nSummary of column types:"
          f"\nNumeric columns: {numeric_cols}"
          f"\nString columns: {string_cols}"
          f"\nBoolean columns: {bool_cols}"
          f"\nOther columns: {other_cols}")

def numeric_col_check(df: pd.DataFrame, df_compare: pd.DataFrame = None, plot: bool = False, plot_all: bool = False, cols_to_plot: list[str] = None):
    """Describe numeric columns and optionally plot distributions.

    If `df_compare` is provided, scatter plots of original vs compare values
    (aligned by `oaw_id` when present) will also be produced.
    """
    numeric_cols = [col for col, dtype in df.dtypes.items() if pd.api.types.is_numeric_dtype(dtype)]

    # If a compare dataframe is provided, restrict to the intersection of
    # numeric columns present in both dataframes so plots align
    if df_compare is not None:
        compare_numeric_cols = [c for c, d in df_compare.dtypes.items() if pd.api.types.is_numeric_dtype(d)]
        numeric_cols = [c for c in numeric_cols if c in compare_numeric_cols]

    df_numeric = df[numeric_cols]

    descriptions = df_numeric.describe()

    # Prepare merged frame for comparisons if requested
    merged = None
    id_col = 'oaw_id'
    if df_compare is not None and id_col in df.columns and id_col in df_compare.columns:
        try:
            # Defensive: if either dataframe has duplicate column labels, try to fix
            for name, df_tmp in [('df', df), ('df_compare', df_compare)]:
                dupes = df_tmp.columns[df_tmp.columns.duplicated()].unique()
                if len(dupes) > 0:
                    print(f"Warning: duplicate column labels in {name} dataframe: {list(dupes)}")
                    # If the id_col is duplicated, combine duplicates by taking the
                    # first non-null value across duplicates, then drop extras.
                    if id_col in dupes:
                        cols = [c for c in df_tmp.columns if c == id_col]
                        df_tmp[id_col] = df_tmp[cols].bfill(axis=1).iloc[:, 0]
                        df_tmp.drop(columns=cols[1:], inplace=True)
                    # Drop any remaining duplicate columns, keeping the first occurrence
                    df_tmp = df_tmp.loc[:, ~df_tmp.columns.duplicated()]
                    if name == 'df':
                        df = df_tmp
                    else:
                        df_compare = df_tmp

            merged = pd.merge(
                df[[id_col] + numeric_cols],
                df_compare[[id_col] + numeric_cols],
                on=id_col,
                how='inner',
                suffixes=("_orig", "_cmp")
            )
        except Exception as e:
            print(f"Warning: could not create merged comparison frame: {e}")

    if plot:
        if plot_all:
            # Save all numeric column distributions (and optional comparison
            # scatter plots) into a single multi-page PDF
            pdf_path = "numeric_distributions.pdf"
            with PdfPages(pdf_path) as pdf:
                # Histograms for the original dataframe
                for col in numeric_cols:
                    fig, ax = plt.subplots()
                    sns.histplot(df_numeric[col].dropna(), kde=True, ax=ax)
                    ax.set_title(f'Distribution of {col} (original)')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

                # If comparison dataframe provided and merged frame exists,
                # produce scatter plots aligned by id
                if merged is not None:
                    for col in numeric_cols:
                        fig, ax = plt.subplots()
                        x = merged[f"{col}_orig"].to_numpy()
                        y = merged[f"{col}_cmp"].to_numpy()
                        ax.scatter(x, y, alpha=0.6, s=10)
                        # Add 1:1 line for reference
                        try:
                            mn = min(np.nanmin(x), np.nanmin(y))
                            mx = max(np.nanmax(x), np.nanmax(y))
                            ax.plot([mn, mx], [mn, mx], color='red', linestyle='--')
                        except Exception:
                            pass
                        ax.set_title(f'Comparison of {col}: original vs compare')
                        ax.set_xlabel('original')
                        ax.set_ylabel('compare')
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)

            print(f"Saved all numeric distribution and comparison plots to {pdf_path}")
        else:
            if cols_to_plot is None:
                cols_to_plot = numeric_cols[:5]  # Default to first 5 numeric columns
            for col in cols_to_plot:
                if col in numeric_cols:
                    plt.figure()
                    sns.histplot(df_numeric[col].dropna(), kde=True)
                    plt.title(f'Distribution of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    plt.show()

                    # If compare_df provided and merged exists, show a scatter for this column
                    if merged is not None:
                        plt.figure()
                        plt.scatter(merged[f"{col}_orig"], merged[f"{col}_cmp"], alpha=0.6, s=10)
                        try:
                            mn = min(merged[f"{col}_orig"].min(), merged[f"{col}_cmp"].min())
                            mx = max(merged[f"{col}_orig"].max(), merged[f"{col}_cmp"].max())
                            plt.plot([mn, mx], [mn, mx], color='red', linestyle='--')
                        except Exception:
                            pass
                        plt.title(f'Comparison of {col}: original vs compare')
                        plt.xlabel('original')
                        plt.ylabel('compare')
                        plt.show()

    return descriptions

def numeric_col_plot_comparison(df_new: pd.DataFrame, df_original: pd.DataFrame):
    numeric_cols = [col for col, dtype in df_new.dtypes.items() if pd.api.types.is_numeric_dtype(dtype)]

    pdf_path = "numeric_distributions_comparison.pdf"
    with PdfPages(pdf_path) as pdf:
        # Helper to coerce common non-numeric placeholders to NaN and convert to numeric
        def clean_numeric_series(s: pd.Series) -> pd.Series:
            if s is None:
                return s
            # Convert to string, lowercase and strip to detect placeholders
            s_str = s.astype(str).str.strip()
            toks = ['no data', 'no_data', 'n/a', 'na', '-', 'none', 'n\a']
            s_str_lower = s_str.str.lower()
            mask = s_str_lower.isin(toks)
            s_str.loc[mask] = np.nan
            return pd.to_numeric(s_str, errors='coerce')

        for col in numeric_cols:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            try:
                orig_series = clean_numeric_series(df_original[col])
                new_series = clean_numeric_series(df_new[col])

                plotted = False
                if orig_series.dropna().size > 0:
                    sns.histplot(orig_series.dropna(), kde=True, color='skyblue', ax=axes[0])
                    plotted = True
                axes[0].set_title(f'Original: {col}')
                axes[0].set_xlabel(col)
                axes[0].set_ylabel('Frequency')

                if new_series.dropna().size > 0:
                    sns.histplot(new_series.dropna(), kde=True, color='salmon', ax=axes[1])
                    plotted = True
                axes[1].set_title(f'New: {col}')
                axes[1].set_xlabel(col)
                axes[1].set_ylabel('')

                # If neither side had numeric data, skip saving this page
                if not plotted:
                    plt.close(fig)
                    continue

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
            except Exception as e:
                plt.close('all')
                print(f"Warning: skipping numeric plot for {col} due to error: {e}")
                continue

    print(f"Saved side-by-side comparison plots to {pdf_path}")

def categoric_col_plot_comparison(df_new: pd.DataFrame, df_original: pd.DataFrame):
    # Select only non-numeric columns
    categoric_cols = [col for col in df_new.columns if not pd.api.types.is_numeric_dtype(df_new[col])]

    pdf_path = "categorical_distributions_comparison.pdf"
    with PdfPages(pdf_path) as pdf:
        for col in categoric_cols:
            if col not in df_original.columns:
                continue  # skip if missing in one df

            fig, ax = plt.subplots(figsize=(8, 5))

            # Value counts normalized to proportions
            orig_counts = df_original[col].value_counts(normalize=True)
            new_counts = df_new[col].value_counts(normalize=True)

            # Combine for consistent categories
            all_categories = sorted(set(orig_counts.index).union(new_counts.index))
            orig_counts = orig_counts.reindex(all_categories, fill_value=0)
            new_counts = new_counts.reindex(all_categories, fill_value=0)

            # Plot bars
            x = range(len(all_categories))
            ax.bar(x, orig_counts, width=0.4, label="Original", align='center', alpha=0.7)
            ax.bar([i + 0.4 for i in x], new_counts, width=0.4, label="New", align='center', alpha=0.7)

            ax.set_xticks([i + 0.2 for i in x])
            ax.set_xticklabels(all_categories, rotation=45, ha='right')
            ax.set_title(f'Category Distribution Comparison: {col}')
            ax.set_ylabel('Proportion')
            ax.legend()
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved categorical comparison plots to {pdf_path}")

if __name__ == "__main__":
    main()
