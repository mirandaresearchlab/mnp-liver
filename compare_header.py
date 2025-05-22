import pandas as pd


def compare_and_show_different_headers(file1_path, file2_path, num_rows=10):
    try:
        # Read the CSV files
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)

        # Get headers
        headers1 = set(df1.columns)
        headers2 = set(df2.columns)

        # Find headers unique to each file
        headers_only_in_file1 = headers1 - headers2
        headers_only_in_file2 = headers2 - headers1

        # Print headers that are different
        print("Headers only in first file:", list(headers_only_in_file1))
        print("Headers only in second file:", list(headers_only_in_file2))

        # Print first ten rows of unique columns from first file
        if headers_only_in_file1:
            print(f"\nFirst {num_rows} rows of unique columns in first file:")
            unique_cols_df1 = df1[list(headers_only_in_file1)]
            print(unique_cols_df1.head(num_rows))
        else:
            print("\nNo unique columns in first file.")

        # Print first ten rows of unique columns from second file
        if headers_only_in_file2:
            print(f"\nFirst {num_rows} rows of unique columns in second file:")
            unique_cols_df2 = df2[list(headers_only_in_file2)]
            print(unique_cols_df2.head(num_rows))
        else:
            print("\nNo unique columns in second file.")

        # Optionally print common headers
        common_headers = headers1.intersection(headers2)
        print("\nCommon headers:", list(common_headers))

    except FileNotFoundError:
        print("Error: One or both files not found")
    except Exception as e:
        print(f"Error: {str(e)}")


# Example usage
file1_path = "/Users/jen-hung/Desktop/df_SingleCell_AO_HEPG2_110341.csv"  # Replace with your first CSV file path
file2_path = "/Users/jen-hung/Desktop/df_SingleCell_AO_HEPG2_231222.csv"  # Replace with your second CSV file path
compare_and_show_different_headers(file1_path, file2_path)
