import pandas as pd
import os

def main():
    # --- CONFIGURATION ---
    CSV_FILE = "../sign_dictionary_more.csv"
    #CSV_FILE = "../sign_dictionary_fitted.csv"
    # ---------------------

    print(f"Loading {CSV_FILE}...")
    
    try:
        # Load the CSV
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find '{CSV_FILE}'. Check the path!")
        return

    # Check if the 'category' column actually exists
    if 'category' not in df.columns:
        print("Error: 'category' column not found in the CSV.")
        return

    # Count the categories and drop any empty/NaN ones just in case
    # .value_counts() automatically sorts from highest to lowest!
    category_counts = df['category'].value_counts()

    # Print a nice formatted table
    print("\nCATEGORY DISTRIBUTION:")
    print("-" * 45)
    print(f"{'CATEGORY':<35} | {'COUNT'}")
    print("-" * 45)

    for category, count in category_counts.items():
        # Strip any accidental hidden spaces from the category name when printing
        category_clean = str(category).strip()
        print(f"{category_clean:<35} | {count}")

    print("-" * 45)
    print(f"TOTAL SIGNS: {len(df)}")
    print(f"UNIQUE CATEGORIES: {len(category_counts)}")

if __name__ == "__main__":
    main()