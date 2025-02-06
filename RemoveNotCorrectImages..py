import os
import pandas as pd
def delete_images_with_patterns(directory: str, patterns: list):
    """
    Deletes image files in the given directory if their filenames contain any of the specified patterns.

    Args:
        directory (str): The path to the directory containing images.
        patterns (list): A list of substrings to check in filenames.
    """
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Check if the filename contains any of the specified patterns
        if any("dr"+pattern in filename for pattern in patterns):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def clean_csv(csv_path: str, patterns: list):
    """
    Removes rows from the CSV if the first column contains filenames matching any pattern (e.g., "1_1" -> "dr1_1").
    Ensures that there are no additional digits after the pattern unless separated by an underscore `_`.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file '{csv_path}' does not exist.")
        return
    
    # Load CSV into a DataFrame
    df = pd.read_csv(csv_path)

    # Ensure the first column is treated as a string
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)

    # Create modified patterns to match filenames
    modified_patterns = [f"dr{p}" for p in patterns]

    # Build a regex pattern to match filenames exactly or with an underscore and additional digits
    regex_patterns = []
    for pattern in modified_patterns:
        # Match the pattern exactly or with an underscore and additional digits
        regex_patterns.append(f"^{pattern}(_\\d+)?$")

    # Combine all regex patterns into a single pattern
    combined_regex = '|'.join(regex_patterns)

    # Filter out rows where the first column matches any of the regex patterns
    df = df[~df.iloc[:, 0].str.match(combined_regex, na=False)]

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Save cleaned data back to CSV
    df.to_csv(csv_path, index=False)
    print(f"Updated CSV saved: {csv_path}")

# List of text patterns to match in filenames
patterns_to_delete = [
    "1_1", 
    "4_1", 
    "4_2", 
    "4_3", 
    "4_4", 
    "4_5", 
    "4_6",
    "5_1", 
    "5_2", 
    "7_1", 
    "10_1", 
    "24_1", 
    "24_2", 
    "25_1", 
    "25_2", 
    "29_1", 
    "30_1", 
    "33_1", 
    "36_1", 
    "36_4", 
    "36_5", 
    "36_6", 
    "38_1", 
    "38_2", 
    "38_3", 
    "38_4", 
    "38_5", 
    "38_6", 
    "38_7", 
    "38_8", 
    "38_9", 
    "42_1", 
    "42_2", 
    "42_4", 
    "43_1", 
    "43_2", 
    "43_3", 
    "43_4", 
    "43_5", 
    "44_1", 
    "44_2", 
    "44_3", 
    "44_4", 
    "44_6", 
    "45_1", 
    "47_1", 
    "50_1", 
    "57_1", 
    "57_2", 
    "63_1", 
    "64_1", 
    "64_2", 
    "64_3", 
    "64_4", 
    "64_5", 
    "64_6", 
    "64_7", 
    "64_8", 
    "64_9", 
    "65_1", 
    "65_2", 
    "66_1", 
    "66_2", 
    "66_3", 
    "66_4", 
    "66_5", 
    "66_6", 
    "66_7", 
    "66_8", 
    "69_1", 
    "69_2", 
    "69_3", 
    "69_4", 
    "69_5", 
    "69_6", 
    "69_7", 
    "69_8", 
    "69_9", 
    "71_1", 
    "71_2", 
    "71_3", 
    "71_4", 
    "71_5", 
    "73_1", 
    "74_1", 
    "75_1", 
    "75_2", 
    "75_3", 
    "75_4", 
    "75_5", 
    "75_6", 
    "77_1", 
    "77_2", 
    "77_3", 
    "76_1", 
    "76_2", 
    "76_3", 
    "76_4", 
    "76_5",
    "80_1", 
    "80_2", 
    "82_1", 
    "86_1", 
    "86_2", 
    "86_3", 
    "86_4", 
    "86_5",
    "87_1", 
    "87_2", 
    "87_3", 
    "87_4", 
    "87_5", 
    "87_6",
    "89_1", 
    "92_1", 
    "92_2", 
    "93_1", 
    "94_2", 
    "94_1", 
    "95_1",
    "97_1", 
    "97_2",
    "102_1",
    "104_1",
    "108_1",
    "109_1",
    "112_1",
    "114_1",
    "114_2",
    "114_3",
    "114_4",
    "114_5",
    "114_6",
    "114_7",
    "114_8",
    "114_9",
    "115_1",
    "115_2",
    "116_1",
    "116_2",
    "116_3",
    "117_1",
    "128_1",
    "130_1",
    "132_1",
    "132_2",
    "132_3",
    "137_1",
    "137_2",
    "137_3",
    "137_4",
    "137_5",
    "137_6",
    "137_7",
    "137_8",
    "137_9",
    "140_5",
    "146_1",
    "146_2",
    "146_3",
    "151_1",
    "151_2",
    "163_1",
    "169_1",
    "173_1",
    "173_2",
    "100_1"
]

# Specify your target directory
target_directory = "./cropped_images"  # Change this to your actual directory

# Run the deletion function
# delete_images_with_patterns(target_directory, patterns_to_delete)
patterns_to_delete = [
    "dr80_2",
    "dr80_3",
    "dr81_1",
    "dr81_1",
    "dr81_2",
    "dr83_1",
    "dr86_1",
    "dr86_2",
    "dr86_3",
    "dr86_4",
    "dr86_5",
    "dr87_1",
    "dr87_2",
    "dr87_3",
    "dr87_4",
    "dr87_5",
    "dr87_6",
    "dr88_1",
    "dr89_1",
    "dr89_2",
    "dr9_1",
    "dr90_1",
    "dr92_1",
    "dr92_1",
    "dr92_2",
    "dr92_3",
    "dr93_1",
    "dr93_2",
    "dr94_1",
    "dr94_2",
    "dr94_3",
    "dr95_1",
    "dr95_2",
    "dr96_1",
    "dr97_1",
    "dr97_2",
    "dr97_3",
    "dr98_1",
]
clean_csv("all_cropped_data.csv",patterns=patterns_to_delete)