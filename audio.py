import os
import pandas as pd
import re

def parse_textgrid(file_path):
    """
    Parse a TextGrid file to extract interval data.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Regular expressions to find intervals
    intervals = re.findall(r'intervals \[(\d+)\]:\s+xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*?)"', content)

    # Create a list of dictionaries for each interval
    data = []
    for interval in intervals:
        interval_data = {
            'index': int(interval[0]),
            'xmin': float(interval[1]),
            'xmax': float(interval[2]),
            'text': interval[3]
        }
        data.append(interval_data)

    return data

def adjust_intervals(data, chunk_start_time):
    """
    Adjust the xmin and xmax of each interval based on the chunk's start time.
    """
    for interval in data:
        interval['xmin'] += chunk_start_time
        interval['xmax'] += chunk_start_time
    return data

def process_textgrid_folder(folder_path, chunk_duration=40):
    """
    Process all TextGrid files in a folder, separating alignments and transcriptions,
    and assembling them while adjusting time offsets.
    """
    alignments = []
    transcriptions = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".TextGrid"):
            file_path = os.path.join(folder_path, filename)
            
            # Parse the TextGrid file
            data = parse_textgrid(file_path)
            
            # Extract chunk number from the filename
            match = re.search(r'chunk_(\d+)', filename)
            if match:
                chunk_number = int(match.group(1))
                chunk_start_time = chunk_number * chunk_duration

                # Adjust intervals to global time
                data = adjust_intervals(data, chunk_start_time)

                # Separate alignments and transcriptions
                if "alignment" in filename.lower():
                    alignments.extend(data)
                elif "transcription" in filename.lower():
                    transcriptions.extend(data)

    # Convert to DataFrames and sort
    alignments_df = pd.DataFrame(alignments).sort_values(by=['xmin', 'xmax']).reset_index(drop=True)
    transcriptions_df = pd.DataFrame(transcriptions).sort_values(by=['xmin', 'xmax']).reset_index(drop=True)

    return alignments_df, transcriptions_df

def save_dataframes(alignments_df, transcriptions_df, output_folder):
    """
    Save the alignment and transcription DataFrames to CSV files.
    """
    os.makedirs(output_folder, exist_ok=True)

    alignments_csv = os.path.join(output_folder, "all_alignments.csv")
    transcriptions_csv = os.path.join(output_folder, "all_transcriptions.csv")

    alignments_df.to_csv(alignments_csv, index=False)
    transcriptions_df.to_csv(transcriptions_csv, index=False)

    print(f"Alignments saved to: {alignments_csv}")
    print(f"Transcriptions saved to: {transcriptions_csv}")


def preprocess_dataframe(df):
    """
    Preprocess the DataFrame by sorting and reordering indexes.
    """
    # Sort by 'xmin' column
    #df = df.sort_values(by=['xmin', 'xmax']).reset_index(drop=True)
    # Reorder the index column sequentially
    df['index'] = range(1, len(df) + 1)

    return df