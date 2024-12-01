import os
import pandas as pd
import re
import textgrid
import csv

def convert_textgrid_to_csv(textgrid_file, output_csv):
    """
    Converts a TextGrid file to a CSV file.

    Parameters:
    textgrid_file (str): Path to the input TextGrid file.
    output_csv (str): Path to the output CSV file where the data will be saved.
    """
    tg = textgrid.TextGrid.fromFile(textgrid_file)
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Tier', 'Start Time', 'End Time', 'Label'])
        for tier in tg:
            for interval in tier:
                writer.writerow([tier.name, interval.minTime, interval.maxTime, interval.mark])

def parse_textgrid(file_path):
    """
    Parse a TextGrid file using convert_textgrid_to_csv to extract interval data into a DataFrame.
    """
    csv_output = file_path.replace(".TextGrid", ".csv")
    convert_textgrid_to_csv(file_path, csv_output)
    return pd.read_csv(csv_output)

def adjust_intervals(df, chunk_start_time):
    """
    Adjust the 'Start Time' and 'End Time' of each interval based on the chunk's start time.
    """
    df['Start Time'] += chunk_start_time
    df['End Time'] += chunk_start_time
    return df

def process_textgrid_folder(folder_path, chunk_duration=40):
    """
    Process all TextGrid files in a folder, separating alignments and transcriptions,
    and assembling them while adjusting time offsets.
    """
    alignments = pd.DataFrame()
    transcriptions = pd.DataFrame()

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".TextGrid"):
            file_path = os.path.join(folder_path, filename)
            data = parse_textgrid(file_path)
            
            match = re.search(r'chunk_(\d+)', filename)
            if match:
                chunk_number = int(match.group(1))
                chunk_start_time = chunk_number * chunk_duration

                data = adjust_intervals(data, chunk_start_time)

                if "alignment" in filename.lower():
                    alignments = pd.concat([alignments, data], ignore_index=True)
                elif "transcription" in filename.lower():
                    transcriptions = pd.concat([transcriptions, data], ignore_index=True)

    alignments = alignments.sort_values(by=['Start Time', 'End Time']).reset_index(drop=True)
    transcriptions = transcriptions.sort_values(by=['Start Time', 'End Time']).reset_index(drop=True)
    
    return alignments, transcriptions

def preprocess_dataframe(df):
    """
    Preprocess the DataFrame by reordering the index column sequentially.
    """
    df['index'] = range(1, len(df) + 1)
    return df

def save_dataframes(alignments_df, transcriptions_df, output_folder):
    """
    Save the alignment and transcription DataFrames to CSV files.
    """
    os.makedirs(output_folder, exist_ok=True)

    phonems_df = alignments_df[alignments_df['Tier']=='phones']
    words_df = alignments_df[alignments_df['Tier']=='words']

    phonems_df.drop('Tier', axis=1, inplace=True)
    words_df.drop('Tier', axis=1, inplace=True)
    alignments_df.drop('Tier', axis=1, inplace=True)
    transcriptions_df.drop('Tier', axis=1, inplace=True)

    phonems_df.rename(columns={'Label': 'Phone'}, inplace=True)
    words_df.rename(columns={'Label': 'Words'}, inplace=True)
    transcriptions_df.rename(columns={'Label': 'Sentence'}, inplace=True)
    
    phonems_csv = os.path.join(output_folder, "all_phonems.csv")
    words_csv = os.path.join(output_folder, 'all_words.csv')
    alignments_csv = os.path.join(output_folder, "all_alignments.csv")
    transcriptions_csv = os.path.join(output_folder, "all_transcriptions.csv")

    alignments_df.to_csv(alignments_csv, index=False)
    transcriptions_df.to_csv(transcriptions_csv, index=False)
    phonems_df.to_csv(phonems_csv, index=False)
    words_df.to_csv(words_csv, index=False)

    print(f"Alignments saved to: {alignments_csv}")
    print(f"Transcriptions saved to: {transcriptions_csv}")
    print(f"Words saved to: {words_csv}")
    print(f"Phonems saved to: {phonems_csv}")