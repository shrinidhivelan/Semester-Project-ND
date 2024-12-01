import pandas as pd
import ast
from Assemble_preprocess import *


""" 
movie_name = 'After_The_Rain_exp'
main_path = "/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/DataframeX"

"""

def assemble(movie_name, main_path):

    movie_name = 'After_The_Rain_exp'
    main_path = "/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/DataframeX"
    file_names = ["action_detection", "all_words", "all_transcriptions", "brightness_contrast_sharpness_data",
                "ColorDataframe", "frequency_time_series_data", "object_detection", "Scene_time_series",
                "texture_analysis_data", "all_phonems"]


    ########### 0 ########### ACTION

    # use one hot labelling : 
    df = pd.read_csv(main_path+"/"+movie_name +"/"+file_names[0]+".csv")
    # One-hot encode the 'Predicted Class' column
    encoded_df_action = pd.get_dummies(df, columns=["Predicted Class"], prefix="action")

    ##### We skip alignments and transcriptions 

    ########### 3 ########### BRIGHTNESS

    encoded_df_brightness = pd.read_csv(main_path+"/"+movie_name +"/"+file_names[3]+".csv")


    ########### 4 ########### COLOR

    encoded_df_color = pd.read_csv(main_path+"/"+movie_name +"/"+file_names[4]+".csv")

    ########### 5 ########### FREQUENCY

    encoded_df_frequency = pd.read_csv(main_path+"/"+movie_name +"/"+file_names[5]+".csv")

    ########### 6 ########### OBJECT DETECTION

    df_objects = pd.read_csv(main_path+"/"+movie_name +"/"+file_names[6]+".csv")
    # Convert the 'Detected Objects' column from string to dictionary
    df_objects["Detected Objects"] = df_objects["Detected Objects"].apply(ast.literal_eval)

    # Process the "Detected Objects" column
    # Extract all unique object keys from the dictionary
    unique_objects = set(
        obj for detected in df_objects["Detected Objects"] for obj in detected.keys()
    )
    # Create a new column for each object
    for obj in unique_objects:
        df_objects["Object"+obj] = df_objects["Detected Objects"].apply(lambda x: x.get(obj, 0))
    # Drop the original "Detected Objects" column if no longer needed
    df_objects = df_objects.drop(columns=["Detected Objects"])
    encoded_objects = df_objects.copy()



    ########### 7 ########### SCENE DETECTION


    df_scene = pd.read_csv(main_path+"/"+movie_name +"/"+file_names[7]+".csv")
    df_scene = df_scene[["Time (s)", "Top Class"]]
    # Use pd.get_dummies to create a column for each unique value in "Top Class"
    encoded_scene = pd.get_dummies(df_scene["Top Class"], prefix="Scene")
    # Combine the original DataFrame with the dummies
    encoded_scene = pd.concat([df_scene, encoded_scene], axis=1)
    # Drop the original "Top Class" column if no longer needed
    encoded_scene = encoded_scene.drop(columns=["Top Class"])


    ########### 8 ########### TEXTURE ANALYSIS

    encoded_df_texture = pd.read_csv(main_path+"/"+movie_name +"/"+file_names[8]+".csv")
    encoded_df_texture = encoded_df_texture.rename(columns={'Contrast': 'Texture contrast'})

    ########### 9 ########### ZOOM
    encoded_df_zoom = pd.read_csv(main_path+"/"+movie_name +"/"+file_names[-1]+".csv")



    ########### START MERGING PROCESS ###########

    ##### First merge :

    # Merge both DataFrames on Timestamp (from action_df) and Time (s) (from brightness_df)
    merged_df = pd.merge(encoded_df_brightness, encoded_df_action, right_on='Timestamp', left_on='Time (s)', how='left')

    # Drop the 'Timestamp' column from the merged DataFrame, keeping 'Time (s)'
    merged_df = merged_df.drop(columns=['Timestamp', 'Frame Index'])

    ##### Second merge :

    merged_df2 = pd.merge(merged_df, encoded_df_color, right_on='Time (s)', left_on='Time (s)', how='left')
    merged_df2 = merged_df2.drop(columns=['Frame Index'])


    ##### Third merge :

    merged_df3 = pd.merge(merged_df2, encoded_df_frequency, right_on='Time (s)', left_on='Time (s)', how='left')


    ##### Fourth merge :

    merged_df4 = pd.merge(merged_df3, encoded_objects, right_on='Time (s)', left_on='Time (s)', how='left')
    merged_df4 = merged_df4.drop(columns=['Frame Index'])

    ##### Fifth merge :

    merged_df5 = pd.merge(merged_df4, encoded_scene, right_on='Time (s)', left_on='Time (s)', how='left')

    ##### Sixth merge : 

    merged_df6 = pd.merge(merged_df5, encoded_df_texture, right_on='Time (s)', left_on='Time (s)', how='left')
    merged_df6 = merged_df6.drop(columns=['Frame Index'])


    ##### final merge on time (s) with zoomed in or not :

    merged_df6 = pd.merge(merged_df6, encoded_df_zoom, right_on='Time (s)', left_on='Time (s)', how='left')
    merged_df6 = merged_df6.drop(columns=['Frame Index'])


    ######## Final preprocessing :

    # Convert all NaN values to 0
    merged_df6 = merged_df6.fillna(0)

    # Convert all True values to 1 and all False values to 0
    merged_df6 = merged_df6.replace({True: 1, False: 0})


    merged_df6 = merged_df6.drop_duplicates(subset=['Time (s)'])


    ####### Now the text Files!!
    df_words = pd.read_csv(main_path+"/"+movie_name +"/"+file_names[1]+".csv")
    df_transcriptions = pd.read_csv(main_path+"/"+movie_name +"/"+file_names[2]+".csv")
    df_phonems = pd.read_csv(main_path+"/"+movie_name +"/"+file_names[-1]+".csv")

    df_to_merge = merge_text(merged_df6, df_words)
    merged_df_intermediate_1 = pd.merge(merged_df6, df_to_merge[['Time (s)', 'Words']], 
                        right_on='Time (s)', left_on='Time (s)', how='left')

    df_to_merge2 = merge_text(merged_df_intermediate_1, df_phonems)
    merged_df_intermediate_2 = pd.merge(merged_df_intermediate_1, df_to_merge2[['Time (s)', 'Phone']], 
                        right_on='Time (s)', left_on='Time (s)', how='left')

    df_to_merge_final = merge_text(merged_df_intermediate_2, df_transcriptions)
    merged_df7 = pd.merge(merged_df_intermediate_2, df_to_merge_final[['Time (s)', 'Sentence']], 
                        right_on='Time (s)', left_on='Time (s)', how='left')




    ##### Final DF to save as a csv file :
    final_df = merged_df7.copy()
    final_df = final_df.drop(columns=['Scene', 'Frame'])


    final_df.to_csv(main_path+"/"+movie_name+"/Final_df2.csv", index=False)




def preprocess_dataframe(df):
    """
    Preprocess the DataFrame by sorting and reordering indexes.
    """
    # Sort by 'xmin' column
    #df = df.sort_values(by=['xmin', 'xmax']).reset_index(drop=True)


    # Reorder the index column sequentially
    df['index'] = range(1, len(df) + 1)

    return df


def merge_text(df, text_df):
    # Initialize an empty list to store results
    merged_rows = []

    # Iterate over transcription DataFrame
    for _, row in text_df.iterrows():
        start_time = row['Start Time']
        end_time = row['End Time']
        
        # Filter scene_df where 'Time (s)' is between 'Start Time' and 'End Time'
        matching_rows = df[(df['Time (s)'] >= start_time) & (df['Time (s)'] <= end_time)]
        
        # Add transcription row to each matching scene row
        for _, match in matching_rows.iterrows():
            merged_rows.append(pd.concat([match, row.drop(['Start Time', 'End Time'])]))

    # Create a new DataFrame from the merged rows
    merged_df = pd.DataFrame(merged_rows)
    return merged_df