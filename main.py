import tqdm
from brightness import *
from frequency_analysis import *
from color import *
from texture_analysis import *
from scene_detection import *
from object_counting import *
from audio_copy import *
from Assemble_preprocess import *
from zoom import *

def main():
    """ 
    movie_names = ['After_The_Rain_exp', 'Between_Viewings_exp', 'Big_Buck_Bunny_exp', 'Chatter_exp', 
               'Damaged_Kung_Fu_exp', 'First_Bite_exp', 'Lesson_Learned_exp', 'Payload_exp', 
               'Riding_The_Rails_exp', 'Sintel_exp', 'Spaceman_exp','Superhero_exp', 'Tears_of_Steel_exp', 
               'The_secret_number_exp', 'To_Claire_From_Sonny_exp', 'You_Again_exp']
    """
    
    #### Let's start with one single movie
    movie_names = ['After_The_Rain_exp']

    print("Starting to generate data...")
    for i, movie in enumerate(movie_names):

        video_path = f'/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/FilmFiles/{movie}.mp4'
        save_path_dataframe = '/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/DataframeX'
        save_path_image = '/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/Plots'
        
        print(f'{i+1}/{len(movie_names)} Starting with {movie}')

        print("Brightness, contrast and sharpness analysis...")
        #analyze_video_characteristics(video_path, save_path_dataframe, save_path_image, movie)

        print("Frequency analysis...")
        #analyze_brightness_frequency(video_path, save_path_dataframe, save_path_image, movie, window_duration=1, step_fraction=0.5)

        print("Color analysis...")
        #extract_and_save_color_histograms(video_path, save_path_dataframe, save_path_image, movie)

        print("Texture analysis...")
        #analyze_video_texture(video_path, save_path_dataframe, save_path_image, movie)

        print("Scene Detection...")
        #analyze_scene_classification(video_path, save_path_dataframe, movie)

        print("Object detection...")
        model_path = "./yolov8n.pt"
        #analyze_object_frequencies(video_path, save_path_dataframe, save_path_image, model_path, movie)

        print("Action detection...")
        ###### => This will be done in the following other folder...

        print("Audio transciption and alignment data...")
        textgrid_path = "/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/FilmTextGrids/"+movie
        output_folder = "/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/DataframeX/"+movie
        """ 
        alignments_df, transcriptions_df,  = process_textgrid_folder(textgrid_path, chunk_duration=40)
        alignments_df = preprocess_dataframe(alignments_df)
        #transcriptions_df = preprocess_dataframe(transcriptions_df)
        save_dataframes(alignments_df, transcriptions_df, output_folder)
        """
        """ 
        alignments_df, transcriptions_df = process_textgrid_folder(textgrid_path, chunk_duration=40)
        save_dataframes(alignments_df, transcriptions_df, output_folder)
        """

        print("Zoom-in detections")
        extract_and_detect_zoom(video_path, save_path_dataframe, movie)


        #print("Preprocessing and creating the final dataframe : ")
        #assemble(movie, save_path_dataframe)


    


if __name__ == "__main__":
    main()