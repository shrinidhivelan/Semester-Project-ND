"""
    Module to get the phonetic alignment of audio file in two steps:

    1. Get the transcription from whisperX with segment boundaries.
    2. Get the words and phones alignment from Montreal Forced Aligner (MFA).

    To run and/or test this code, here is a workflow you can follow:

    1. Create a virtual env. Example: `conda create --name alignment python=3.10`
    2. Install whisperx and its necessary dependencies. See here: https://github.com/m-bain/whisperX/tree/main
    3. Install MFA and download the required pretrained models and dictionaries. 
    See here: https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html
    4. Install TextGridTools for textgrid file operations. 
    See here: https://github.com/hbuschme/TextGridTools/tree/master

    In the example, be aware to set the data (audio files) path accordingly.
"""

from glob import glob
from pathlib import Path
from shutil import copyfile
from subprocess import run
from typing import Any, Dict, List
import tgt
import torch
import whisperx
import time
from split_audio import *
import tqdm

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

SAMPLING_RATE = 16000
# load SILERO VAD model
SILERO_MODEL, SILERO_UTILS = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=True,
    onnx=False
)
(get_speech_timestamps,
save_audio,
read_audio,
VADIterator,
collect_chunks) = SILERO_UTILS
vad_iterator = VADIterator(SILERO_MODEL)

def detect_speech(audio_file: str, threshold: float =0.5) -> bool:
    """Detect if audio file contains speech or no.

    Args:
        audio_file (str): The audio file to test.
        threshold (float): The threshold to consider the file as containing speech. Defaults to 0.5.

    Returns:
        bool: True if speech file, False otherwise.
    """
    wav = read_audio(audio_file, sampling_rate=SAMPLING_RATE)
    speech_probs = []
    window_size_samples = 512 if SAMPLING_RATE == 16000 else 256
    ## Get just probabilities
    for i in range(0, len(wav), window_size_samples):
        chunk = wav[i: i+ window_size_samples]
        if len(chunk) < window_size_samples:
            break
        speech_prob = SILERO_MODEL(chunk, SAMPLING_RATE).item()
        speech_probs.append(speech_prob)
    vad_iterator.reset_states() # reset model states after each audio
    mean_proba = sum(speech_probs) / len(speech_probs)
    # print(mean_proba)
    return mean_proba >= threshold
    print(f"Mean probs = {sum(speech_probs) / len(speech_probs)}") # first 10 chunks predicts


def get_transcription(
        audio_files: List[str],
        audio_language: str = "fr",
) -> Dict[str, List[Dict[str, Any]]]:
    """Transcribe audio files.

    Args:
        audio_files (List[str]): List of audio files to transcribe.
        audio_language (str): Code for the audio language. Default: fr (French).

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary of the results in whisperx format.
    """
    transcriptions = {}
    # whisper model for ASR
    model = whisperx.load_model(
        "large-v2",
        DEVICE,
        language=audio_language,
        compute_type=COMPUTE_TYPE
    )
    # whisperx model for alignment
    model_a, metadata = whisperx.load_align_model(
        language_code=audio_language,
        device=DEVICE
    )

    for i, audio_file in enumerate(audio_files):
        # Log info
        if i%500 == 0:
            print(f"Transcribing file number {i+1}...")
        if Path(audio_file).exists():
            is_speech = detect_speech(
                audio_file,
                0.01
            )
            if is_speech:
                audio = whisperx.load_audio(audio_file)
                result = model.transcribe(audio, batch_size=BATCH_SIZE)
                # Align whisper output using whisperx
                result = whisperx.align(
                    result["segments"], 
                    model_a, 
                    metadata, 
                    audio, 
                    DEVICE, 
                    return_char_alignments=False
                )
                transcriptions[audio_file] = result
            # else:
            #     print(f"The audio file {audio_file} does not contain 'enough' speech.")
        else:
            print(f"The audio file {audio_file} does not exist.")

    return transcriptions

def save_transcriptions_in_textgrid(
        transcriptions: Dict[str, List[Dict[str, Any]]],
        output_directory: str
) -> Dict[str, str]:
    """Save the transcriptions formatted in a textgrid with segment boundaries.

    Args:
        transcriptions (Dict[str, List[Dict[str, Any]]]): Dictionary of transcriptions in whisperx format.
        output_directory (str): Directory where to save generated files.

    Returns:
        List[str]: List of correctly saved files.
    """
    saved_files = {}
    name = 'Ortho'
    
    if not Path(output_directory).is_dir():
        raise FileNotFoundError(f"The folder {output_directory} does not exist.")
    else:
        for audio_file, transcription in transcriptions.items():
            # create ouptut file
            output_file = audio_file.split("/")[-1]
            extension = output_file[-4:]
            output_file = output_file.replace(extension, "_transcription.TextGrid")
            output_file = str(Path(output_directory).joinpath(output_file))
            if not Path(output_file).exists():
                segments = transcription['segments']
                # TODO: just for debug
                # Don't align when no real speech found in audio
                if len(segments) == 1:
                    if segments[0]['text'] == " ...":
                        continue
                if len(segments) > 0:
                    tg = tgt.TextGrid()
                    tier = tgt.IntervalTier(
                        start_time=segments[0]['start'],
                        end_time=segments[-1]['end'],
                        name=name
                    )
                    tg.add_tier(tier)
                    for seg in segments:
                        start = seg['start']
                        end = seg['end']
                        txt = seg['text']
                        seg_annotation = tgt.Annotation(
                            start_time=start,
                            end_time=end,
                            text=txt
                        )
                        tier.add_annotation(seg_annotation)
                    # write to textgrid file
                    tgt.write_to_file(tg, output_file, format="long")
                    saved_files[audio_file] = output_file

    return saved_files

def get_alignment(
        audio_files: List[str],
        transcription_files: List[str],
        output_directory: str,
        language: str = "en",
) -> List[str]:
    """Get the alignment at word and phone levels in a textgrid.
    Note: The audio files and transcription files must coincide in length 
    and order of inserted files, so that at each index the files correspond 
    exactly.

    Args:
        audio_files (List[str]): List of audio files.
        transcription_files (List[str]): List of corresponding transcriptions.
        output_directory (str): Directory where to save the generated files.
        language (str): Language of audio files. Default: fr (French).

    Returns:
        List[str]: List of correctly saved files.
    """
    saved_files = []
    # create a temporary MFA inputs/outputs directory
    #input_dir = Path("./sintel/MFA_inputs") #Path("./audio/MFA_inputs")
    #input_dir = Path("./sintel/MFA_inputs") #Path("./audio/MFA_inputs")
    #input_dir = Path("./sintel/MFA_inputs") #Path("./audio/MFA_inputs")
    input_dir = Path(output_directory+"/MFA_inputs") #Path("./audio/MFA_inputs")
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_directory+"/MFA_outputs")#Path("./audio/MFA_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    # this will only work if these MFA resources
    # are already downloaded on your machine
    mfa_model = {
        "en": "english_us_arpa", #"english_mfa",
        "fr": "french_mfa", 
        "ge": "german_mfa"
    }
    print("In the alignment function...")

    for i, (audio_file, transcription_file) in enumerate(zip(audio_files, transcription_files)):
        # Log info
        if i%500 == 0:
            print(f"Aligning file number {i+1}...")

        copyfile(audio_file, str(input_dir.joinpath(audio_file.split("/")[-1])))
        copyfile(
            transcription_file, 
            # remove the transcription in the name to cope with MFA requirements
            str(input_dir.joinpath(transcription_file.split("/")[-1].replace('_transcription', '')))
        )
        # run MFA align command
        cmd = f"mfa align {str(input_dir)} {mfa_model[language]} {mfa_model[language]} {output_dir} --clean"
        print("******************")
        print(cmd)
        print("******************")

        ret_code = run(cmd, shell=True).returncode
        if ret_code:
            raise OSError(f"Error aligning the file {audio_file}.")
        # move the created output file to output_directory
        mfa_outputs = output_dir.glob("*")
        for out in mfa_outputs:
            new_out = str(out).split("/")[-1].replace(".TextGrid", "_alignment.TextGrid")
            new_out = Path(output_directory).joinpath(new_out)
            cmd = f"mv {out} {new_out}"
            ret_code = run(cmd, shell=True).returncode
            if ret_code:
                raise OSError(f"Error moving the file {audio_file}.")
            else:
                saved_files.append(new_out)
        # clean temporary MFA inputs directory
        # to prepare for the next run
        _ = [f.unlink() for f in Path(input_dir).glob("*") if f.is_file()]
    return saved_files


if __name__ == "__main__":
    ### The name of all movies : 
    movie_names = ['Superhero_exp', 'Tears_of_Steel_exp', 'The_secret_number_exp', 
                   'To_Claire_From_Sonny_exp', 'You_Again_exp']


    # This is an example of how to run 
    # the pipeline. To use with precaution.
    # TODO: Modify the following
    # two variables accordingly
    # audio_files = [ "path/to/audio1.wav", "path/to/audio2.wav"]
    # audio_files = [ "/Users/shrinidhivelan/Desktop/Projet CSE I/audio/After_The_Rain_exp.wav"]
    # audio_files = ["/Volumes/LaCie/EPFL/Master sem3/Semester Project ND/FilmWaveFiles/Sintel_exp.wav"] #
    # 
    #audio_files = ['/Volumes/LaCie/EPFL/Master sem3/Semester Project ND/FilmWaveFiles/trial2.wav']#[ "audio/trial2.wav"]
    ### Do the following to separate

    # source of wave files :
    audio_source = '/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/FilmWaveFiles/'

    output_univ = '/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/FilmTextGrids/'

    Audios = []
    for movie in movie_names:
       print(f"Starting process for movie {movie}")
       chunks_dir = audio_source+movie+"Chunks"
       
       os.makedirs(chunks_dir, exist_ok=True)
       audio_files = split_audio_torchaudio(audio_source+movie+".wav", chunk_duration_sec=40, output_folder=chunks_dir)

       output_dir = output_univ+movie           
       os.makedirs(output_dir, exist_ok=True)

       start_time = time.time()
       print("Beginning transcribing files...")
       transcs = get_transcription(audio_files, "en")
       print("got it!")
       transcs_files = save_transcriptions_in_textgrid(transcs, output_dir)
       print(transcs_files)

       print("Beginning aligning files...")
       alignment_files = get_alignment(
            list(transcs_files.keys()), 
            list(transcs_files.values()), 
            output_dir

        )
       print("End of pipeline...")
       end_time = time.time()
       print(f"\t Execution time: {end_time - start_time:.2f}")
       print(f"\t# of initial audio files: {len(audio_files)}")
       print(f"\t# of transcribed audio files: {len(transcs_files)}")
       print(f"\t# of non transcribed files: {len(audio_files) - len(transcs_files)}")

       



""" 

    #output_dir = "/Volumes/LaCie/EPFL/Master sem3/Semester Project ND/FilmTextGrids/sintel/" 
    #output_dir = "sintel"
    output_dir = "/Volumes/LaCie/EPFL/Master sem3/Semester Project ND/FilmTextGrids"
    start_time = time.time()
    print("Beginning transcribing files...")
    transcs = get_transcription(audio_files, "en")
    print("got it!")
    transcs_files = save_transcriptions_in_textgrid(transcs, output_dir)
    print(transcs_files)

    # Problème est arrivé ici
    print("Beginning aligning files...")
    alignment_files = get_alignment(
        list(transcs_files.keys()), 
        list(transcs_files.values()), 
        output_dir
    )
    print("End of pipeline...")
    end_time = time.time()
    print(f"\t Execution time: {end_time - start_time:.2f}")
    print(f"\t# of initial audio files: {len(audio_files)}")
    print(f"\t# of transcribed audio files: {len(transcs_files)}")
    print(f"\t# of non transcribed files: {len(audio_files) - len(transcs_files)}")
"""