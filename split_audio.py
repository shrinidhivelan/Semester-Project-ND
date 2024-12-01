import torchaudio
import os

def split_audio_torchaudio(file_path, chunk_duration_sec, output_folder):
    # Load audio
    waveform, sample_rate = torchaudio.load(file_path)

    # Calculate the number of samples per chunk
    chunk_samples = sample_rate * chunk_duration_sec
    total_samples = waveform.size(1)
    num_chunks = total_samples // chunk_samples

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    chunks = []

    for i in range(num_chunks + 1):  # Include remainder
        start_sample = int(i * chunk_samples)
        end_sample = min(int((i + 1) * chunk_samples), total_samples)
        
        chunk = waveform[:, start_sample:end_sample]
        output_file = os.path.join(output_folder, f"chunk_{i + 1}.wav")
        
        # Save the chunk
        torchaudio.save(output_file, chunk, sample_rate)
        chunks.append(output_file)

    print(f"Audio split into {num_chunks + 1} chunks.")
    return chunks


