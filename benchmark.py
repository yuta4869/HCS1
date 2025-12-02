# To run this script, use the virtual environment's python:
# .venv/bin/python3 benchmark.py

import time
import os
import glob
import csv
import soundfile as sf
from faster_whisper import WhisperModel
import config # To get model name and compute type

def run_benchmark(test_audio_dir: str):
    """
    Runs a benchmark on all WAV files in a directory and prints the results.
    """
    model_name = config.WHISPER_MODEL_NAME
    compute_type = config.WHISPER_COMPUTE_TYPE
    
    print(f"Loading model '{model_name}' with compute type '{compute_type}'...")
    try:
        model = WhisperModel(model_name, device="cuda", compute_type=compute_type)
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure CUDA is available and dependencies are installed correctly.")
        return

    wav_files = glob.glob(os.path.join(test_audio_dir, "*.wav"))
    if not wav_files:
        print(f"No WAV files found in '{test_audio_dir}'")
        return

    results = []
    print(f"\nFound {len(wav_files)} audio file(s) to benchmark.")

    for audio_path in wav_files:
        try:
            # Get audio duration
            info = sf.info(audio_path)
            audio_duration = info.duration

            # Measure transcription time
            start_time = time.perf_counter()
            segments, _ = model.transcribe(audio_path, beam_size=config.WHISPER_TRANSCRIBE_BEAM_SIZE)
            # We need to consume the generator
            transcribed_text = "".join(segment.text for segment in segments)
            end_time = time.perf_counter()

            transcription_time = end_time - start_time
            rtf = transcription_time / audio_duration if audio_duration > 0 else 0

            results.append({
                "filename": os.path.basename(audio_path),
                "audio_duration_s": f"{audio_duration:.3f}",
                "transcription_time_s": f"{transcription_time:.3f}",
                "real_time_factor": f"{rtf:.3f}",
            })
            print(f"- Processed {os.path.basename(audio_path)} (RTF: {rtf:.3f}) | Text: '{transcribed_text.strip()}'")

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    # Print results in a nice table format
    if results:
        print("\n--- Benchmark Results ---")
        header = results[0].keys()
        # Find max column widths
        widths = {k: max(len(str(r[k])) for r in results) for k in header}
        widths = {k: max(widths[k], len(k)) for k in header} # include header length

        # Header
        header_line = " | ".join(f"{k:<{widths[k]}}" for k in header)
        print(header_line)
        print("-" * len(header_line))

        # Rows
        for res in results:
            row_line = " | ".join(f"{str(res[k]):<{widths[k]}}" for k in header)
            print(row_line)
        print("-" * len(header_line))

if __name__ == "__main__":
    TEST_DIR = "test_audio"
    run_benchmark(TEST_DIR)