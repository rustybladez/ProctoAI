import pyaudio
import wave
import numpy as np
import time
from threading import Thread

# Parameters
THRESHOLD = 2000  # Adjust based on environment
CHUNK = 2048  # Larger chunk size for smoother audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000  # High-quality audio
SOUND_END_DELAY = 4 # Time in seconds to stop recording after sound ends

# Initialize the audio system
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


def record_segment(frames, file_index):
    """Saves the audio frames to a file."""
    filename = f"speaking_event_{file_index}.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    print(f"Audio recorded and saved as {filename}")


def audio_detection():
    """Detects speaking and records audio segments during speaking."""
    print("Monitoring for speech during the exam...")
    sound_detected = False
    last_sound_time = 0
    frames = []
    file_index = 1  # To keep track of saved files

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Check if the audio exceeds the threshold
            if np.max(np.abs(audio_data)) > THRESHOLD:
                if not sound_detected:
                    print("Speaking detected, starting recording...")
                    sound_detected = True
                last_sound_time = time.time()

                # Collect audio frames for this segment
                frames.append(data)

            # If sound stops for SOUND_END_DELAY, save the recording
            if sound_detected and (time.time() - last_sound_time > SOUND_END_DELAY):
                print("Speaking stopped, saving recording...")
                record_segment(frames, file_index)
                frames = []  # Reset frames for the next segment
                sound_detected = False
                file_index += 1

        except KeyboardInterrupt:
            break

    print("Stopping audio detection...")
    stream.stop_stream()
    stream.close()
    p.terminate()


def start_proctoring():
    """Starts the proctoring system."""
    detect_thread = Thread(target=audio_detection)
    detect_thread.daemon = True
    detect_thread.start()

    # Keep the program running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nProctoring stopped.")


# # Start the proctoring system
# if __name__ == "__main__":
#     start_proctoring()
