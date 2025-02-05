import pyaudio
import wave
import numpy as np
import time
from datetime import datetime

# Parameters
THRESHOLD = 2000  # Adjust based on environment
CHUNK = 2048  # Larger chunk size for smoother audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000  # High-quality audio
SOUND_END_DELAY = 4  # Time in seconds to stop recording after sound ends

# Initialize the audio system
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

def save_audio(frames):
    """Save recorded audio as a WAV file with a unique timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recorded_audio_{timestamp}.wav"
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Saved: {filename}")
    return filename

def audio_detection():
    """Detects speech and saves it as an audio file."""
    print("Monitoring for speech during the exam...")
    sound_detected = False
    last_sound_time = 0
    frames = []

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
                frames.append(data)

            # If sound stops for SOUND_END_DELAY, save the recording
            if sound_detected and (time.time() - last_sound_time > SOUND_END_DELAY):
                print("Speaking stopped, saving audio...")
                save_audio(frames)
                frames = []
                sound_detected = False

        except KeyboardInterrupt:
            break

    print("Stopping audio detection...")
    stream.stop_stream()
    stream.close()
    p.terminate()

# # Run the detection
# if __name__ == "__main__":
#     audio_detection()