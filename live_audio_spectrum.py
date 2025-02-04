import matplotlib
matplotlib.use('TkAgg')

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft
import time
import threading
from tkinter import TclError

# ------------ Audio Setup ---------------
CHUNK = 1024 * 2  # samples per frame
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
AMPLITUDE_LIMIT = 4096  # Adjusted amplitude range

p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

# ------------ Plot Setup ---------------
plt.ion()

plt.style.use('dark_background')  # Dark background style
fig, (ax1, ax2) = plt.subplots(2, 1, layout='constrained', figsize=(12, 6))

x = np.arange(0, 2 * CHUNK, 2)  # Time domain
xf = np.linspace(0, RATE / 2, CHUNK // 2 + 1)  # Frequency domain

# Pre-allocate NumPy arrays
data_np = np.zeros(CHUNK, dtype=np.int16)
yf = np.zeros(CHUNK // 2 + 1, dtype=np.float32)

# Create line objects for waveform
line, = ax1.plot(x, np.random.rand(CHUNK), '-', lw=2)

# Create bar object for frequency spectrum (with bar thickness indepedent of frequency log-scale and edge color)
bars = ax2.bar(xf, np.random.rand(CHUNK // 2 + 1), width=.0, align='center', edgecolor='#98F5E1', linewidth=2)

# Format plots
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
ax1.set_ylim(-AMPLITUDE_LIMIT, AMPLITUDE_LIMIT)
ax1.set_xlim(0, 2 * CHUNK)

# Setup frequency spectrum plot with log-scale
ax2.set_title('FREQUENCY SPECTRUM')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude')
ax2.set_xlim(20, RATE / 2)
ax2.set_xscale('log')  # Log-scale frequency axis

print('Stream started')

# ------------ Audio Processing Thread ---------------
def audio_processing():
    global data_np, yf  # Allow the main thread to access updated values

    while True:
        try:
            # Read audio data
            data = stream.read(CHUNK, exception_on_overflow=False)
            data_np[:] = np.frombuffer(data, dtype=np.int16)  # ✅ Update in-place

            # Compute FFT using rfft
            yf[:] = np.abs(rfft(data_np)) / (512 * CHUNK)  # ✅ Update in-place

        except Exception as e:
            print(f"Error in audio_processing: {e}")
            break

# ------------ Update Function with Autoscaling ---------------
def update_plot():
    # Autoscaling for waveform
    max_amplitude = np.max(np.abs(data_np))  # Get max amplitude of the waveform
    ax1.set_ylim(-max_amplitude * 1.1, max_amplitude * 1.1)  # Scale y-axis dynamically
    
    # Autoscaling for spectrum
    max_spectrum = np.max(yf)  # Get max FFT magnitude
    ax2.set_ylim(0, max_spectrum * 1.1)  # Scale y-axis dynamically

    # Update plots using latest data
    line.set_ydata(data_np)

    # Update the bars of the frequency spectrum (instead of line)
    for bar, value in zip(bars, yf):
        bar.set_height(value)

    fig.canvas.draw()
    fig.canvas.flush_events()

if __name__ == '__main__':
    # Start audio processing in a separate thread
    audio_thread = threading.Thread(target=audio_processing, daemon=True)
    audio_thread.start()

    # ------------ GUI Update Loop (Must Run in Main Thread) ---------------
    while True:
        try:
            # Update the plot and autoscale
            update_plot()
            time.sleep(0.01)  # Prevents 100% CPU usage
        except KeyboardInterrupt:
            print("Exiting...")
            stream.stop_stream()
            stream.close()
            p.terminate()
            break
