import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import moviepy
    print(f"moviepy version: {moviepy.__version__}")
    from moviepy.editor import AudioFileClip
    print("Successfully imported AudioFileClip from moviepy.editor")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Other error: {e}")

try:
    import librosa
    print(f"librosa version: {librosa.__version__}")
except ImportError:
    print("librosa not installed")
