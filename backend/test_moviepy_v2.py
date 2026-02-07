try:
    from moviepy import AudioFileClip
    print("Success: from moviepy import AudioFileClip")
except ImportError:
    print("Failed: from moviepy import AudioFileClip")

try:
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    print("Success: from moviepy.audio.io.AudioFileClip import AudioFileClip")
except ImportError:
    print("Failed: from moviepy.audio.io.AudioFileClip import AudioFileClip")
