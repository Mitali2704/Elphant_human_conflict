import os

try:
    from playsound import playsound

    def play_alert(sound_file):
        if os.path.exists(sound_file):
            playsound(sound_file)
        else:
            print(f"Sound file {sound_file} not found.")

    def stop_alert():
        # playsound can't stop once started
        print("Stop not supported with playsound.")

except ImportError:
    import pygame

    def play_alert(sound_file):
        if os.path.exists(sound_file):
            pygame.mixer.init()
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play(-1)  # 🔁 loop until stopped
        else:
            print(f"Sound file {sound_file} not found.")

    def stop_alert():
        pygame.mixer.music.stop()
