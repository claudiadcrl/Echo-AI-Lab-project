import tkinter as tk
from tkinter import ttk

class AudioVisualizerGUI:
    def __init__(self, root, active_effects, track_list):
        self.root = root
        self.root.title("Your Board")
        self.root.configure(background='black')
        self.root.geometry("500x400+800+250")

        self.active_effects = active_effects
        self.track_list = track_list

        self.track_label = tk.Label(root, text="\n\nTrack: -", font=("Segoe UI", 14), fg='white', bg='black')
        self.track_label.pack(pady=10)

        self.rotation_label = tk.Label(root, text="Rotation: 0.0°", font=("Segoe UI", 10),fg='white', bg='black')
        self.rotation_label.pack(pady=10) #adapts automatically to window

        #Frame contains effects names and bars that show values
        self.effects_frame = tk.Frame(root, bg='black')
        self.effects_frame.pack(pady=10, fill="x")

        self.effect_bars = {}

    def update(self, rotation_angle, current_track_index):

        track_titles={
            0: "\'Fusion Bass Loop\' - Alesis",
            1: "\'36 Chambers\' - 1000 Handz",
            2: "\'Crash in Space\' - Paweł Spychała",
            3: "\'Conscience\' - Paweł Spychała",
            4: "\'The Skies\' - Tebo Steele",
            5: "\'Fusion Bass Loop\' - Alesis"
        }

        #Update track name
        track_name = self.track_list[current_track_index]
        self.track_label.config(text=f"\n\nTrack {current_track_index}: {track_titles[current_track_index]}")

        #Update rotation angle
        self.rotation_label.config(text=f"Rotation: {rotation_angle:.1f}°")

        #clears widgets to get new ones with updated values
        for widget in self.effects_frame.winfo_children():
            widget.destroy()
        self.effect_bars.clear()

        style=ttk.Style()
        style.theme_use('clam')
        style.configure("cyan.Horizontal.TProgressbar", foreground='#57ffd8', background='#57ffd8')
        #Show active effects values
        for effect_name, effect in self.active_effects.items():
            #make frame (container)
            frame = tk.Frame(self.effects_frame, bg='black')
            frame.pack(fill="x", pady=5, padx=30)

            #Make text label
            label = tk.Label(frame, text=effect_name.capitalize(), width=12, anchor="w", fg='white', bg='black')
            label.pack(side="left")

            #initialize progress bar
            bar = ttk.Progressbar(frame, style="cyan.Horizontal.TProgressbar",length=250, mode="determinate")
            bar.pack(side="left", padx=5)
            self.effect_bars[effect_name] = bar

            #set value of progress bar
            value = self._get_normalized_value(effect_name, effect)
            bar["value"] = value * 100 

    def _get_normalized_value(self, effect_name, effect):
        #normalizes the effects values for bars
        match effect_name:
            case "gain":
                return (effect.gain_db + 20) / 40  # -20..20 dB
            case "bitcrush":
                return (24 - effect.bit_depth) / 20
            case "distortion":
                return min(max(effect.drive_db / 40, 0), 1)
            case "chorus":
                return min(effect.rate_hz / 5, 1)
            case "compressor":
                return min(-effect.threshold_db / 40, 1)
            case _:
                return 0.0