import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from signal_processor import SignalProcessor

def run_interactive_dashboard():
    RAW_DIR = "../../dataset/raw"
    csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))

    if not csv_files:
        print("[ERROR] No CSV files found in dataset/raw/")
        return

    processor = SignalProcessor()
    fs = processor.fs
    channels = processor.eeg_channels

    print(f"[INFO] Loading data from {len(csv_files)} subject(s)...")

    DURATION = 5
    TOTAL_SAMPLES = DURATION * fs
    WINDOW_SIZE = fs * 1

    all_data = {}
    colors = plt.cm.tab10.colors

    for subj_idx, file in enumerate(csv_files):
        subj_name = os.path.basename(file).split('.')[0]

        # Detect CSV header row
        header_idx = 0
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                if 'EEG.AF3' in line or 'AF3' in line:
                    header_idx = i
                    break

        df = pd.read_csv(file, header=header_idx, low_memory=False)

        # Drop units row if present
        try:
            float(df.iloc[0][channels[0]])
        except ValueError:
            df = df.drop(0).reset_index(drop=True)

        for col in channels:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        eeg_filtered = processor.apply_filter(df[channels].values)

        marker_col = 'MarkerValueInt' if 'MarkerValueInt' in df.columns else 'Marker'
        df[marker_col] = pd.to_numeric(df[marker_col], errors='coerce').fillna(0)
        marker_indices = df.index[df[marker_col] > 0].tolist()

        if marker_indices:
            start_idx = marker_indices[0]
            snippet = eeg_filtered[start_idx: start_idx + TOTAL_SAMPLES]
            if len(snippet) < TOTAL_SAMPLES:
                snippet = np.pad(snippet, ((0, TOTAL_SAMPLES - len(snippet)), (0, 0)), mode='constant')
            all_data[subj_name] = snippet

    if not all_data:
        print("[ERROR] No markers found — cannot initialise visualisation.")
        return

    fig, ax = plt.subplots(figsize=(15, 9))
    fig.canvas.manager.set_window_title("BCI EEG Dashboard — Neurandiar")
    plt.subplots_adjust(bottom=0.3, left=0.1, right=0.95, top=0.9)

    time_axis = np.linspace(0, DURATION, TOTAL_SAMPLES)
    state = {'current_ch': 0, 'paused': False}

    lines = []
    for s_idx, (name, data) in enumerate(all_data.items()):
        line, = ax.plot(time_axis, data[:, 0], label=name,
                        color=colors[s_idx % 10], alpha=0.8, lw=1.5)
        lines.append(line)

    ax.set_title(f"EEG Monitor — Channel: {channels[0]}", fontsize=16, fontweight='bold')
    ax.set_ylim(-150, 150)
    ax.set_ylabel("Amplitude (uV)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', ncol=5, fontsize=8)

    ax_pause = plt.axes([0.45, 0.22, 0.1, 0.05])
    btn_pause = Button(ax_pause, 'PAUSE', color='indianred', hovercolor='tomato')

    def toggle_pause(event):
        state['paused'] = not state['paused']
        btn_pause.label.set_text('PLAY' if state['paused'] else 'PAUSE')
        btn_pause.color = 'seagreen' if state['paused'] else 'indianred'
        fig.canvas.draw_idle()
    btn_pause.on_clicked(toggle_pause)

    channel_buttons = []
    for i, ch_name in enumerate(channels):
        row, col = i // 7, i % 7
        ax_btn = plt.axes([0.05 + col * 0.13, 0.12 - row * 0.06, 0.11, 0.045])
        btn = Button(ax_btn, ch_name.replace("EEG.", ""), color='whitesmoke', hovercolor='lightblue')

        def make_callback(ch_idx):
            def switch(event):
                state['current_ch'] = ch_idx
                ax.set_title(f"EEG Monitor — Channel: {channels[ch_idx]}", fontsize=16, fontweight='bold')
                for s_idx, (name, data) in enumerate(all_data.items()):
                    lines[s_idx].set_ydata(data[:, ch_idx])
                fig.canvas.draw_idle()
            return switch

        btn.on_clicked(make_callback(i))
        channel_buttons.append(btn)

    def update(frame):
        if not state['paused']:
            curr_start_time = (frame % (TOTAL_SAMPLES - WINDOW_SIZE)) / fs
            ax.set_xlim(curr_start_time, curr_start_time + 1.0)
        return lines

    print("[INFO] Dashboard ready. Use the channel buttons to switch views.")

    ani = FuncAnimation(fig, update, frames=np.arange(0, TOTAL_SAMPLES, 4),
                        interval=20, blit=False)

    plt.show()

if __name__ == "__main__":
    run_interactive_dashboard()
