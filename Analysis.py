# Notes 
# Assume 24 bit ADC:
# 2^24 = 16,777,216 possible values (−8,388,608 to 8,388,607)

# Assume +- 60,000 nT

#Scale factor:
# 60,000 nT / 8,388,608 = 0.007152557 nT per ADC count


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

file_path = Path("data/magic_20250815_121509.rawtm.ob")

time = []
Bx_raw, By_raw, Bz_raw = [], [], []

with open(file_path, "r") as file:
    for line in file:
        try:
            t, x, y, z = map(float, line.strip().split(","))
            time.append(t)
            Bx_raw.append(x)
            By_raw.append(y)
            Bz_raw.append(z)
        except ValueError:
            continue

time = np.array(time) 
Bx_raw = np.array(Bx_raw)
By_raw = np.array(By_raw)
Bz_raw = np.array(Bz_raw)

scale_factor = 60000 / (2**23)
Bx = Bx_raw * scale_factor
By = By_raw * scale_factor
Bz = Bz_raw * scale_factor

# Plotting Magnetic Raw Data. 
fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

axs[0].plot(time, Bx, label="Bx [nT]", color="tab:blue")
axs[1].plot(time, By, label="By [nT]", color="tab:orange")
axs[2].plot(time, Bz, label="Bz [nT]", color="tab:green")

axs[0].set_ylabel("Bx [nT]")
axs[1].set_ylabel("By [nT]")
axs[2].set_ylabel("Bz [nT]")
axs[2].set_xlabel("Time [s]")

for ax in axs:
    ax.grid(True)
    ax.legend(loc="upper right")

plt.suptitle("MAGIC Outboard Magnetic Field (Raw Converted)")
plt.tight_layout()
plt.show()

# Gain Calculation 
def analyze_spikes(B, time, threshold=3000, min_spacing=5, window_size=5, offset=0.5):
    diff_B = np.diff(B)
    spike_indices = np.where(np.abs(diff_B) > threshold)[0]
    spike_times = time[spike_indices]

    # Group nearby spikes
    grouped_times = []
    last_time = -np.inf
    for t in spike_times:
        if t - last_time >= min_spacing:
            grouped_times.append(t)
            last_time = t

    # Compute averages before and after each grouped spike

    ### Need to add an offset to the time array to avoid the spike time itself ###
    B_before = []
    B_after = []
    for t_spike in grouped_times:
        before_mask = (time >= t_spike - window_size) & (time < t_spike - offset)
        after_mask = (time > t_spike) & (time <= t_spike + window_size - offset)
        if np.any(before_mask) and np.any(after_mask):
            B_before.append(np.mean(B[before_mask]))
            B_after.append(np.mean(B[after_mask]))

    return grouped_times, B_before, B_after

bx_times, bx_before, bx_after = analyze_spikes(Bx, time)
by_times, by_before, by_after = analyze_spikes(By, time)
bz_times, bz_before, bz_after = analyze_spikes(Bz, time)

bx_spike_data = list(zip(bx_times, bx_before, bx_after))
by_spike_data = list(zip(by_times, by_before, by_after))
bz_spike_data = list(zip(bz_times, bz_before, bz_after))

# Results!! Yay
def print_spike_table(label, times, B_before, B_after):
    print(f"\nDetected {len(times)} {label} spikes:")
    for i, (t, b0, b1) in enumerate(zip(times, B_before, B_after), 1):
        print(f"Spike {i}:")
        print(f"  Time       : {t:.3f} s")
        print(f"  {label} before : {b0:.2f} nT")
        print(f"  {label} after  : {b1:.2f} nT")

print_spike_table("Bx", bx_times, bx_before, bx_after)
print_spike_table("By", by_times, by_before, by_after)
print_spike_table("Bz", bz_times, bz_before, bz_after)

Deltas = [-0.5, -1, -1.5, -2, -2.5, 0.5, 1, 1.5, 2, 2.5]

def gain_calcualtion(spike_data, deltas):
    gains = []
    spike_data = spike_data[:len(deltas) * 2] 

    for i in range(0, len(spike_data), 2):
        spike1 = spike_data[i]
        spike2 = spike_data[i + 1]
        dv1 = deltas[i // 2]
        dv2 = deltas[i // 2]

        g1 = (spike1[2] - spike1[1]) / dv1 if dv1 != 0 else None
        g2 = (spike2[2] - spike2[1]) / dv2 if dv2 != 0 else None

        if g1 is not None and g2 is not None:
            avg_gain = (abs(g1) + abs(g2)) / 2
            gains.append(avg_gain)
        else:
            gains.append(None)

    return gains

bx_gains = gain_calcualtion(bx_spike_data, Deltas)
by_gains = gain_calcualtion(by_spike_data, Deltas)
bz_gains = gain_calcualtion(bz_spike_data, Deltas)

def print_gain_summary(label, gains):
    valid_gains = [g for g in gains if g is not None]
    print(f"\n{label} Gains [nT/V]:")
    for i, g in enumerate(valid_gains, 1):
        print(f"  Step {i}: {g:.2f} nT/V")
    print(f"  Average: {np.mean(valid_gains):.2f} ± {np.std(valid_gains):.2f} nT/V")

print_gain_summary("Bx", bx_gains)
print_gain_summary("By", by_gains)
print_gain_summary("Bz", bz_gains)

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(Deltas, bx_gains, marker='o', color='tab:blue', label="Bx Gain")
axs[1].plot(Deltas, by_gains, marker='o', color='tab:orange', label="By Gain")
axs[2].plot(Deltas, bz_gains, marker='o', color='tab:green', label="Bz Gain")

axs[0].set_ylabel("Gain [nT/V]")
axs[1].set_ylabel("Gain [nT/V]")
axs[2].set_ylabel("Gain [nT/V]")
axs[2].set_xlabel("Applied Voltage Step [V]")

for ax in axs:
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.grid(True)
    ax.legend(loc="upper left")

plt.suptitle("Magnetic Gain vs Voltage Step")
plt.tight_layout()
plt.show()