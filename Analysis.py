import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

file_path = Path("data/IB/magic_20250819_095514.rawtm.ib")

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

scale_factor = 1/128
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

# Results!! Yay (For spikes detected)
def print_spike_table(label, times, B_before, B_after):
    print(f"\nDetected {len(times)} {label} spikes:")
    for i, (t, b0, b1) in enumerate(zip(times, B_before, B_after), 1):
        print(f"Spike {i}:")
        print(f"  Time       : {t:.3f} s")
        print(f"  {label} before : {b0:.2f} nT")
        print(f"  {label} after  : {b1:.2f} nT")

# print_spike_table("Bx", bx_times, bx_before, bx_after)
# print_spike_table("By", by_times, by_before, by_after)
# print_spike_table("Bz", bz_times, bz_before, bz_after)


Deltas = [-0.5, -1, -1.5, -2, -2.5, 0.5, 1, 1.5, 2, 2.5]

# Scale factors (μT/V)
scale_factors = {
    'x': 19.75022,
    'y': 18.24994,
    'z': 20.25004
}

# Convert ΔV to applied magnetic field (nT)
Bx_applied = [v * scale_factors['x'] * 1000 for v in Deltas]
By_applied = [v * scale_factors['z'] * 1000 for v in Deltas]
Bz_applied = [v * scale_factors['y'] * 1000 for v in Deltas]

def compute_avg_deltaB(spike_data, deltas):
    deltaBs = []
    spike_data = spike_data[:len(deltas) * 2] 

    for i in range(0, len(spike_data), 2):
        s1 = spike_data[i]
        s2 = spike_data[i + 1]

        dB1 = s1[2] - s1[1] 
        dB2 = s2[2] - s2[1]

        avg_dB = (dB1 - dB2) / 2
        deltaBs.append(avg_dB)

    return deltaBs

# Compute average ΔB for each axis
bx_deltaB = compute_avg_deltaB(bx_spike_data, Deltas)
by_deltaB = compute_avg_deltaB(by_spike_data, Deltas)
bz_deltaB = compute_avg_deltaB(bz_spike_data, Deltas)

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(Bx_applied, bx_deltaB, marker='o', label="Bx ΔB")
axs[1].plot(By_applied, by_deltaB, marker='o', label="By ΔB")
axs[2].plot(Bz_applied, bz_deltaB, marker='o', label="Bz ΔB")

axs[0].set_ylabel("ΔB [nT]")
axs[1].set_ylabel("ΔB [nT]")
axs[2].set_ylabel("ΔB [nT]")
axs[2].set_xlabel("Applied Magnetic Field [nT]")

for ax in axs:
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.grid(True)
    ax.legend(loc="upper left")

plt.suptitle("Average ΔB vs Applied Magnetic Field")
plt.tight_layout()
plt.show()

fit_bx, cov_bx = np.polyfit(Bx_applied, bx_deltaB, 1, cov=True)
fit_by, cov_by = np.polyfit(By_applied, by_deltaB, 1, cov=True)
fit_bz, cov_bz = np.polyfit(Bz_applied, bz_deltaB, 1, cov=True)

gain_bx = abs(fit_bx[0])
gain_by = abs(fit_by[0])
gain_bz = abs(fit_bz[0])
unc_bx = np.sqrt(cov_bx[0, 0])
unc_by = np.sqrt(cov_by[0, 0])
unc_bz = np.sqrt(cov_bz[0, 0])

print("Estimated Sensor Gain:")
print(f"  Bx: {gain_bx:.2f} +- {unc_bx}")
print(f"  By: {gain_by:.2f} +- {unc_by}")
print(f"  Bz: {gain_bz:.2f} +- {unc_bz}")

# Cross analysis

def compute_avg_deltaB_other(spike_times, other_B, time, deltas, window_size=5, offset=0.5):
    deltaBs = []
    spike_times = spike_times[:len(deltas) * 2]
    
    for i in range(0, len(spike_times), 2):
        t1 = spike_times[i]
        t2 = spike_times[i + 1]

        mask_before_1 = (time >= t1 - window_size) & (time < t1 - offset)
        mask_after_1 = (time > t1) & (time <= t1 + (window_size - offset))

        mask_before_2 = (time >= t2 - window_size) & (time < t2 - offset)
        mask_after_2 = (time > t2) & (time <= t2 + (window_size - offset))

        if all([np.any(mask_before_1), np.any(mask_after_1),
                np.any(mask_before_2), np.any(mask_after_2)]):
            dB1 = np.mean(other_B[mask_after_1]) - np.mean(other_B[mask_before_1])
            dB2 = np.mean(other_B[mask_after_2]) - np.mean(other_B[mask_before_2])
            deltaBs.append((dB1 - dB2) / 2)

    return deltaBs

by_from_bx = compute_avg_deltaB_other(bx_times, By, time, Deltas)
bz_from_bx = compute_avg_deltaB_other(bx_times, Bz, time, Deltas)

bx_from_by = compute_avg_deltaB_other(by_times, Bx, time, Deltas)
bz_from_by = compute_avg_deltaB_other(by_times, Bz, time, Deltas)

bx_from_bz = compute_avg_deltaB_other(bz_times, Bx, time, Deltas)
by_from_bz = compute_avg_deltaB_other(bz_times, By, time, Deltas)

df = pd.DataFrame({
    "Bx_applied [nT]": [v * scale_factors['x'] for v in Bx_applied],
    "Delta By from Bx": by_from_bx,
    "Delta Bz from Bx [nT]": bz_from_bx,
    "By_applied [nT]": [v * scale_factors['z'] for v in By_applied],
    "Delta Bx from By [nT]": bx_from_by,
    "Delta Bz from By [nT]": bz_from_by,
    "Bz_applied [nT]": [v * scale_factors['y'] for v in Bz_applied],
    "Delta Bx from Bz [nT]": bx_from_bz,
    "Delta By from Bz [nT]": by_from_bz,
})

df.to_csv("cross_axis_gain_plot_data.csv", index=False)
print("Saved plot data to 'cross_axis_gain_plot_data.csv'")

# Fit and plot — same style
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axs[0].plot(Bx_applied, by_from_bx, marker='o', label="ΔBy from Bx")
axs[0].plot(Bx_applied, bz_from_bx, marker='o', label="ΔBz from Bx")
axs[0].set_title("Spikes in Bx")
axs[1].plot(By_applied, bx_from_by, marker='o', label="ΔBx from By")
axs[1].plot(By_applied, bz_from_by, marker='o', label="ΔBz from By")
axs[1].set_title("Spikes in By")
axs[2].plot(Bz_applied, bx_from_bz, marker='o', label="ΔBx from Bz")
axs[2].plot(Bz_applied, by_from_bz, marker='o', label="ΔBy from Bz")
axs[2].set_title("Spikes in Bz")

for ax in axs:
    ax.set_ylabel("ΔB [nT]")
    ax.axhline(0, color='grey', linestyle='--')
    ax.grid(True)
    ax.legend()

axs[2].set_xlabel("Applied Magnetic Field [nT]")
plt.suptitle("Cross-Axis Gain Analysis")
plt.tight_layout()
plt.show()

# Fit and print gains for cross-axis analysis
def fit_and_print_gain(applied, deltaB, axis_label):
    fit, cov = np.polyfit(applied, deltaB, 1, cov=True)
    gain = abs(fit[0])
    unc = np.sqrt(cov[0, 0])
    print(f"Cross-axis gain for {axis_label}: {gain:.5f} +- {unc}")

fit_and_print_gain(Bx_applied, by_from_bx, "By from Bx")
fit_and_print_gain(Bx_applied, bz_from_bx, "Bz from Bx")
fit_and_print_gain(By_applied, bx_from_by, "Bx from By")
fit_and_print_gain(By_applied, bz_from_by, "Bz from By")
fit_and_print_gain(Bz_applied, bx_from_bz, "Bx from Bz")
fit_and_print_gain(Bz_applied, by_from_bz, "By from Bz")