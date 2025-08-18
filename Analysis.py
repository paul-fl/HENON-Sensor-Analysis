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

    ### Need to add an offset to the time array to avoid the spike time itself ### DONE!!
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

print_spike_table("Bx", bx_times, bx_before, bx_after)
print_spike_table("By", by_times, by_before, by_after)
print_spike_table("Bz", bz_times, bz_before, bz_after)


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
axs[2].set_xlabel("Applied Magnetic Field [μT]")

for ax in axs:
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.grid(True)
    ax.legend(loc="upper left")

plt.suptitle("Average ΔB vs Applied Magnetic Field (Gain = Slope)")
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


