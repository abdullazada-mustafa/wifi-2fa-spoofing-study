# Testing a copied-signal attack on Wi-Fi-based two-factor authentication

A small first experiment following up on:

> A. A. S. AlQahtani, T. Alshayeb, M. Nabil, A. Patooghy,
> "Leveraging Machine Learning for Wi-Fi-Based Environmental Continuous
> Two-Factor Authentication," *IEEE Access*, vol. 12, 2024.

## The idea

The system grants access only when the user's devices see the same nearby Wi-Fi
access points with similar signal strengths. The paper evaluates attacks mainly
as random noise added to the signals, and notes that it does not directly address
adversarial attack techniques.

A real attacker can be more targeted: broadcast fake Wi-Fi beacons that copy the
names of legitimate access points at a convincing signal strength. This
experiment asks how often the paper's model types accept such forged readings.

## What I did

1. **Collected my own Wi-Fi scans** on a Windows laptop with
   `collect_wifi_simple.py` (passive scanning only, nothing transmitted), in the
   same column format as the authors' dataset: SSID, frequency, RSSI and label.
   - "Authentic" (label 1): scans at the user's position.
   - "Unauthorized" (label 0): scans about 10 meters away.
2. **Trained three of the paper's model types** (decision tree, k-nearest
   neighbors, random forest) on the signal features only: network name,
   frequency and signal strength.
3. **Simulated the copied-signal attack:** took the far-away samples, copied
   legitimate network names onto them and moved their signal strength 70% of the
   way toward the authentic average.
4. **Measured how many forged samples each model accepts**, compared with how
   many real far-away samples it accepts without any attack (the baseline).

## Results

178 samples (60 authentic, 118 unauthorized). Clean accuracy 0.778 for all three
models.

| Model | Real far samples accepted (baseline) | Forged samples accepted (attack) |
|---|---|---|
| Decision tree | 20.8% | **51.1%** (48/94) |
| Random forest | 20.8% | 28.7% (27/94) |
| K-nearest neighbors | 8.3% | 3.2% (3/94) |

**Takeaways**

- For the decision tree, the best model in the original paper, copied signals
  more than doubled the false acceptance rate (20.8% to 51.1%).
- The random forest was affected less, and k-nearest neighbors resisted the
  attack, so robustness to spoofing depends strongly on the model choice.

## Limitations

This is a small first test, not a replication of the paper:

- One laptop, one building, few visible networks, and only 178 samples.
- Windows reports signal strength as a percentage, which I converted to dBm with
  a standard linear approximation.
- The attack is simulated on recorded data, not performed with real hardware.

## Next steps

- Repeat with more access points, locations and the authors' dataset.
- Run the attack live with a low-cost board broadcasting look-alike beacons in a
  controlled lab setting.
- Make the models harder to fool, for example by adding features that are
  difficult to copy, such as the timing of each access point's beacons, and by
  training on forged examples.

## Files

- `collect_wifi_simple.py`: passive Wi-Fi scan collector for Windows.
- `wifi_2fa_spoof_attack.py`: trains the models and runs the attack simulation.
- `wifi_dataset_public.csv`: the scans used for the results above. I collected
  this data myself rather than using the authors' dataset. Network names are
  replaced with neutral IDs (AP01, AP02, ...) for privacy, in the same
  alphabetical order as the originals, so the results reproduce exactly.

To reproduce the results, rename `wifi_dataset_public.csv` to
`wifi_dataset.csv` and run the attack script.

## Running it

```bash
pip install pandas scikit-learn numpy
python collect_wifi_simple.py      # set LABEL / LOCATION inside for each session
python wifi_2fa_spoof_attack.py
```
