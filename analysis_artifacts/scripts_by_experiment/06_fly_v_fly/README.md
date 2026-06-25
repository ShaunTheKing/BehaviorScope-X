# Fly-v-Fly Adaptation Runner

Use this folder for the final focused Fly-v-Fly short-bout adaptation.

The runner forwards the final manuscript settings to the independent
Fly-v-Fly adaptation suite: attention head, hidden dimension 256, windows
`w16s8` and `w8s4`, seed 42, batch 64, square-root inverse class weighting,
class-weight clamp 1.2, and random training sampler.

Plan first:

```bash
python run_fly_v_fly_adaptation.py plan
```

Run training and evaluation using the final cached NPZ/feature data:

```bash
python run_fly_v_fly_adaptation.py run-all --skip_completed --keep_going
```

Pass `--rebuild_npz` only if you intentionally want to regenerate the Fly
train and held-out NPZ/cache inside the selected suite root.
