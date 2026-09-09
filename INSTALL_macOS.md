# Installing Onyx on macOS

Open Terminal, go to the pack folder, and run one line:

```bash
cd ~/ComfyUI/custom_nodes/Onyx
bash install_all_custom_node_requirements.sh
```

Adjust the first line if your ComfyUI lives elsewhere. Tip: type `cd ` with a
trailing space, then drag the folder from Finder into the Terminal window — it
fills in the path for you.

Then restart ComfyUI.

Running it through `bash` is deliberate. It sidesteps two things that trip
people up with downloaded scripts: the file does not need the executable
permission, and Gatekeeper does not block it the way it blocks a
double-clicked one.

## Options

```bash
bash install_all_custom_node_requirements.sh --dry-run   # show, install nothing
bash install_all_custom_node_requirements.sh --force     # reinstall Onyx's own packages
```

`--force` uses `--no-deps` on purpose. Reinstalling dependencies as well would
also replace numpy and protobuf, and a numpy swapped underneath the torch that
ComfyUI already installed leaves you with a ComfyUI that no longer starts.

## Which Python it installs into

In order: the virtualenv you have active, then a `venv` or `.venv` next to
ComfyUI, then `python3` from your PATH.

If it lands on `/usr/bin/python3` — the Python that ships with macOS — the
script stops and asks before going further. Installing there can break system
tools, and modern pip refuses outright with `externally-managed-environment`.
Activate the environment you start ComfyUI from first:

```bash
source ~/ComfyUI/venv/bin/activate
```

## What gets installed

Required — the pack does not load without these:

- `google-genai` — Image & Video Edit AIO
- `opencv-python` — Eye Detailer

Optional — each only breaks its own node, and only when you run it:

`mediapipe`, `colour-science`, `PyWavelets`, `imageio`, `fal-client`,
`gallery-dl`

The script ends by printing which of the two groups is present. A clean pip run
is not the same thing as a working pack, which is why that check is there:

```
Checking the imports this pack needs at load time:
  OK       google.genai
  MISSING  cv2   <- the pack will not load
```
