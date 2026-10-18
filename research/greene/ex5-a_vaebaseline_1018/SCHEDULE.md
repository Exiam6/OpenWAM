# One-shot preflight schedule

The CPU-only schedule_preflight.py waits until the already registered06:00UTC
window, honors every pause path, then invokes the unchanged frozen preflight on
GPU5. The registration hash must remain unchanged. The frozen parent rechecks the
complete code/assets, remaining allocation and actual idle capacity; scheduling
is not GPU admission. A refusal or failure is retained and is not retried.
The schedule does not launch full training, bypass visual review or reserve an
occupied GPU. Its own source is separately committed and copied to window-code;
the existing GPU runtime and registration are not edited.
