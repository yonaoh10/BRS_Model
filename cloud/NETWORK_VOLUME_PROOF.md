# Host-independence proof — network-volume migration (2026-09-16)

The migration's whole point is the FAILURE MODE, not the feature: a pod-local
volume is pinned to one physical host, so a stopped/destroyed pod can only come
back where that host has a free GPU — the "not enough free GPUs on the host
machine" (HTTP 500) wall that stranded us for hours. This is the live proof
that a network volume removes it.

## Baseline
- Pod `x7ez2kuwmy5cnc` on machine **`ygmojncm58qt`**, network volume
  `7q1qj337op` (80 GB, EU-RO-1). Served gemma-3-27b and judged REAL002 (87.5).

## Step A — stop then start (the host-pinning it does NOT fix)
- `down` → pod EXITED, GPU billing ended.
- `up` (start the stopped pod) → **HTTP 500 "There are not enough free GPUs on
  the host machine to start this pod."** — the exact wall. A *stopped* pod is
  still host-pinned even with a network volume; starting it needs a free GPU on
  its original machine. So stop/start is NOT the escape.

## Step B — destroy then recreate (the escape)
- `destroy` (no `--volume`) → pod deleted, **volume `7q1qj337op` PERSISTS**
  (verified via GET /networkvolumes).
- `up` → creates a FRESH pod. First DC attempt hit momentary capacity; on the
  next it placed pod `rxxbb7t7awjwl7` on machine **`o5d2qxwjenu4`** — a
  DIFFERENT physical host than the baseline `ygmojncm58qt`.
- SSH into the new pod: `/workspace/models/gemma-w4a16` still holds **19 GB, 4
  safetensors shards — NO re-download**. The whole volume (incl. the serve
  script) came across intact.

## Conclusion
- Different machine id, same volume, weights intact: **host-independence
  proven** — not a same-host coincidence.
- Honest scope: a network volume is DATA-CENTRE-scoped, so recreate still needs
  a free GPU *somewhere in that DC* (EU-RO-1 here). That is a far weaker
  constraint than one saturated host — the class of failure (a single host
  full → stranded) is gone; a whole DC being momentarily out of every 24 GB+
  card is the remaining, transient limit, and `up` walks DCs on the FIRST
  creation to pick one with capacity.
- Idle state: the pod is left EXITED (GPU billing stopped) per instruction;
  the volume is kept ($0.07/GB-mo → $5.60/mo). For a network-volume setup,
  `destroy` (which keeps the volume) is the more robust idle state than a
  host-pinned stopped pod, since a fresh `up` is not host-pinned.
