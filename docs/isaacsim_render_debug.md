# Isaac Sim RTX Startup Debug Report

Date: 2026-10-18

## Environment

- Ubuntu 26.04.1, glibc 2.43.
- NVIDIA RTX 3060, 12 GB VRAM; driver 595.99.02.
- AMD Ryzen 7 5800, 16 logical CPUs; approximately 32 GB RAM.
- Repository `.venv_isaacsim`, Python 3.11.17, Isaac Sim 5.1.

## Reproduction

Run from the repository root with no other Isaac Sim process running:

```bash
env -u LD_LIBRARY_PATH -u DISPLAY \
  OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
  timeout --kill-after=15s 3m .venv_isaacsim/bin/python -c \
  'from isaacsim import SimulationApp; import os; app = SimulationApp({"headless": True}); print("SIMULATION_APP_OK", flush=True); os._exit(0)'
```

Observed: SIGSEGV, exit status 139, before the success marker. No project task,
checkpoint, or policy is loaded. The failure also reproduces with
`AppLauncher(headless=True, enable_cameras=True)`.

## Native Debugger Evidence

GDB stops inside `omni.hydra.rtx`'s `librtx.scenedb.plugin.so`. The call chain
passes through `libcarb.scenerenderer-rtx.plugin.so` and Hydra engine creation.
The library is stripped, so internal function names are unavailable.

```text
rbp = 0
r13 = 0x10000
faulting instruction: lock add %r13d,0x3380(%rbp)
effective address: 0x3380
```

This is an internal null-pointer dereference during renderer initialization.
The caller loads the null pointer from an internal object's offset `0x4f0`.
The earlier initialization failure that left this member null is not yet known.
System `libstdc++.so.6` is loaded, not Conda's C++ runtime.

## Controlled Checks

| Check | Result |
| --- | --- |
| Non-camera Kit load | Success marker observed |
| Gym registration | Success marker observed |
| Eight environments, ten simulation steps | Object counts and no-NaN assertions pass; clean process shutdown not established |
| Camera boot without project/model | SIGSEGV |
| Remove compatibility library path, X11 forwarding and Conda-related environment entries; restrict Vulkan to NVIDIA | SIGSEGV |
| Direct NVIDIA SimulationApp, bypassing Isaac Lab | SIGSEGV |
| Set TBB_MALLOC_DISABLE_REPLACEMENT=1 | SIGSEGV |
| Disable Fabric Scene Delegate and multi-GPU, request one task worker | SIGSEGV |

The TBB setting was an experimental allocator hypothesis, not a verified fix.
Standalone `ldd` reports missing Kit libraries outside the running application;
this does not establish missing runtime dependencies because Kit loads its own
libraries dynamically.

## Logs

Paths relative to the repository root:

- `logs/debug-camera-clean.log`
- `logs/debug-camera-gdb.log`
- `logs/debug-simulation-app.log`
- `logs/debug-rtx-pointer.log`
- `logs/debug-rtx-caller.log`
- `logs/debug-no-tbb-proxy.log`
- `logs/debug-camera-compat.log`
- `logs/video-preview.log`

## Compatibility Boundary and Next Steps

[NVIDIA Isaac Sim 5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
list Ubuntu 22.04/24.04 and tested Linux driver 580.65.06. They list RTX 4080
and 16 GB VRAM as the minimum GPU configuration. This host is outside that
published configuration, but the evidence does not identify which difference
causes this particular null pointer. There is no established out-of-memory
diagnosis.

The next discriminating check is the minimal reproduction in an official 5.1
container or on Ubuntu 24.04, retaining the same GPU and driver when possible.
A container replaces userspace libraries but still uses the host GPU driver.
If that succeeds, host userspace compatibility becomes the leading cause;
if it fails, investigate driver compatibility and upstream renderer defects.
No Docker or Podman executable was found during this session.

Do not downgrade system glibc, replace packaged RTX binaries, or change the
driver merely to hide the crash. Those changes need a separate controlled
environment and rollback plan. No driver or system-library changes were made
during this investigation.