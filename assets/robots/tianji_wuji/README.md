# Tianji + Wuji source assets

This directory contains the Tianji CCS 696 robot and Wuji hand source assets
needed by the standalone Tianji + Wuji training environment. Both left and
right sides are included. The single task uses one right arm and hand; the
dual task uses both arms and hands. No sibling DexTele or sim_infra checkout,
external symlink, or asset download is required for these robot assets.

The package preserves the DexTele root layout expected by the environment:

```text
robots/tianji/
  tianji_marvin_CCS_real_limit_wuji_sim_infra.xml  # canonical combined model
  scene_tianji_wuji_sim_infra.xml                # combined demonstration scene
  *.xml                                        # other DexTele Tianji scenes/models
  tianji_marvin_CCS.urdf                        # native Tianji robot description
  meshes_sim_infra/                             # 17 Tianji STL meshes
  wuji_hand_sim_infra/
    left.xml, right.xml                         # native Wuji MJCF models
    urdf/left.urdf, urdf/right.urdf              # native Wuji URDF models
    meshes/left/, meshes/right/                 # 52 Wuji STL meshes
```

The canonical combined model and its collision variant are copied byte for
byte from DexTele. Some other XML files and the native Tianji URDF have only
mesh paths rewritten so every reference resolves within this directory. All
mesh files are unmodified. The original symlink targets have been copied as
regular files; there are no symlinks in this package.

`MANIFEST.json` records the source repository, original relative path, original
SHA-256, bundled SHA-256, byte size, and modifications for all 86 asset files
(23,606,721 bytes, approximately 22.5 MiB). The largest file is 2,577,384 bytes.
Its inventory excludes this README and the manifest itself. Each hash describes
the actual local source file, including any changes beyond its repository HEAD.

Sources used:

- DexTele: `robots/tianji/*.xml`, local checkout HEAD
  `fbc3d487d1f975954fd90b75185ed9ff18da9388`.
- sim_infra: `assets/robots/tianji_marvin_ccs_696/meshes/`,
  `assets/robots/tianji_marvin_ccs_696/tianji_marvin_CCS.urdf`, and
  `assets/robots/wuji_hand/`, local checkout HEAD
  `39104253416b1203a041812159cbbc515fe01809`.

The bundle includes all 12 Tianji MJCF source/scene XMLs supplied in the
DexTele robot directory, the native Tianji URDF, both native Wuji MJCFs, both
native Wuji URDFs, and every mesh referenced by those files. The 647 file
references across the 17 XML/URDF files were checked for existence and package
containment. Generated `sim_infra_spec_*.xml`, unrelated Robotiq models,
DexTele application code, installed environments, converted USD caches, logs,
and training outputs are not part of this robot asset package.

The original robot assets remain the property of their respective upstream
owners. No license file was present at the source repository roots or inside
the copied asset directories. This package does not grant a new license or
change the upstream assets' ownership or terms.

This makes the Tianji + Wuji **robot asset dependency** self-contained. Running
training still requires the simulator and Python dependencies described in the
repository's training documentation.
