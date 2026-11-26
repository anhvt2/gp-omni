# Example geometry images

The PNG files are rendered from the bundled pump mesh and the FEM reference
geometry functions. FEM surfaces are triangulated by Gmsh. Shading represents
surface orientation. The captions identify reference geometries and view scales.

Regenerate all images from the repository root.

```bash
uv run --locked --with 'matplotlib==3.11.2' --with 'gmsh==4.15.2' --with 'scikit-fem==12.0.2' python examples/render_geometry.py
```

The pump rendering derives from the bundled GPL-3.0 case and retains its license
and attribution. See the [case provenance](../cfd/pump_case/README.md) and
[license](../cfd/pump_case/LICENSE).
