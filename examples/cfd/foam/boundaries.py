"""Boundary conditions by physical meaning, shared by every solver set-up."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class boundary:
    kind: str
    options: dict = field(default_factory=dict)

    def mesh_entry(self) -> dict[str, object]:
        """The ``constant/polyMesh/boundary`` entry for a mesh written by this package."""
        simple = {"symmetry": "symmetry", "empty": "empty", "wedge": "wedge"}
        if self.kind in simple:
            return {"type": simple[self.kind]}
        if self.kind in ("wall", "moving_wall", "rotating_wall", "slip"):
            return {"type": "wall" if self.kind != "slip" else "patch"}
        if self.kind == "periodic":
            # matching faces: non-conformal coupling of twisted periodic faces makes a
            # rotating zone create swirl out of nothing
            return {
                "type": "cyclic",
                "neighbourPatch": self.options["neighbour"],
                "transform": "rotational",
                "rotationAxis": "(0 0 1)",
                "rotationCentre": "(0 0 0)",
            }
        if self.kind == "ami":
            return {
                "type": "cyclicAMI",
                "neighbourPatch": self.options["neighbour"],
                "transform": "noOrdering",
                "lowWeightCorrection": 0.2,
            }
        if self.kind == "acmi":
            return {
                "type": "cyclicACMI",
                "neighbourPatch": self.options["neighbour"],
                "nonOverlapPatch": self.options["blockage"],
                "transform": "noOrdering",
            }
        if self.kind == "blockage":
            return {"type": "wall", "copy_of": self.options["couple"]}
        return {"type": "patch"}


def wall(temperature: float | None = None) -> boundary:
    """A no-slip wall, adiabatic unless a temperature is given."""
    return boundary("wall", {"temperature": temperature})


def moving_wall() -> boundary:
    """A wall that moves with the mesh."""
    return boundary("moving_wall")


def rotating_wall(omega: float, axis=(0, 0, 1), origin=(0, 0, 0)) -> boundary:
    return boundary("rotating_wall", {"omega": omega, "axis": axis, "origin": origin})


def slip() -> boundary:
    return boundary("slip")


def symmetry() -> boundary:
    return boundary("symmetry")


def empty() -> boundary:
    return boundary("empty")


def wedge() -> boundary:
    return boundary("wedge")


def velocity_inlet(velocity, temperature: float | None = None, **scalars) -> boundary:
    return boundary(
        "velocity_inlet", {"velocity": velocity, "temperature": temperature, "scalars": scalars}
    )


def flow_inlet(flow: float, temperature: float | None = None, **scalars) -> boundary:
    """Volumetric flow rate for incompressible solvers, mass flow rate otherwise."""
    return boundary("flow_inlet", {"flow": flow, "temperature": temperature, "scalars": scalars})


def swirl_inlet(radial: float, tangential: float, axial: float = 0.0) -> boundary:
    """Velocity given in cylindrical components about the z axis."""
    return boundary("swirl_inlet", {"radial": radial, "tangential": tangential, "axial": axial})


def total_pressure_inlet(
    pressure: float, temperature: float | None = None, **scalars
) -> boundary:
    return boundary(
        "total_pressure_inlet",
        {"pressure": pressure, "temperature": temperature, "scalars": scalars},
    )


def wind_profile_inlet(speed: float, height: float, roughness: float) -> boundary:
    """Atmospheric boundary layer flowing along +x, ``speed`` at ``height`` above z = 0."""
    return boundary(
        "wind_profile_inlet", {"speed": speed, "height": height, "roughness": roughness}
    )


def supersonic_inlet(velocity, pressure: float, temperature: float) -> boundary:
    return boundary(
        "supersonic_inlet",
        {"velocity": velocity, "pressure": pressure, "temperature": temperature},
    )


def pressure_outlet(pressure: float = 0.0) -> boundary:
    return boundary("pressure_outlet", {"pressure": pressure})


def open_outlet() -> boundary:
    """Zero gradient for every variable, for supersonic outflow."""
    return boundary("open_outlet")


def freestream(velocity, pressure: float = 0.0, temperature: float | None = None) -> boundary:
    return boundary(
        "freestream",
        {"velocity": velocity, "pressure": pressure, "temperature": temperature},
    )


def periodic(neighbour: str) -> boundary:
    """Rotationally periodic pair about the z axis, with faces listed in matching order."""
    return boundary("periodic", {"neighbour": neighbour})


def interface(neighbour: str) -> boundary:
    """Non-conformal interface, for sliding meshes."""
    return boundary("ami", {"neighbour": neighbour})


def partial_interface(neighbour: str, blockage: str) -> boundary:
    return boundary("acmi", {"neighbour": neighbour, "blockage": blockage})


def blockage(couple: str, moving: bool = False) -> boundary:
    """The wall left where a partial-overlap interface has no partner."""
    return boundary("blockage", {"couple": couple, "moving": moving})
