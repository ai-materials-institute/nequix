#!/usr/bin/env python3
"""Nequix MCP server
Exposes energy / force / stress predictions from Nequix.
"""

from __future__ import annotations

import argparse
import platform
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read as ase_read

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_MODELS_DIR = _REPO_ROOT / "models"

# Safety check to prevent local ./mcp folder from shadowing this mcp
_normalized_sys_path = []
for _entry in sys.path:
    candidate = str(Path(_entry or ".").resolve())
    if candidate != str(_REPO_ROOT):
        _normalized_sys_path.append(_entry)
sys.path = _normalized_sys_path

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations
except Exception as exc:
    raise SystemExit(
        "Failed to import MCP SDK. Install the dependencies and then rerun with Python version >= 3.10."
    ) from exc

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nequix.calculator import NequixCalculator

READ_ONLY_TOOL_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)

DEFAULT_MODEL_NAME = "nequix-mp-1"
DEFAULT_BACKEND = "jax"

SERVER_INSTRUCTIONS = (
    "You are connected to a Nequix inference server for crystal structures. "
    "When the user asks for energies/forces/stress and CIF input is available, "
    "call `predict_structure` immediately. "
    "Use `cif_path` for server-local files and `cif_text` for uploaded/raw CIF text."
)

AUTO_ROUTING_GUIDE = """# Nequix MCP Routing Guide

- Primary tool: `predict_structure`
- Use `cif_path` for paths that exist on this server host
- Use `cif_text` for remote/user-uploaded structures
- Defaults: `model_name=nequix-mp-1`, `backend=jax`, `use_kernel=true`
- Call `list_models` when a user asks which model checkpoints are available
"""


@dataclass
class LoadedCalculator:
    model_name: str
    backend: str
    use_kernel: bool
    calculator: NequixCalculator
    loaded_at_epoch_s: float


class InferenceRuntime:
    """Caches Nequix calculators + performs CIF inference for MCP."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calculators: dict[tuple[str, str, bool], LoadedCalculator] = {}

    def _resolve_model_path(self, model_name: str) -> Path | None:
        model_candidates = [
            _MODELS_DIR / f"{model_name}.nqx",
            _MODELS_DIR / f"{model_name}.pt",
        ]
        for candidate in model_candidates:
            if candidate.exists():
                return candidate
        return None

    def list_model_availability(self) -> dict[str, Any]:
        model_names = sorted(NequixCalculator.URLS.keys())
        available = []
        missing = []

        for name in model_names:
            path = self._resolve_model_path(name)
            if path is not None:
                available.append({"model_name": name, "local_path": str(path)})
            else:
                missing.append(name)

        return {
            "repo_models_dir": str(_MODELS_DIR),
            "available_local": available,
            "missing_local": missing,
            "all_supported_model_names": model_names,
        }

    def _get_calculator(
        self, model_name: str, backend: str, use_kernel: bool
    ) -> tuple[NequixCalculator, bool, float]:
        key = (model_name, backend, use_kernel)
        cached = self._calculators.get(key)
        if cached is not None:
            return cached.calculator, False, 0.0

        with self._lock:
            cached = self._calculators.get(key)
            if cached is not None:
                return cached.calculator, False, 0.0

            t0 = time.perf_counter()
            local_model_path = self._resolve_model_path(model_name)

            calculator_kwargs: dict[str, Any] = {
                "model_name": model_name,
                "backend": backend,
                "use_kernel": use_kernel,
            }
            if local_model_path is not None:
                calculator_kwargs["model_path"] = str(local_model_path)

            calculator = NequixCalculator(**calculator_kwargs)
            t1 = time.perf_counter()

            self._calculators[key] = LoadedCalculator(
                model_name=model_name,
                backend=backend,
                use_kernel=use_kernel,
                calculator=calculator,
                loaded_at_epoch_s=time.time(),
            )
            return calculator, True, round(t1 - t0, 6)

    @staticmethod
    def _atoms_summary(atoms: Any) -> dict[str, Any]:
        symbols = atoms.get_chemical_symbols()
        unique_species = sorted(set(symbols))
        pbc = [bool(v) for v in np.asarray(atoms.pbc).tolist()]
        return {
            "natoms": len(atoms),
            "chemical_formula": atoms.get_chemical_formula(),
            "unique_species": unique_species,
            "cell_angstrom": np.asarray(atoms.cell).round(8).tolist(),
            "pbc": pbc,
        }

    def predict_atoms(
        self,
        atoms: Any,
        model_name: str = DEFAULT_MODEL_NAME,
        backend: str = DEFAULT_BACKEND,
        use_kernel: bool = True,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        calculator, loaded_now, load_time_s = self._get_calculator(
            model_name=model_name,
            backend=backend,
            use_kernel=use_kernel,
        )
        t_after_load = time.perf_counter()

        atoms_copy = atoms.copy()
        atoms_copy.calc = calculator

        energy = float(atoms_copy.get_potential_energy())
        forces = np.asarray(atoms_copy.get_forces())
        stress = atoms_copy.get_stress(voigt=True)
        stress_out = None if stress is None else np.asarray(stress).tolist()
        t1 = time.perf_counter()

        return {
            "input": {
                "model_name": model_name,
                "backend": backend,
                "use_kernel": use_kernel,
            },
            "structure": self._atoms_summary(atoms_copy),
            "prediction": {
                "energy_eV": energy,
                "forces_eV_per_A": forces.tolist(),
                "max_force_eV_per_A": float(np.linalg.norm(forces, axis=1).max(initial=0.0)),
                "stress_voigt_eV_per_A3": stress_out,
            },
            "timings_sec": {
                "model_load": load_time_s,
                "inference": round(t1 - t_after_load, 6),
                "total": round(t1 - t0, 6),
            },
            "notes": {
                "calculator_loaded_now": loaded_now,
                "model_cache_key": f"{model_name}:{backend}:kernel={str(use_kernel).lower()}",
            },
        }

    def predict_cif_path(
        self,
        cif_path: str,
        model_name: str = DEFAULT_MODEL_NAME,
        backend: str = DEFAULT_BACKEND,
        use_kernel: bool = True,
    ) -> dict[str, Any]:
        abs_cif_path = str(Path(cif_path).expanduser().resolve())
        path = Path(abs_cif_path)
        if not path.exists():
            raise FileNotFoundError(f"CIF file not found: {abs_cif_path}")

        atoms = ase_read(abs_cif_path)
        out = self.predict_atoms(
            atoms,
            model_name=model_name,
            backend=backend,
            use_kernel=use_kernel,
        )
        out["input"]["cif_path"] = abs_cif_path
        out["input"]["source"] = "cif_path"
        return out

    def predict_cif_text(
        self,
        cif_text: str,
        filename_hint: str = "input.cif",
        model_name: str = DEFAULT_MODEL_NAME,
        backend: str = DEFAULT_BACKEND,
        use_kernel: bool = True,
    ) -> dict[str, Any]:
        safe_name = Path(filename_hint).name
        if not safe_name.lower().endswith(".cif"):
            safe_name = f"{safe_name}.cif"

        with tempfile.TemporaryDirectory(prefix="nequix_mcp_") as tmp_dir:
            cif_file = Path(tmp_dir) / safe_name
            cif_file.write_text(cif_text, encoding="utf-8")
            out = self.predict_cif_path(
                str(cif_file),
                model_name=model_name,
                backend=backend,
                use_kernel=use_kernel,
            )
            out["input"]["source"] = "inline_cif_text"
            out["input"]["filename_hint"] = safe_name
            out["input"].pop("cif_path", None)
            return out


def create_server(
    *,
    host: str,
    port: int,
    mount_path: str,
    streamable_http_path: str,
) -> tuple[FastMCP, InferenceRuntime]:
    runtime = InferenceRuntime()

    mcp = FastMCP(
        name="nequix-inference",
        instructions=SERVER_INSTRUCTIONS,
        host=host,
        port=port,
        mount_path=mount_path,
        streamable_http_path=streamable_http_path,
    )

    @mcp.tool(
        name="predict_cif_path",
        description="Run Nequix inference from a server-local CIF path.",
        annotations=READ_ONLY_TOOL_ANNOTATIONS,
        structured_output=True,
    )
    def predict_cif_path(
        cif_path: str,
        model_name: str = DEFAULT_MODEL_NAME,
        backend: str = DEFAULT_BACKEND,
        use_kernel: bool = True,
    ) -> dict[str, Any]:
        return runtime.predict_cif_path(
            cif_path=cif_path,
            model_name=model_name,
            backend=backend,
            use_kernel=use_kernel,
        )

    @mcp.tool(
        name="predict_cif_text",
        description="Run Nequix inference from raw CIF text.",
        annotations=READ_ONLY_TOOL_ANNOTATIONS,
        structured_output=True,
    )
    def predict_cif_text(
        cif_text: str,
        filename_hint: str = "input.cif",
        model_name: str = DEFAULT_MODEL_NAME,
        backend: str = DEFAULT_BACKEND,
        use_kernel: bool = True,
    ) -> dict[str, Any]:
        return runtime.predict_cif_text(
            cif_text=cif_text,
            filename_hint=filename_hint,
            model_name=model_name,
            backend=backend,
            use_kernel=use_kernel,
        )

    @mcp.tool(
        name="predict_structure",
        description=(
            "Main Nequix entry-point. Provide exactly one of `cif_path` or `cif_text`. "
            "Returns energy/forces/stress with structure summary and timings."
        ),
        annotations=READ_ONLY_TOOL_ANNOTATIONS,
        structured_output=True,
    )
    def predict_structure(
        cif_path: str | None = None,
        cif_text: str | None = None,
        filename_hint: str = "input.cif",
        model_name: str = DEFAULT_MODEL_NAME,
        backend: str = DEFAULT_BACKEND,
        use_kernel: bool = True,
    ) -> dict[str, Any]:
        if bool(cif_path) == bool(cif_text):
            raise ValueError("Provide only one of `cif_path` or `cif_text`.")
        if cif_path is not None:
            return runtime.predict_cif_path(
                cif_path=cif_path,
                model_name=model_name,
                backend=backend,
                use_kernel=use_kernel,
            )
        return runtime.predict_cif_text(
            cif_text=cif_text or "",
            filename_hint=filename_hint,
            model_name=model_name,
            backend=backend,
            use_kernel=use_kernel,
        )

    @mcp.tool(
        name="list_models",
        description="List supported model names and whether each model file is present locally.",
        annotations=READ_ONLY_TOOL_ANNOTATIONS,
        structured_output=True,
    )
    def list_models() -> dict[str, Any]:
        return runtime.list_model_availability()

    @mcp.tool(
        name="server_status",
        description="Return runtime status for the Nequix MCP server.",
        annotations=READ_ONLY_TOOL_ANNOTATIONS,
        structured_output=True,
    )
    def server_status() -> dict[str, Any]:
        return {
            "server_name": "nequix-inference",
            "python_version": sys.version,
            "platform": platform.platform(),
            "repo_root": str(_REPO_ROOT),
            "models_dir": str(_MODELS_DIR),
            "cached_calculators": len(runtime._calculators),
            "streamable_http_url_example": f"http://{host}:{port}{streamable_http_path}",
            "primary_tool": "predict_structure",
            "routing_guide_resource_uri": "guide://nequix/auto-tool-routing",
        }

    @mcp.resource(
        "guide://nequix/auto-tool-routing",
        name="nequix-routing-guide",
        title="Nequix Tool Routing",
        description="Guide for picking the right Nequix inference tool.",
        mime_type="text/markdown",
    )
    def nequix_routing_guide() -> str:
        return AUTO_ROUTING_GUIDE

    @mcp.prompt(
        name="predict_energy_forces_from_path",
        title="Predict Energy/Forces From CIF Path",
        description="Prompt template that routes to the Nequix primary tool.",
    )
    def predict_energy_forces_from_path(cif_path: str) -> str:
        return (
            "Call `predict_structure` with "
            f"cif_path='{cif_path}', model_name='{DEFAULT_MODEL_NAME}', "
            f"backend='{DEFAULT_BACKEND}', use_kernel=true. "
            "Then summarize energy_eV, max_force_eV_per_A, and stress_voigt_eV_per_A3."
        )

    return mcp, runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nequix MCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="streamable-http",
        help="MCP transport mode.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind for HTTP transports.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8766,
        help="Port to bind for HTTP transports.",
    )
    parser.add_argument(
        "--mount-path",
        default="/",
        help="Mount path for HTTP transports.",
    )
    parser.add_argument(
        "--streamable-http-path",
        default="/",
        help="Path for streamable HTTP endpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mcp, _ = create_server(
        host=args.host,
        port=args.port,
        mount_path=args.mount_path,
        streamable_http_path=args.streamable_http_path,
    )

    if args.transport == "streamable-http":
        print(
            "[nequix-mcp] streamable endpoint: "
            f"http://{args.host}:{args.port}{args.streamable_http_path}",
            file=sys.stderr,
        )

    mcp.run(transport=args.transport, mount_path=args.mount_path)


if __name__ == "__main__":
    main()
