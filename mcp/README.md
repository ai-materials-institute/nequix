# Nequix MCP

This folder contains a Model Context Protocol (MCP) server for Nequix.

## Exposed tools

- `predict_structure` (primary): predicts from either `cif_path` or `cif_text`
- `predict_cif_path`: predicts from a server-local CIF file path
- `predict_cif_text`: predicts from raw CIF text
- `list_models`: shows supported model names + local availability
- `server_status`: reports runtime/server state

Outputs include:
- `energy_eV`
- `forces_eV_per_A`
- `max_force_eV_per_A`
- `stress_voigt_eV_per_A3`

## Run

From the repo root run:

```bash
cd <path-to-nequix-repo>
uv run python mcp/server.py --transport streamable-http --host 127.0.0.1 --port 8766
```

## Connect from MCP clients

Use streamable HTTP URL:

- `http://127.0.0.1:8766`

For stdio clients, use command in a separate terminal:

- `uv --directory <path-to-nequix-repo> run python mcp/server.py --transport stdio`

## Add Nequix to your MCP

Either add Nequix manually to your `config.toml` file, or:

[Codex] `codex mcp add nequix -- \ uv --directory /<path-to-nequix-repo> \ run python mcp/server.py --transport stdio `



## Misc

- The server prefers model files in `models/` in the repo
- If a model file is missing locally, Nequix might download it from its configured URL.
- For remote clients use `cif_text`, probably easier.


