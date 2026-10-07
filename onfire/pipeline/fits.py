"""Multi-Extension FITS persistence for :class:`ReductionState` checkpoints.

Checkpoint layout
-----------------
The file is a Multi-Extension FITS (MEF) container:

* ``PRIMARY`` -- empty primary HDU.
* one ``ImageHDU`` per NumPy array product, carrying ``EXTNAME`` (the product
  name), ``ONFNAME`` (the exact dictionary key) and ``ONFDTYPE`` (the original
  dtype, so boolean masks and byte order round-trip exactly).
* one ``BinTableHDU`` per Astropy table product.
* ``ONFIREMETA`` -- a single-row table holding the step ordinal, the step name,
  the Frame Table as fixed-width text, and the Parameter Config, scalar
  metadata, Order Mask, product provenance, and Stale marks as JSON.

Plotly figures are intentionally not serialized here; they are written as
standalone HTML by :meth:`ReductionState.save_plots`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.table import Table as AstropyTable

from onfire.setup.table import FrameTable

from onfire.pipeline.state import OrderMask, Provenance, ReductionState, StaleMarks

_META_EXTNAME = "ONFIREMETA"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialize {type(obj).__name__} to JSON")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, sort_keys=True)


def _optional_json(row: Any, names: list[str], column: str) -> Any:
    """Read a JSON column, tolerating checkpoints written before it existed."""
    return json.loads(str(row[column])) if column in names else {}


def _product_hdu(name: str, value: Any) -> fits.hdu.base.ExtensionHDU:
    if isinstance(value, np.ndarray):
        dtype = value.dtype
        hdu = fits.ImageHDU(data=value.astype(np.uint8) if dtype == bool else value)
        hdu.header["ONFDTYPE"] = dtype.str
    elif isinstance(value, AstropyTable):
        hdu = fits.BinTableHDU(data=value)
    else:
        raise TypeError(
            f"data product {name!r} has unsupported type {type(value).__name__}; "
            "expected a NumPy array or Astropy Table"
        )
    hdu.header["ONFNAME"] = name
    hdu.name = name
    return hdu


def _meta_hdu(state: ReductionState) -> fits.BinTableHDU:
    columns = {
        "step": np.array([state.step], dtype=np.int32),
        "step_name": [state.step_name],
        "source": [state.source],
        "frames_source": [state.frames.source],
        "frames_text": [state.frames.to_string()],
        "frames_paths_json": [_dumps([str(f.path) for f in state.frames.frames])],
        "config_json": [_dumps(state.config)],
        "meta_json": [_dumps(state.meta)],
        "order_mask_json": [_dumps(state.order_mask.to_dict())],
        "provenance_json": [_dumps(state.provenance.products)],
        "stale_json": [_dumps(state.stale.to_json())],
    }
    hdu = fits.BinTableHDU(AstropyTable(columns))
    hdu.name = _META_EXTNAME
    return hdu


def write_checkpoint(state: ReductionState, path: str | Path) -> Path:
    """Write ``state`` to a Multi-Extension FITS checkpoint at ``path``."""
    path = Path(path)
    hdus: list[fits.hdu.base.ExtensionHDU] = [fits.PrimaryHDU()]
    for name, value in state.products.items():
        hdus.append(_product_hdu(name, value))
    hdus.append(_meta_hdu(state))
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def _read_product(hdu: fits.hdu.base.ExtensionHDU) -> Any:
    if isinstance(hdu, fits.BinTableHDU):
        return AstropyTable(hdu.data)
    array = np.asarray(hdu.data)
    dtype = hdu.header.get("ONFDTYPE")
    if dtype:
        array = array.astype(np.dtype(dtype))
    return array


def read_checkpoint(path: str | Path) -> ReductionState:
    """Reload a :class:`ReductionState` from a checkpoint written by :func:`write_checkpoint`."""
    with fits.open(Path(path), memmap=False) as hdul:
        meta_hdu = hdul[_META_EXTNAME]
        row = meta_hdu.data[0]
        columns = meta_hdu.columns.names

        products: dict[str, Any] = {}
        for hdu in hdul:
            name = hdu.header.get("ONFNAME")
            if name is None:
                continue
            products[str(name)] = _read_product(hdu)

        frames = FrameTable.from_string(str(row["frames_text"]))
        frames.source = str(row["frames_source"])
        paths = json.loads(str(row["frames_paths_json"]))
        if len(paths) == len(frames.frames):
            for frame, frame_path in zip(frames.frames, paths):
                frame.path = Path(frame_path)

        return ReductionState(
            frames=frames,
            config=json.loads(str(row["config_json"])),
            products=products,
            meta=json.loads(str(row["meta_json"])),
            step=int(row["step"]),
            order_mask=OrderMask.from_dict(json.loads(str(row["order_mask_json"]))),
            source=str(row["source"]),
            provenance=Provenance(
                products={
                    key: int(step)
                    for key, step in _optional_json(
                        row, columns, "provenance_json"
                    ).items()
                }
            ),
            stale=StaleMarks.from_json(_optional_json(row, columns, "stale_json")),
        )
