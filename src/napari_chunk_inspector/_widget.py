"""
Chunk Inspector Widget
======================
Visualize dask array chunk boundaries and indices for OME-Zarr layers.
"""

from __future__ import annotations

import warnings
from typing import Optional

import dask.array as da
import numpy as np
from napari.layers import Image
from napari.viewer import Viewer
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def _get_dask_array_at_level(layer: Image, level: int) -> Optional[da.Array]:
    """Return the dask array at the given scale level.

    For single-scale layers the level argument is ignored.
    Handles da.Array, xarray.DataArray, and zarr.Array.
    """
    arr = layer.data[level] if hasattr(layer, "data_level") else layer.data

    if isinstance(arr, da.Array):
        return arr

    try:
        import xarray as xr  # type: ignore[import-untyped]
        if isinstance(arr, xr.DataArray) and isinstance(arr.data, da.Array):
            return arr.data
    except ImportError:
        pass

    try:
        import zarr
        if isinstance(arr, zarr.Array):
            return da.from_zarr(arr)
    except ImportError:
        pass

    return None


def _world_to_chunk_index(
    data_coords: np.ndarray,
    chunks: tuple[tuple[int, ...], ...],
) -> Optional[tuple[tuple[int, ...], tuple[slice, ...], tuple[int, ...]]]:
    """Return (chunk_index, chunk_slices, chunk_shape) or None if out of bounds."""
    ndim = len(chunks)
    if len(data_coords) < ndim:
        return None

    chunk_index, chunk_slices, chunk_shape = [], [], []

    for coord, dim_chunks in zip(data_coords[-ndim:], chunks):
        c = int(np.floor(coord))
        if c < 0 or c >= sum(dim_chunks):
            return None
        cumsum = 0
        for ci, chunk_size in enumerate(dim_chunks):
            if cumsum + chunk_size > c:
                chunk_index.append(ci)
                chunk_slices.append(slice(cumsum, cumsum + chunk_size))
                chunk_shape.append(chunk_size)
                break
            cumsum += chunk_size
        else:
            return None

    return tuple(chunk_index), tuple(chunk_slices), tuple(chunk_shape)


class ChunkInspector(QWidget):
    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self._updating_layers = False

        self._build_ui()
        self.viewer.mouse_move_callbacks.append(self._on_mouse_move)
        self.viewer.layers.events.inserted.connect(self._on_layers_change)
        self.viewer.layers.events.removed.connect(self._on_layers_change)
        self._refresh_layer_combo()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("Chunk Inspector")
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        title.setFont(font)
        layout.addWidget(title)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Layer:"))
        self._layer_combo = QComboBox()
        self._layer_combo.currentIndexChanged.connect(self._on_layer_selected)
        row1.addWidget(self._layer_combo)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Scale level:"))
        self._level_combo = QComboBox()
        self._level_combo.currentIndexChanged.connect(self._on_level_selected)
        row2.addWidget(self._level_combo)
        layout.addLayout(row2)

        info_box = QGroupBox("Hover Info")
        info_layout = QVBoxLayout(info_box)
        info_layout.setSpacing(3)

        self._lbl_ndim = QLabel("Ndim: —")
        self._lbl_dtype = QLabel("Dtype: —")
        self._lbl_chunk_idx = QLabel("Chunk index: —")
        self._lbl_chunk_slice = QLabel("Chunk slices: —")
        self._lbl_chunk_shape = QLabel("Chunk shape: —")
        self._lbl_nchunks = QLabel("Total chunks: —")

        mono = QFont("Monospace", 9)
        mono.setStyleHint(QFont.Monospace)
        for lbl in (
            self._lbl_dtype,
            self._lbl_chunk_idx,
            self._lbl_chunk_slice,
            self._lbl_chunk_shape,
            self._lbl_nchunks,
        ):
            lbl.setFont(mono)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)

        for lbl in (
            self._lbl_ndim,
            self._lbl_dtype,
            self._lbl_chunk_idx,
            self._lbl_chunk_slice,
            self._lbl_chunk_shape,
            self._lbl_nchunks,
        ):
            info_layout.addWidget(lbl)

        layout.addWidget(info_box)

        btn = QPushButton("Redraw grid")
        btn.clicked.connect(self._update_grid)
        layout.addWidget(btn)

        layout.addStretch()

    # ------------------------------------------------------------ Combos

    def _refresh_layer_combo(self):
        current = self._layer_combo.currentText()
        self._layer_combo.blockSignals(True)
        self._layer_combo.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                self._layer_combo.addItem(layer.name)
        idx = self._layer_combo.findText(current)
        self._layer_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._layer_combo.blockSignals(False)
        self._refresh_level_combo()

    def _refresh_level_combo(self):
        self._level_combo.blockSignals(True)
        self._level_combo.clear()
        layer = self._selected_layer()
        if layer is not None and hasattr(layer, "data_level"):
            for i in range(len(layer.data)):
                darr = _get_dask_array_at_level(layer, i)
                shape_str = str(darr.shape) if darr is not None else "?"
                self._level_combo.addItem(f"{i}  {shape_str}")
            self._level_combo.setCurrentIndex(
                min(layer.data_level, len(layer.data) - 1)
            )
        else:
            self._level_combo.addItem("0  (single scale)")
        self._level_combo.blockSignals(False)

    def _selected_layer(self) -> Optional[Image]:
        name = self._layer_combo.currentText()
        for layer in self.viewer.layers:
            if isinstance(layer, Image) and layer.name == name:
                return layer
        return None

    def _selected_level(self) -> int:
        return max(self._level_combo.currentIndex(), 0)

    def _on_layers_change(self, event=None):
        if not self._updating_layers:
            self._refresh_layer_combo()

    def _on_layer_selected(self, _):
        self._refresh_level_combo()
        self._clear_info()

    def _on_level_selected(self, _):
        self._clear_info()

    # --------------------------------------------------------- Mouse hover

    def _on_mouse_move(self, _, event):
        layer = self._selected_layer()
        if not isinstance(layer, Image):
            self._clear_info()
            return

        darr = _get_dask_array_at_level(layer, self._selected_level())
        if darr is None:
            self._clear_info("(not a dask array)")
            return

        try:
            data_coords = layer.world_to_data(event.position)
        except Exception:
            return

        result = _world_to_chunk_index(data_coords, darr.chunks)
        if result is None:
            self._clear_info("(out of bounds)")
            return

        chunk_idx, chunk_slices, chunk_shape = result
        nchunks = tuple(len(c) for c in darr.chunks)

        self._lbl_ndim.setText(f"Ndim: {darr.ndim}")
        self._lbl_dtype.setText(f"Dtype: {darr.dtype}")
        self._lbl_chunk_idx.setText(
            f"Chunk index: ({', '.join(str(i) for i in chunk_idx)})"
        )
        self._lbl_chunk_slice.setText(
            f"Chunk slices: ({', '.join(f'{s.start}:{s.stop}' for s in chunk_slices)})"
        )
        self._lbl_chunk_shape.setText(
            f"Chunk shape: ({', '.join(str(s) for s in chunk_shape)})"
        )
        self._lbl_nchunks.setText(
            f"Total chunks: ({', '.join(str(n) for n in nchunks)}) = {np.prod(nchunks)}"
        )

    # ---------------------------------------------------------- Chunk grid

    def _update_grid(self):
        if self.viewer.dims.ndisplay != 2:
            self._remove_grid()
            return

        layer = self._selected_layer()
        if not isinstance(layer, Image):
            self._remove_grid()
            return

        darr = _get_dask_array_at_level(layer, self._selected_level())
        if darr is None or darr.ndim < 2:
            self._remove_grid()
            return

        chunks = darr.chunks
        ndim = darr.ndim
        displayed = list(self.viewer.dims.displayed)
        dim_y, dim_x = displayed[-2], displayed[-1]
        if dim_y >= ndim or dim_x >= ndim:
            self._remove_grid()
            return

        shape_y = sum(chunks[dim_y])
        shape_x = sum(chunks[dim_x])

        lines = []
        cumsum = 0
        for chunk_size in chunks[dim_y][:-1]:
            cumsum += chunk_size
            lines.append([
                self._nd_point(ndim, dim_y, cumsum, dim_x, 0),
                self._nd_point(ndim, dim_y, cumsum, dim_x, shape_x),
            ])
        cumsum = 0
        for chunk_size in chunks[dim_x][:-1]:
            cumsum += chunk_size
            lines.append([
                self._nd_point(ndim, dim_y, 0, dim_x, cumsum),
                self._nd_point(ndim, dim_y, shape_y, dim_x, cumsum),
            ])

        self._updating_layers = True
        try:
            self._remove_grid()
            if not lines:
                return
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grid_layer = self.viewer.add_shapes(
                    np.array(lines),
                    shape_type="line",
                    edge_color="yellow",
                    edge_width=1.5,
                    opacity=0.5,
                    name=f"[chunk grid] {layer.name}",
                    blending="translucent",
                )
            try:
                grid_layer.affine = layer.affine
            except Exception:
                pass
        finally:
            self._updating_layers = False

    def _nd_point(
        self, ndim: int, dim_y: int, val_y: float, dim_x: int, val_x: float
    ) -> list[float]:
        point = list(self.viewer.dims.point)[:ndim]
        point += [0.0] * (ndim - len(point))
        point[dim_y] = val_y
        point[dim_x] = val_x
        return point

    def _remove_grid(self):
        for layer in list(self.viewer.layers):
            if layer.name.startswith("[chunk grid]"):
                try:
                    self.viewer.layers.remove(layer)
                except Exception:
                    pass

    # ------------------------------------------------------------ Helpers

    def _clear_info(self, msg: str = ""):
        dash = msg or "—"
        self._lbl_ndim.setText(f"Ndim: {dash}")
        self._lbl_dtype.setText(f"Dtype: {dash}")
        self._lbl_chunk_idx.setText(f"Chunk index: {dash}")
        self._lbl_chunk_slice.setText(f"Chunk slices: {dash}")
        self._lbl_chunk_shape.setText(f"Chunk shape: {dash}")
        self._lbl_nchunks.setText(f"Total chunks: {dash}")

    def close(self):
        try:
            self.viewer.mouse_move_callbacks.remove(self._on_mouse_move)
        except ValueError:
            pass
        self.viewer.layers.events.inserted.disconnect(self._on_layers_change)
        self.viewer.layers.events.removed.disconnect(self._on_layers_change)
        self._remove_grid()
        super().close()
