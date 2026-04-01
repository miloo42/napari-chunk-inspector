# napari-chunk-inspector

Inspect dask chunk boundaries and indices for OME-Zarr data in napari.

Hover over any pixel to see which chunk it belongs to, and draw a grid overlay marking every chunk boundary in the current 2D view.

![Chunk Inspector screenshot](assets/screenshot.png)

## Features

- **Hover info** — chunk index, data-space slices, chunk shape, dtype, and total chunk count
- **Chunk grid** — yellow lines at every chunk boundary in 2D view, drawn on demand
- **Multi-scale** — select any pyramid level from a dropdown; shows shape per level
- **Layer selector** — choose which image layer to inspect without switching the active layer
- **nD-aware** — respects the currently displayed axes, including after axis swapping

## Requirements

- napari >= 0.4.19
- dask[array]
- numpy
- qtpy
- napari-ome-zarr (for opening OME-Zarr files)

## Installation

Clone the repository and install in editable mode:

```bash
git clone <repo-url>
cd napari-chunk-inspector
pip install -e .
```

## Usage

After installing, open napari and go to **Plugins → Chunk Inspector**.

Or add it programmatically:

```python
import napari
from napari_chunk_inspector._widget import ChunkInspector

viewer = napari.Viewer()
viewer.open("path/to/data.ome.zarr", plugin="napari-ome-zarr")
widget = ChunkInspector(viewer)
viewer.window.add_dock_widget(widget, name="Chunk Inspector", area="right")
napari.run()
```

### Controls

| Control | Description |
| --- | --- |
| **Layer** dropdown | Select the image layer to inspect |
| **Scale level** dropdown | Select the pyramid level; shows shape of each level |
| **Hover info panel** | Updates in real time as you move the cursor |
| **Redraw grid** button | Draws chunk boundary lines for the selected layer and level at the current slice position |

The grid is drawn at the current slice and axis orientation when the button is clicked. Click again after scrolling to a new slice or swapping axes to update it.

## Changelog

### 0.1.0

- Initial release with hover info panel and chunk grid overlay
- Multi-scale support via `data_level` attribute on napari Image layers
- Handles `da.Array`, `xarray.DataArray`, and `zarr.Array` backing stores
- Layer selector dropdown to inspect any image layer independently of the active layer selection
- Scale level dropdown showing per-level array shapes
- Fixed layer accumulation bug caused by feedback loop between `layers.inserted` event and grid redraw
- Removed chunk highlight overlay in favour of the index/slice readout
- Grid redraws only on explicit button click to avoid slowdown when scrolling through large datasets
- Axes-aware grid: uses `viewer.dims.displayed` so the grid follows axis swapping correctly
