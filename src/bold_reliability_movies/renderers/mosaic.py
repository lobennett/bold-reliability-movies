"""Default in-tree renderer: a grid of axial slices with a text label."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import numpy.typing as npt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


class MosaicRenderer:
    """Render a 3D mean image as an N×M grid of evenly-spaced axial slices."""

    def __init__(
        self,
        n_rows: int = 5,
        n_cols: int = 5,
        fig_size: tuple[float, float] = (8.0, 8.0),
        dpi: int = 100,
        cmap: str = "gray",
        vmin_pct: float = 1.0,
        vmax_pct: float = 99.0,
    ) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.fig_size = fig_size
        self.dpi = dpi
        self.cmap = cmap
        self.vmin_pct = vmin_pct
        self.vmax_pct = vmax_pct

    def __call__(self, mean_img: nib.Nifti1Image, label: str) -> npt.NDArray[np.uint8]:
        data = np.asarray(mean_img.get_fdata(), dtype=np.float32)
        n_slices = self.n_rows * self.n_cols
        nz = data.shape[2]
        z_idx = np.linspace(0, nz - 1, n_slices).astype(int)
        vmin = float(np.percentile(data, self.vmin_pct))
        vmax = float(np.percentile(data, self.vmax_pct))

        fig = Figure(figsize=self.fig_size, dpi=self.dpi, facecolor="black")
        canvas = FigureCanvasAgg(fig)
        for k, z in enumerate(z_idx):
            ax = fig.add_subplot(self.n_rows, self.n_cols, k + 1)
            ax.imshow(
                np.rot90(data[:, :, z]),
                cmap=self.cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.05, hspace=0.05)
        fig.text(
            0.5, 0.96, label,
            ha="center", va="top", color="white", fontsize=14, family="monospace",
        )

        canvas.draw()  # type: ignore[no-untyped-call]
        rgba = np.asarray(canvas.buffer_rgba())  # type: ignore[no-untyped-call]
        rgb: npt.NDArray[np.uint8] = rgba[:, :, :3].copy()
        return rgb
