"""Triplet renderer: mid-cut axial + sagittal + coronal panels."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import numpy.typing as npt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


class TripletRenderer:
    """Render a 3D mean image as three orthogonal mid-cuts side-by-side."""

    def __init__(
        self,
        fig_size: tuple[float, float] = (12.0, 4.5),
        dpi: int = 100,
        cmap: str = "gray",
        vmin_pct: float = 1.0,
        vmax_pct: float = 99.0,
    ) -> None:
        self.fig_size = fig_size
        self.dpi = dpi
        self.cmap = cmap
        self.vmin_pct = vmin_pct
        self.vmax_pct = vmax_pct

    def __call__(self, mean_img: nib.Nifti1Image, label: str) -> npt.NDArray[np.uint8]:
        data = np.asarray(mean_img.get_fdata(), dtype=np.float32)
        if data.ndim != 3:
            raise ValueError(
                f"TripletRenderer expects a 3D image, got ndim={data.ndim}. "
                "Pass a mean image, not a 4D BOLD series."
            )
        nx, ny, nz = data.shape
        vmin = float(np.percentile(data, self.vmin_pct))
        vmax = float(np.percentile(data, self.vmax_pct))

        fig = Figure(figsize=self.fig_size, dpi=self.dpi, facecolor="black")
        canvas = FigureCanvasAgg(fig)

        cuts = [
            ("axial",    np.rot90(data[:, :, nz // 2])),
            ("sagittal", np.rot90(data[nx // 2, :, :])),
            ("coronal",  np.rot90(data[:, ny // 2, :])),
        ]
        for k, (title, slc) in enumerate(cuts):
            ax = fig.add_subplot(1, 3, k + 1)
            ax.imshow(
                slc,
                cmap=self.cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_title(title, color="white", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.subplots_adjust(
            left=0.02, right=0.98, top=0.85, bottom=0.02, wspace=0.05
        )
        fig.text(
            0.5, 0.95, label,
            ha="center", va="top", color="white", fontsize=12, family="monospace",
        )

        canvas.draw()  # type: ignore[no-untyped-call]
        rgba = np.asarray(canvas.buffer_rgba())  # type: ignore[no-untyped-call]
        rgb: npt.NDArray[np.uint8] = rgba[:, :, :3].copy()
        return rgb
