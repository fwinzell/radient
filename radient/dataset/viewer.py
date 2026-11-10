import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


class VolumeSliceViewer:
    """
    Interactive 2D slice viewer for 3D CT volumes.

    Supported image shapes:
        [D, H, W]
        [C, D, H, W]   -> first channel is used

    Supported segmentation shapes:
        [D, H, W]
        [C, D, H, W]   -> first channel is used

    Keyboard:
        Right arrow / Up arrow   -> next slice
        Left arrow / Down arrow  -> previous slice
        Home                     -> first slice
        End                      -> last slice
        Escape                   -> close viewer
    """

    def __init__(
        self,
        image,
        segmentation=None,
        cmap="gray",
        seg_cmap="tab10",
        figsize=(12, 6),
        vmin=None,
        vmax=None,
    ):
        self.image = self._to_numpy(image)
        self.segmentation = (
            None
            if segmentation is None
            else self._to_numpy(segmentation)
        )

        # Basic shape checks
        if self.image.ndim != 3:
            raise ValueError(
                f"Image must have shape [D,H,W] or [C,D,H,W]. "
                f"Got {self.image.shape}"
            )

        if self.segmentation is not None:
            if self.segmentation.shape != self.image.shape:
                raise ValueError(
                    "Image and segmentation must have identical "
                    f"spatial shape. Got {self.image.shape} and "
                    f"{self.segmentation.shape}"
                )

        self.n_slices = self.image.shape[0]
        self.slice_idx = self.n_slices // 2

        self.cmap = cmap
        self.seg_cmap = seg_cmap
        self.figsize = figsize

        self.vmin = (
            np.percentile(self.image, 1)
            if vmin is None
            else vmin
        )
        self.vmax = (
            np.percentile(self.image, 99)
            if vmax is None
            else vmax
        )

        self.fig = None
        self.axes = None
        self.image_artist = None
        self.seg_artist = None
        self.title_artist = None

    @staticmethod
    def _to_numpy(x):
        """Convert torch tensor / numpy array to [D,H,W]."""

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        x = np.asarray(x)

        # [C,D,H,W] -> first channel
        if x.ndim == 4:
            if x.shape[0] != 1:
                raise ValueError(
                    "4D input must have shape [1,D,H,W]. "
                    f"Got {x.shape}"
                )
            x = x[0]

        if x.ndim != 3:
            raise ValueError(
                f"Expected [D,H,W] or [1,D,H,W], got {x.shape}"
            )

        return x

    def _update(self):
        """Update displayed slice."""

        img = self.image[self.slice_idx]

        self.image_artist.set_data(img)

        if self.segmentation is not None:
            seg = self.segmentation[self.slice_idx]

            # Hide background (label 0)
            masked_seg = np.ma.masked_where(
                seg == 0,
                seg,
            )

            self.seg_artist.set_data(masked_seg)

        self.title_artist.set_text(
            f"Slice {self.slice_idx + 1}/{self.n_slices}"
        )

        self.fig.canvas.draw_idle()

    def _on_key(self, event):

        if event.key in ("right", "up"):
            self.slice_idx = min(
                self.slice_idx + 1,
                self.n_slices - 1,
            )

        elif event.key in ("left", "down"):
            self.slice_idx = max(
                self.slice_idx - 1,
                0,
            )

        elif event.key == "home":
            self.slice_idx = 0

        elif event.key == "end":
            self.slice_idx = self.n_slices - 1

        elif event.key == "escape":
            plt.close(self.fig)
            return

        else:
            return

        self._update()

    def show(self):
        """Open interactive viewer."""

        if self.segmentation is None:

            self.fig, ax = plt.subplots(
                figsize=self.figsize
            )

            self.axes = [ax]

            self.image_artist = ax.imshow(
                self.image[self.slice_idx],
                cmap=self.cmap,
                vmin=self.vmin,
                vmax=self.vmax,
            )

            ax.set_title(
                f"Slice {self.slice_idx + 1}/{self.n_slices}"
            )

            ax.axis("off")

            self.title_artist = ax.title

        else:

            self.fig, self.axes = plt.subplots(
                1,
                2,
                figsize=self.figsize,
            )

            ax_img, ax_seg = self.axes

            # CT
            self.image_artist = ax_img.imshow(
                self.image[self.slice_idx],
                cmap=self.cmap,
                vmin=self.vmin,
                vmax=self.vmax,
            )

            ax_img.set_title("CT")
            ax_img.axis("off")

            # Segmentation
            masked_seg = np.ma.masked_where(
                self.segmentation[self.slice_idx] == 0,
                self.segmentation[self.slice_idx],
            )

            self.seg_artist = ax_seg.imshow(
                masked_seg,
                cmap=self.seg_cmap,
                vmin=0,
                vmax=max(
                    1,
                    int(self.segmentation.max()),
                ),
                interpolation="nearest",
            )

            ax_seg.set_title("Segmentation")

            # Add legend for segmentation labels (only labels 1 and 2 exist)
            try:
                num_classes = max(1, int(self.segmentation.max()))
            except Exception:
                num_classes = 2

            labels_to_show = [1, 2]
            cmap = plt.get_cmap(self.seg_cmap)
            handles = []
            for lbl in labels_to_show:
                if lbl <= num_classes:
                    # Map label to colormap color (normalized)
                    try:
                        color = cmap(float(lbl) / float(max(num_classes, 1)))
                    except Exception:
                        color = cmap(0)

                    handles.append(
                        Patch(facecolor=color, edgecolor="black", label=f"Label {lbl}")
                    )

            if handles:
                ax_seg.legend(handles=handles, title="Seg Labels", loc="upper right", framealpha=0.8)

            ax_seg.axis("off")

            self.title_artist = ax_img.set_title(
                f"Slice {self.slice_idx + 1}/{self.n_slices}"
            )

        self.fig.canvas.mpl_connect(
            "key_press_event",
            self._on_key,
        )

        plt.tight_layout()
        plt.show()