"""Put a rendered scene in a figure with its legend: Open3D draws no legend of its own."""

from collections.abc import Sequence

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from dataset_automation.view.scene import ANGLE_COLOURMAP, UNKNOWN_ANGLE_COLOUR

LegendEntry = tuple[str, tuple[float, float, float]]


def legend_figure(
    image: np.ndarray,
    title: str,
    entries: Sequence[LegendEntry],
    max_angle_deg: float | None,
    has_unknown_angle: bool = False,
) -> Figure:
    """`image` with a legend of `entries` (label, RGB) and, with `max_angle_deg`, the scale of
    the angle between the two views of a link."""
    height_px, width_px = image.shape[:2]
    figure = Figure(figsize=(width_px / 100, height_px / 100 + 0.8), dpi=100)
    axis = figure.add_subplot()
    axis.imshow(image)
    axis.set_axis_off()
    axis.set_title(title)

    handles = [Patch(color=colour, label=label) for label, colour in entries]
    if max_angle_deg is not None:
        if has_unknown_angle:
            handles.append(
                Patch(color=UNKNOWN_ANGLE_COLOUR, label="link, angle unknown")
            )
        colour_bar = figure.colorbar(
            ScalarMappable(Normalize(0.0, max_angle_deg), ANGLE_COLOURMAP),
            ax=axis,
            orientation="horizontal",
            fraction=0.04,
            pad=0.02,
            extend="max",
        )
        colour_bar.set_label("angle between the two views of a link (degrees)")
    if handles:
        axis.legend(handles=handles, loc="upper left", fontsize="small")
    figure.tight_layout()
    return figure
