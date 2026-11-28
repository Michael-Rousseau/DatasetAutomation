"""Subset in which every image stays seen by enough kept images.

The overlap rule of `extraction.py` thins every survey line at the same pace, wherever it is.
This one asks, image by image, whether what it shows is still shown by enough kept images:
it thins the middle of a well-covered area more, and the ends of lines and the rim of the
area less, since few images see them. On Mermaid, for the same number of images kept, it
leaves 15 to 20 points more of the rim reconstructible (docs/overlap_journal.md).

It does not decide how many images a reconstruction needs: depth precision follows the number
of images kept, whatever the rule.
"""

from collections.abc import Collection, Mapping

Pair = tuple[int, int]

# Two images show the same place when each sees at least this share of the other.
SUPPORT_OVERLAP = 0.5
# Views closer than this triangulate poorly; the same bound as the judge in `reference/`.
MIN_ANGLE_DEG = 10.0
# The kept images around a removed one must still overlap by this much: the sequence
# keeps chaining in time, which matching between neighbours relies on.
CHAIN_OVERLAP = 0.3


def select_supported_subset(
    image_count: int,
    mutual_overlaps: Mapping[Pair, float],
    min_views: int,
    view_angles: Mapping[Pair, float] | None = None,
    protected: Collection[int] = (),
) -> list[int]:
    """Positions of the images to keep so that every image has `min_views` kept views.

    The views of an image are itself, if kept, and the kept images that overlap it by
    SUPPORT_OVERLAP. `mutual_overlaps` maps (lower position, higher position) to the known
    overlaps: neighbours in time, and loop closures when there are any.

    With `view_angles`, the angle in degrees between the two views of a pair, two of the
    views must also be MIN_ANGLE_DEG apart. Without a calibration there is no angle, and
    only the number of views counts.

    An image that does not have this support with every image kept is kept: nothing known
    says it is redundant. So are the first and the last image, and the `protected` ones:
    those the caller knows to be hard to relate to their neighbours, where an overlap says
    little about how well a reconstruction will hold. Greedy, most seen images
    first; the result depends on that order and is not the smallest possible subset.
    """
    if min_views < 1:
        raise ValueError(f"min_views must be at least 1, got {min_views}")
    support = _Support(image_count, mutual_overlaps, min_views, view_angles)
    # Only what holds with every image kept is asked to keep holding: an image that never
    # had the support can neither be removed nor stop another one from being removed.
    supported = {image for image in range(image_count) if support.holds(image)}
    candidates = supported - set(protected)

    removed_one = True
    while removed_one:
        removed_one = False
        # Most seen first: they are the ones the others need least.
        for image in sorted(candidates, key=lambda image: -support.kept_views(image)):
            if not support.chain_holds_without(image):
                continue
            support.kept[image] = False
            watched = [image, *(support.neighbours[image] & supported)]
            if all(support.holds(other) for other in watched):
                candidates.discard(image)
                removed_one = True
            else:
                support.kept[image] = True
    return [image for image in range(image_count) if support.kept[image]]


class _Support:
    def __init__(
        self,
        image_count: int,
        mutual_overlaps: Mapping[Pair, float],
        min_views: int,
        view_angles: Mapping[Pair, float] | None,
    ) -> None:
        self.kept = [True] * image_count
        self.mutual_overlaps = mutual_overlaps
        self.min_views = min_views
        self.view_angles = view_angles
        self.neighbours: list[set[int]] = [set() for _ in range(image_count)]
        for (first, second), overlap in mutual_overlaps.items():
            if overlap >= SUPPORT_OVERLAP:
                self.neighbours[first].add(second)
                self.neighbours[second].add(first)

    def kept_views(self, image: int) -> int:
        return sum(self.kept[other] for other in self.neighbours[image])

    def holds(self, image: int) -> bool:
        views = sorted(other for other in self.neighbours[image] if self.kept[other])
        if self.kept[image]:
            views.append(image)
        if len(views) < self.min_views:
            return False
        if self.view_angles is None:
            return True
        angles = self.view_angles
        return any(
            angles.get((min(first, second), max(first, second)), 0.0) >= MIN_ANGLE_DEG
            for index, first in enumerate(views)
            for second in views[index + 1 :]
        )

    def chain_holds_without(self, image: int) -> bool:
        before = self._kept_from(range(image - 1, -1, -1))
        after = self._kept_from(range(image + 1, len(self.kept)))
        if before is None or after is None:
            return False
        return self.mutual_overlaps.get((before, after), 0.0) >= CHAIN_OVERLAP

    def _kept_from(self, positions: range) -> int | None:
        return next((position for position in positions if self.kept[position]), None)
