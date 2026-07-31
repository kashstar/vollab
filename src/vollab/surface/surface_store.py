import csv
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from vollab.surface.models import ForwardEstimate
from vollab.surface.svi_slice import SVISlice

FIELDNAMES = [
    "snapshot_ts",
    "underlying",
    "expiry",
    "forward",
    "discount_factor",
    "a",
    "b",
    "rho",
    "m",
    "sigma",
]


class SurfaceStore:
    """Persists fitted SVI slices per snapshot to a CSV file.

    One row per (snapshot_ts, expiry), so the file builds into a time
    series you can load with pandas later and see how surface parameters
    move day to day. Idempotent on (snapshot_ts, expiry): saving the same
    snapshot twice does not duplicate rows.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def save(
        self,
        snapshot_ts: datetime,
        underlying: str,
        forward_estimates: Sequence[ForwardEstimate],
        slices: Sequence[SVISlice],
    ) -> int:
        """Append one row per (forward_estimate, slice) pair for this snapshot.

        forward_estimates and slices must be parallel: forward_estimates[i]
        and slices[i] describe the same expiry.

        Returns the number of rows actually written. Rows whose
        (snapshot_ts, expiry) already exist in the file are skipped rather
        than duplicated.
        """
        existing_keys = self._existing_keys()
        file_exists = self._path.exists()

        rows_to_write = []
        for forward_estimate, slice_ in zip(forward_estimates, slices, strict=True):
            key = (snapshot_ts.isoformat(), slice_.expiry.isoformat())
            if key in existing_keys:
                continue
            rows_to_write.append(
                {
                    "snapshot_ts": snapshot_ts.isoformat(),
                    "underlying": underlying,
                    "expiry": slice_.expiry.isoformat(),
                    "forward": forward_estimate.forward,
                    "discount_factor": forward_estimate.discount_factor,
                    "a": slice_.a,
                    "b": slice_.b,
                    "rho": slice_.rho,
                    "m": slice_.m,
                    "sigma": slice_.sigma,
                }
            )

        if not rows_to_write:
            return 0

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows_to_write)

        return len(rows_to_write)

    def _existing_keys(self) -> set[tuple[str, str]]:
        """Return the set of (snapshot_ts, expiry) pairs already stored."""
        if not self._path.exists():
            return set()
        with self._path.open(newline="") as f:
            reader = csv.DictReader(f)
            return {(row["snapshot_ts"], row["expiry"]) for row in reader}
