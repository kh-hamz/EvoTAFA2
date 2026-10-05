"""Decimal clock diagnostics provide evidence, never recovery authorization."""

from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal

from edgefl.data.schema import projected_time


def clock_offset_exact(source_time, epoch):
    projected = projected_time(source_time)
    if projected is None:
        return None
    year, clock = projected.split()
    instant = datetime.fromtimestamp(int(Decimal(epoch)), timezone.utc)
    if abs(int(year) - instant.year) > 1:
        return None
    hours, minutes, seconds = clock.split(":")
    source = Decimal(hours) * 3600 + Decimal(minutes) * 60 + Decimal(seconds)
    packet = Decimal(epoch) % 86400
    return (source - packet + 129600) % 86400 - 43200


class CaptureDiagnostics:
    def __init__(self):
        self.previous = None
        self.previous_frame = None
        self.regressions = Counter()
        self.offsets = Counter()
        self.frame_regressions = 0
        self.maximum_backward_step = Decimal(0)

    def packet(self, frame, epoch):
        value = Decimal(epoch)
        anomaly = None
        if self.previous is not None and value < self.previous:
            magnitude = self.previous - value
            self.regressions[str(magnitude)] += 1
            self.maximum_backward_step = max(self.maximum_backward_step, magnitude)
            anomaly = {"previous_frame": self.previous_frame, "frame": frame,
                       "previous_epoch": str(self.previous), "epoch": str(value),
                       "backward_seconds": str(magnitude)}
        if self.previous_frame is not None and frame <= self.previous_frame:
            self.frame_regressions += 1
        self.previous, self.previous_frame = value, frame
        return anomaly

    def anchor(self, stamp, epoch):
        offset = clock_offset_exact(stamp, epoch)
        self.offsets[str(offset)] += 1

    def summary(self):
        modal = self.offsets.most_common(1)
        mode = modal[0][0] if modal else None
        residuals = Counter()
        if mode is not None and mode != "None":
            for value, count in self.offsets.items():
                residuals["invalid" if value == "None" else str(Decimal(value) - Decimal(mode))] += count
        return {"timestamp_regression_magnitudes_seconds": dict(self.regressions),
                "maximum_backward_step_seconds": str(self.maximum_backward_step),
                "packet_frame_regressions": self.frame_regressions,
                "exact_clock_offsets_seconds": dict(self.offsets),
                "modal_clock_offset_seconds": mode, "residuals_from_modal_offset_seconds": dict(residuals),
                "recovery_authorized": False}
