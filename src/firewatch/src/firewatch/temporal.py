"""Per-class temporal confirmation. Missing frames are never negative evidence."""
from collections import deque
from dataclasses import dataclass
import math
from .config import Rule


@dataclass(frozen=True)
class Transition:
    kind: str
    label: str
    timestamp: float


class TemporalRule:
    def __init__(self, label: str, rule: Rule):
        rule.validate()
        self.label, self.rule = label, rule
        self.samples = deque()
        self.active = False
        self.last_time = None
        self.negative_since = None

    def update(self, timestamp: float, present: bool):
        r = self.rule
        if not math.isfinite(timestamp):
            raise ValueError("Frame timestamps must be finite")
        if self.last_time is not None:
            if timestamp <= self.last_time:
                raise ValueError("Frame timestamps must strictly increase")
            if timestamp - self.last_time > r.max_gap_seconds:
                self.samples.clear()
                self.negative_since = None
                # Keep an existing alarm: a camera outage cannot prove clearance.
        self.last_time = timestamp
        self.samples.append((timestamp, present))
        while self.samples and timestamp - self.samples[0][0] > r.window_seconds:
            self.samples.popleft()

        if self.active:
            if present:
                self.negative_since = None
            elif self.negative_since is None:
                self.negative_since = timestamp
            elif timestamp - self.negative_since >= r.clear_seconds:
                self.active = False
                self.negative_since = None
                self.samples.clear()
                return Transition("alarm_end", self.label, timestamp)
        elif present:
            hits = [t for t, hit in self.samples if hit]
            if (len(hits) >= r.min_hits
                    and hits[-1] - hits[0] >= r.confirm_seconds
                    and len(hits) / len(self.samples) >= r.hit_ratio):
                self.active = True
                return Transition("alarm_start", self.label, timestamp)
        return None


class AlarmManager:
    def __init__(self, rules):
        self.states = {name: TemporalRule(name, rule) for name, rule in rules.items()}

    def update(self, timestamp, detections):
        labels = {d.label for d in detections}
        events = []
        for name, state in self.states.items():
            event = state.update(timestamp, name in labels)
            if event:
                events.append(event)
        return events

    @property
    def active_labels(self):
        return [name for name, state in self.states.items() if state.active]
