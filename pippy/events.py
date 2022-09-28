# Copyright (c) Meta Platforms, Inc. and affiliates
import os
import socket
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Allocator:
    id: str
    attrs: Dict[str, int]


@dataclass
class Event:
    rank: int
    host: str
    pid: int
    start_ts: float
    finish_ts: float
    id: Optional[str]
    name: Optional[str]
    type: Optional[Any]
    mbid: Optional[Any]


@dataclass
class MemDumpEvent(Event):
    allocators: Dict[str, Allocator]


@dataclass
class EventDependency:
    from_id: str
    to_id: str
    type: Optional[Any]


@dataclass
class EventsContext:
    events: List[Event] = field(default_factory=list)
    next_events: Dict[str, List[EventDependency]] = field(
        default_factory=lambda: defaultdict(list)
    )
    prev_events: Dict[str, List[EventDependency]] = field(
        default_factory=lambda: defaultdict(list)
    )

    @staticmethod
    def _update(
        dst: Dict[str, List[EventDependency]],
        src: Dict[str, List[EventDependency]],
    ):
        for k, v in src.items():
            dst[k].extend(v)

    def update(self, events_context: "EventsContext"):
        self.events.extend(events_context.events)
        self._update(self.next_events, events_context.next_events)
        self._update(self.prev_events, events_context.prev_events)
        return self

    def reset(self):
        self.__init__()


class EventRecorder:
    events_context: EventsContext = EventsContext()
    hostname: str = socket.gethostname()
    pid = os.getpid()

    def record_event(
        self,
        rank: int,
        start_ts: float,
        finish_ts: float,
        id: str,
        name: str,
        type: Optional[Any],
        mbid: Optional[Any],
    ):
        self.events_context.events.append(
            Event(
                rank=rank,
                host=self.hostname,
                pid=self.pid,
                start_ts=start_ts,
                finish_ts=finish_ts,
                id=id,
                name=name,
                type=type,
                mbid=mbid,
            )
        )

    def record_dump(
        self,
        rank: int,
        ts: float,
        id: str,
        name: str,
        type: Optional[Any],
        allocators: Dict[str, Allocator],
    ):
        self.events_context.events.append(
            MemDumpEvent(
                rank=rank,
                host=self.hostname,
                pid=self.pid,
                start_ts=ts,
                finish_ts=ts,
                id=id,
                name=name,
                type=type,
                allocators=allocators,
                mbid=None,
            )
        )

    def record_event_dependency(
        self, from_id: str, to_id: str, type: Optional[Any]
    ):
        dep = EventDependency(from_id=from_id, to_id=to_id, type=type)
        self.events_context.next_events[from_id].append(dep)
        self.events_context.prev_events[to_id].append(dep)

    def retrieve_events(self):
        events_context = deepcopy(self.events_context)
        self.events_context.reset()
        return events_context
