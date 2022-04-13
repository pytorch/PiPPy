# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Dict, List

from pippy.events import Event, EventDependency, EventsContext


def generate_event_str(name=None, id=None, cat=None, ph=None, bp=None, ts=None, tts=None, dur=None, pid=None, tid=None,
                       args=None, cname=None):
    items = []

    def append_if_not_none(key, val):
        if val is not None:
            if type(val) == str:
                items.append(f'"{key}": "{val}"')
            else:
                items.append(f'"{key}": {val}')

    append_if_not_none('name', name)
    append_if_not_none('id', id)
    append_if_not_none('cat', cat)
    append_if_not_none('ph', ph)
    append_if_not_none('bp', bp)
    append_if_not_none('ts', ts)
    append_if_not_none('tts', tts)
    append_if_not_none('dur', dur)
    append_if_not_none('pid', pid)
    append_if_not_none('tid', tid)
    # append_if_not_none('args', args) # TODO!
    append_if_not_none('cname', cname)
    if len(items):
        return '{' + ', '.join(items) + '}'
    else:
        return None


def event_to_json(event: Event, prev_events: Dict[str, List[EventDependency]],
                  next_events: Dict[str, List[EventDependency]]) -> str:
    lines = []
    if event.id is not None:
        for prev_event in prev_events[event.id]:
            lines.append(generate_event_str(
                pid=f"{event.host}",
                tid=f"{event.rank}",
                name=f"{prev_event.from_id} -> {prev_event.to_id}",
                id=f"{prev_event.from_id} -> {prev_event.to_id}",
                ph="f",
                cat=f"{prev_event.type}",
                ts=event.start_ts * 1_000_000
            ))
    lines.append(generate_event_str(
        pid=f"{event.host}",
        tid=f"{event.rank}",
        name=event.name,
        id=event.id,
        ph="B",
        ts=event.start_ts * 1_000_000
    ))
    if event.id is not None:
        for next_event in next_events[event.id]:
            lines.append(generate_event_str(
                pid=f"{event.host}",
                tid=f"{event.rank}",
                name=f"{next_event.from_id} -> {next_event.to_id}",
                id=f"{next_event.from_id} -> {next_event.to_id}",
                ph="s",
                cat=f"{next_event.type}",
                ts=event.finish_ts * 1_000_000
            ))
    lines.append(generate_event_str(
        pid=f"{event.host}",
        tid=f"{event.rank}",
        name=event.name,
        id=event.id,
        ph="E",
        ts=event.finish_ts * 1_000_000
    ))
    return ',\n'.join(lines)


def events_to_json(events_context: EventsContext) -> str:
    result = '[\n'
    result += ',\n'.join(
        [event_to_json(event, events_context.prev_events, events_context.next_events) for event in
         events_context.events])
    result += '\n]\n'
    return result
