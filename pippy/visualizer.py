from typing import List

from pippy.PipelineDriver import Event


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
    # append_if_not_none('args', args)
    append_if_not_none('cname', cname)
    if len(items):
        return '{' + ', '.join(items) + '}'
    else:
        return None


def event_to_json(event: Event) -> str:
    return generate_event_str(
        pid=f"{0}",
        tid=f"{event.rank}",
        name=event.name,
        id=event.id,
        ph="X",
        ts=event.start_ts * 1_000_000,
        dur=(event.finish_ts - event.start_ts) * 1_000_000
    )


def events_to_json(events: List[Event]) -> str:
    result = '[\n'
    result += ',\n'.join([event_to_json(event) for event in events])
    result += '\n]'
    return result
