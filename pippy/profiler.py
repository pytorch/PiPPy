import os
import socket
import time
from typing import Optional


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


def write_event_content(rank=None, content=None):
    if content is not None and rank is not None:
        with open(f"{rank}.json", "a") as f:
            f.write(content)
            f.write('\n')


def write_common_event(rank: int, name: str, cat: str, ph: str, ts: float, bp=None):
    content = generate_event_str(name=name, id=name, cat=cat, ph=ph, bp=bp, ts=ts * 1_000_000, pid=socket.gethostname(),
                                 tid=f'{rank}')
    write_event_content(rank=rank, content=content)


def write_begin_event(rank: int, name: str, cat: str, ts: float):
    write_common_event(rank=rank, name=name, cat=cat, ph="B", ts=ts)


def write_end_event(rank: int, name: str, cat: str, ts: float):
    write_common_event(rank=rank, name=name, cat=cat, ph="E", ts=ts)


def write_finish_event(rank: int, name: str, cat: str, ts: float):
    write_common_event(rank=rank, name=name, cat=cat, ph="f", ts=ts)


def write_start_event(rank: int, name: str, cat: str, ts: float):
    write_common_event(rank=rank, name=name, cat=cat, ph="s", ts=ts)


class RecordEventContextManager:
    def __init__(self, rank: int, name: str, cat: str, prev_name: Optional[str], next_name: Optional[str]):
        self.rank: int = rank
        self.name: str = name
        self.cat: str = cat
        self.prev_name: Optional[str] = prev_name
        self.next_name: Optional[str] = next_name

    def __enter__(self):
        self.start = time.time()
        if self.prev_name is not None:
            write_finish_event(rank=self.rank, name=f"{self.prev_name} -> {self.name}", cat="transfer", ts=self.start)
        write_begin_event(rank=self.rank, name=self.name, cat=self.cat, ts=self.start)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.finish = time.time()
        if self.next_name is not None:
            write_start_event(rank=self.rank, name=f"{self.name} -> {self.next_name}", cat="transfer", ts=self.finish)
        write_end_event(rank=self.rank, name=self.name, cat=self.cat, ts=self.finish)


def record_event(rank: int, name: str, cat: str, prev_name: Optional[str] = None, next_name: Optional[str] = None):
    return RecordEventContextManager(rank=rank, name=name, cat=cat, prev_name=prev_name, next_name=next_name)


def merge_jsons(world_size, result_file_name):
    with open(result_file_name, "w") as res:
        lines = []
        for i in range(world_size):
            with open(f"{i}.json", "r") as f:
                lines.extend([l.rstrip() for l in f.readlines()])
            os.remove(f"{i}.json")
        res.write("[\n")
        res.write(",\n".join(lines))
        res.write("\n]\n")
