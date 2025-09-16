"""
Simple scheduling utilities for the web app (no ML).

Inputs:
- tasks: list of dicts {id, name, minutes, deadline (YYYY-MM-DD), subject}
- preferences: dict with keys
  - study_windows: list of {weekday (0=Mon), start("HH:MM"), end("HH:MM")}
  - work_block_minutes: int (e.g., 50)
  - short_break_minutes: int (e.g., 10)
  - long_break_every_blocks: int (e.g., 4)
  - long_break_minutes: int (e.g., 20)

Output: list of sessions with {date, start, end, task_id, task_name, subject, minutes}
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, date, time
from typing import List, Dict


@dataclass
class Task:
    id: str
    name: str
    subject: str
    minutes: int
    deadline: date


def _parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def _daterange(start_date: date, end_date: date):
    cur = start_date
    while cur <= end_date:
        yield cur
        cur += timedelta(days=1)


def _minutes_between(t1: time, t2: time) -> int:
    dt1 = datetime.combine(date.today(), t1)
    dt2 = datetime.combine(date.today(), t2)
    return int((dt2 - dt1).total_seconds() // 60)


def _add_minutes(t: time, minutes: int) -> time:
    dt = datetime.combine(date.today(), t) + timedelta(minutes=minutes)
    return dt.time()


def schedule_tasks(tasks: List[Dict], preferences: Dict, start_from: date | None = None) -> List[Dict]:
    # Convert tasks
    task_objs: List[Task] = []
    for t in tasks:
        task_objs.append(
            Task(
                id=str(t["id"]),
                name=t["name"].strip(),
                subject=t.get("subject", "General").strip() or "General",
                minutes=int(t["minutes"]),
                deadline=datetime.strptime(t["deadline"], "%Y-%m-%d").date(),
            )
        )

    # Sort by earliest deadline, then longest tasks first (to ensure fit)
    task_objs.sort(key=lambda x: (x.deadline, -x.minutes))

    # Preferences
    windows = preferences.get("study_windows", [])
    block = int(preferences.get("work_block_minutes", 50))
    short_break = int(preferences.get("short_break_minutes", 10))
    long_every = int(preferences.get("long_break_every_blocks", 4))
    long_break = int(preferences.get("long_break_minutes", 20))

    # Build windows per weekday
    by_weekday: Dict[int, List[Dict]] = {}
    for w in windows:
        wd = int(w["weekday"])  # 0..6
        by_weekday.setdefault(wd, []).append(
            {"start": _parse_hhmm(w["start"]), "end": _parse_hhmm(w["end"])}
        )

    if start_from is None:
        start_from = date.today()

    # Horizon until max deadline
    max_deadline = max((t.deadline for t in task_objs), default=start_from)

    sessions: List[Dict] = []

    # Remaining minutes per task
    remaining = {t.id: t.minutes for t in task_objs}

    # Iterate days
    block_counter = 0
    for d in _daterange(start_from, max_deadline):
        day_wds = by_weekday.get(d.weekday(), [])
        if not day_wds:
            continue

        for w in sorted(day_wds, key=lambda x: x["start"]):
            cur = w["start"]
            end = w["end"]
            available = _minutes_between(cur, end)
            if available <= 0:
                continue

            while available > 0:
                # Choose next task with earliest deadline that still has time
                next_task = None
                for t in task_objs:
                    if remaining[t.id] > 0 and d <= t.deadline:
                        next_task = t
                        break
                if not next_task:
                    # Either all done or remaining tasks already missed deadline; try still schedule them
                    for t in task_objs:
                        if remaining[t.id] > 0:
                            next_task = t
                            break
                    if not next_task:
                        break

                # Work chunk
                work_minutes = min(block, remaining[next_task.id], available)
                if work_minutes <= 0:
                    break

                start_time = cur
                end_time = _add_minutes(cur, work_minutes)

                sessions.append(
                    {
                        "date": d.isoformat(),
                        "start": start_time.strftime("%H:%M"),
                        "end": end_time.strftime("%H:%M"),
                        "task_id": next_task.id,
                        "task_name": next_task.name,
                        "subject": next_task.subject,
                        "minutes": work_minutes,
                    }
                )

                # Update counters
                remaining[next_task.id] -= work_minutes
                moved = work_minutes
                cur = end_time
                available -= moved
                block_counter += 1 if work_minutes == block else 0

                if available <= 0:
                    break

                # Insert break if there is still time in the window
                break_len = 0
                if work_minutes == block:
                    break_len = long_break if (long_every and block_counter % long_every == 0) else short_break
                if break_len > 0 and available - break_len > 0:
                    cur = _add_minutes(cur, break_len)
                    available -= break_len

    return sessions


