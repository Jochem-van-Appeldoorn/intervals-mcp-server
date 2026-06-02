"""
Planning-related MCP tools for Intervals.icu.

ATP (Annual Training Plan) tools that adapt to the available time and the
athlete's current fitness level:
- create_atp_plan  fetches current CTL and auto-determines which phases are
                   needed, how long they should be, and posts them as NOTE
                   events on the Intervals.icu calendar.
- get_atp_plan     reads all phase notes from the calendar.
- get_atp_week_note  returns the phase note covering a specific week.
- get_planning_context  returns fitness + ATP note + events for weekly planning.
- add_race_event   adds a race event with priority A, B, or C.
"""

import asyncio
import re
from datetime import date, datetime, timedelta
from typing import Any

from intervals_mcp_server.api.client import make_intervals_request
from intervals_mcp_server.config import get_config
from intervals_mcp_server.mcp_instance import mcp  # noqa: F401
from intervals_mcp_server.utils.validation import resolve_athlete_id, validate_date

config = get_config()

# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

_PHASES: dict[str, dict[str, Any]] = {
    "preparation": {
        "label": "Preparation",
        "color": "green",
        "focus": "Build general fitness. Establish aerobic base, strength and technique.",
        "intensity": "80% Z2, 15% sweet spot, 5% threshold.",
        "key_sessions": ["Long Z2 endurance rides", "Strength training", "Flexibility / technique"],
        "tss_factor": 0.60,
    },
    "base": {
        "label": "Base",
        "color": "blue",
        "focus": "Strengthen the aerobic engine. High Z2 volume, sweet spot, threshold introduction.",
        "intensity": "65% Z2, 25% sweet spot, 10% threshold.",
        "key_sessions": ["Long endurance rides (3–5 h)", "2× sweet spot per week", "1× threshold interval"],
        "tss_factor": 0.80,
    },
    "build": {
        "label": "Build",
        "color": "orange",
        "focus": "Race-specific qualities. Develop threshold, VO2max and anaerobic capacity.",
        "intensity": "40% Z2, 30% threshold, 30% VO2max / anaerobic.",
        "key_sessions": ["2× VO2max or threshold per week", "Long endurance ride", "Race simulation"],
        "tss_factor": 1.00,
    },
    "peak": {
        "label": "Peak",
        "color": "red",
        "focus": "Get sharp for the race. Volume drops, intensity stays high. Taper.",
        "intensity": "50% Z2, 25% threshold, 25% VO2max — shorter intervals.",
        "key_sessions": ["Race-pace intervals", "Activation ride 2 days before race", "Rest day before race"],
        "tss_factor": 0.65,
    },
    "race": {
        "label": "Race",
        "color": "red",
        "focus": "Perform at the A-race. Minimal training, full focus on racing and recovery.",
        "intensity": "Race + recovery rides.",
        "key_sessions": ["Activation ride the day before", "Race", "Active recovery afterwards"],
        "tss_factor": 0.50,
    },
}

_ATP_PREFIX = "ATP"


def _is_atp_note(e: object) -> bool:
    return (
        isinstance(e, dict)
        and e.get("category") == "NOTE"  # type: ignore[union-attr]
        and (e.get("name") or "").startswith(_ATP_PREFIX)  # type: ignore[union-attr]
    )


# ---------------------------------------------------------------------------
# Phase auto-selection logic (relative to goal CTL, not absolute thresholds)
# ---------------------------------------------------------------------------


def _determine_phases(total_weeks: int, current_ctl: float, goal_ctl: float) -> list[tuple[str, int]]:
    """Return ordered (phase_name, weeks) based on available weeks and CTL gap.

    Uses the ratio current_ctl / goal_ctl so thresholds scale with the event:
    - UCI u23 rider targeting CTL 110 with current 70 → big gap → needs base
    - Gran fondo finisher targeting CTL 60 with current 50 → small gap → straight to build
    """
    if total_weeks <= 1:
        return [("race", total_weeks)]
    race_wks = 1
    remaining_after_race = total_weeks - race_wks
    peak_wks = 2 if remaining_after_race >= 4 else 1
    if peak_wks >= remaining_after_race:
        peak_wks = remaining_after_race
    remaining = total_weeks - peak_wks - race_wks

    tail = [("peak", peak_wks), ("race", race_wks)]

    if remaining <= 0:
        return tail

    ratio = current_ctl / max(goal_ctl, 1.0)

    if remaining <= 4 or ratio >= 0.90:
        # Almost at goal CTL or very little time → straight to build
        return [("build", remaining)] + tail

    if ratio >= 0.70:
        # Moderate gap (70–90% of goal) → short base block + build
        base_wks = max(3, remaining // 3)
        build_wks = remaining - base_wks
        if build_wks < 3:
            return [("build", remaining)] + tail
        return [("base", base_wks), ("build", build_wks)] + tail

    # Large gap (< 70% of goal) → prep + base + build
    prep_wks = max(2, remaining // 5)
    base_wks = max(3, (remaining - prep_wks) * 2 // 3)
    build_wks = remaining - prep_wks - base_wks

    if build_wks < 3:
        prep_wks = 0
        base_wks = max(3, remaining // 2)
        build_wks = remaining - base_wks
        if build_wks < 3:
            # remaining too small for both base and build — collapse to pure build
            return [("build", remaining)] + tail

    phases: list[tuple[str, int]] = []
    if prep_wks >= 2:
        phases.append(("preparation", prep_wks))
    phases.append(("base", base_wks))
    phases.append(("build", build_wks))
    return phases + tail


def _reason_for_phases(current_ctl: float, goal_ctl: float, total_weeks: int) -> str:
    ratio = current_ctl / max(goal_ctl, 1.0)
    goal_ctl_int = round(goal_ctl)
    current_ctl_int = round(current_ctl)
    gap = goal_ctl_int - current_ctl_int
    if total_weeks <= 5:
        phases = _determine_phases(total_weeks, current_ctl, goal_ctl)
        phase_seq = " → ".join(_PHASES[name]["label"] for name, _ in phases)
        return f"Only {total_weeks} weeks available — {phase_seq}."
    if gap <= 0:
        return (
            f"CTL {current_ctl_int} already meets or exceeds goal CTL {goal_ctl_int} "
            f"— going straight to build phase."
        )
    if ratio >= 0.90:
        return (
            f"CTL {current_ctl_int} is close to goal CTL {goal_ctl_int} "
            f"(gap: {gap}) — going straight to build phase."
        )
    if ratio >= 0.70:
        return (
            f"CTL {current_ctl_int} is {gap} points below goal CTL {goal_ctl_int} "
            f"({round(ratio * 100)}%) — short base block followed by build phase."
        )
    return (
        f"CTL {current_ctl_int} is {gap} points below goal CTL {goal_ctl_int} "
        f"({round(ratio * 100)}%) — preparation and base block needed before build."
    )


# ---------------------------------------------------------------------------
# Note content builders
# ---------------------------------------------------------------------------

def _monday_of(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _weeks_in(start: date, end: date) -> int:
    return max(1, ((end - start).days + 1 + 6) // 7)



def _week_tss(phase: str, goal_tss: int, week: int, cycle: int, load_week_idx: dict[int, int]) -> str:
    factor = _PHASES[phase]["tss_factor"]
    if week % cycle == 0:
        return f"~{round(goal_tss * factor * 0.60 / 50) * 50} TSS  ← recovery week"
    idx = load_week_idx.get(week, 0)
    prog = 0.85 + 0.15 * (idx / max(len(load_week_idx) - 1, 1))
    tss = round(goal_tss * factor * prog / 50) * 50
    return f"~{tss} TSS"


def _build_phase_note(
    phase: str,
    phase_num: int,
    total_phases: int,
    start: date,
    end: date,
    cycle: int,
    goal_tss: int,
    race_name: str,
    race_date: date,
    phase_races: list[dict],
) -> str:
    p = _PHASES[phase]
    total_wks = _weeks_in(start, end)
    load_week_idx = {w: i for i, w in enumerate(w for w in range(1, total_wks + 1) if w % cycle != 0)}

    lines = [
        f"Phase {phase_num}/{total_phases}: {p['label']}",
        f"Period: {start} – {end} ({total_wks} weeks)",
        f"Goal: {p['focus']}",
        f"Intensity: {p['intensity']}",
        "",
        "Key sessions per week:",
        *[f"  • {s}" for s in p["key_sessions"]],
        "",
        "Weekly schedule:",
    ]
    for w in range(1, total_wks + 1):
        w_start = start + timedelta(weeks=w - 1)
        w_end = min(w_start + timedelta(days=6), end)
        lines.append(f"  Week {w} ({w_start} – {w_end}): {_week_tss(phase, goal_tss, w, cycle, load_week_idx)}")

    lines += ["", f"A-race: {race_name} ({race_date})"]
    if phase_races:
        lines.append("Races in this phase:")
        for r in phase_races:
            lines.append(f"  • {r['date']} — {r['name']} [{r.get('priority', 'C')}]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def create_atp_plan(
    race_date: str,
    race_name: str,
    goal_ctl: int,
    recovery_cycle: int = 4,
    additional_races: str | None = None,
    athlete_id: str | None = None,
    api_key: str | None = None,
) -> str:
    """Create an ATP (Annual Training Plan) in Intervals.icu for a goal race.

    Fetches current CTL, compares it to goal_ctl, and automatically determines
    which phases are needed and how long they should be. The bigger the gap
    between current and goal CTL, the more base-building phases are added.

    Phase selection (based on current CTL as % of goal CTL):
    - >= 90% of goal: already close → Build → Peak → Race
    - 70–90% of goal: moderate gap → Base → Build → Peak → Race
    - < 70% of goal:  large gap   → Preparation → Base → Build → Peak → Race

    goal_ctl reference values (CTL needed at race day):
    - Finish a sportive / gran fondo:       50–70
    - Club-level amateur racer:             70–90
    - Competitive amateur / cat. 3–4:       90–110
    - Semi-pro / UCI U23:                  100–130
    - Pro continental / WorldTour:         130+

    Each phase is posted as a NOTE event spanning its full date range on the
    Intervals.icu calendar. The note contains training focus, intensity split,
    weekly TSS targets, key sessions, and recovery-week schedule.

    Args:
        race_date: Date of the A-race in YYYY-MM-DD format
        race_name: Name of the A-race (e.g. "Tour of Flanders")
        goal_ctl: Target CTL (fitness) needed at race day — see reference above
        recovery_cycle: Recovery week every N weeks — 3 for older/less experienced
                        athletes, 4 for younger/experienced athletes (default 4)
        additional_races: Optional B/C races, one per line as
                          "YYYY-MM-DD Name [A/B/C]"
                          e.g. "2026-03-21 Dwars door Vlaanderen [C]"
        athlete_id: The Intervals.icu athlete ID (optional)
        api_key: The Intervals.icu API key (optional)
    """
    athlete_id_to_use, error_msg = resolve_athlete_id(athlete_id, config.athlete_id)
    if error_msg:
        return error_msg

    try:
        validate_date(race_date)
        race_dt = date.fromisoformat(race_date)
    except ValueError as e:
        return f"Error in race_date: {e}"

    if goal_ctl < 1:
        return "Error: goal_ctl must be at least 1."

    if recovery_cycle not in (3, 4):
        return "Error: recovery_cycle must be 3 or 4."

    today = date.today()
    if race_dt <= today:
        return f"Error: race_date {race_date} is in the past. Provide a future date."

    today_monday = _monday_of(today)
    race_monday = _monday_of(race_dt)
    total_weeks = max(1, (race_monday - today_monday).days // 7 + 1)

    # Fetch current CTL from wellness
    wellness_result = await make_intervals_request(
        url=f"/athlete/{athlete_id_to_use}/wellness",
        api_key=api_key,
        params={
            "oldest": (today - timedelta(days=7)).isoformat(),
            "newest": today.isoformat(),
        },
    )

    current_ctl: float = 30.0  # fallback if no data
    ctl_source = "unknown (no wellness data)"
    if isinstance(wellness_result, list) and wellness_result:
        for entry in reversed(wellness_result):
            if isinstance(entry, dict) and entry.get("ctl") is not None:
                current_ctl = float(entry["ctl"])
                ctl_source = f"{round(current_ctl, 1)} (from wellness {entry.get('id', '')})"
                break

    # Weekly TSS in peak build weeks ≈ goal_ctl × 7
    goal_weekly_tss = round(goal_ctl * 7 / 50) * 50

    # Parse additional races
    extra_races: list[dict] = []
    if additional_races:
        _trailing_priority = re.compile(r'^(.*?)\s*\[([A-Ca-c])\]\s*$')
        for line in additional_races.strip().splitlines():
            if not line.strip():
                continue
            parts = line.strip().split(None, 1)  # split on first whitespace: [date, rest]
            if len(parts) != 2:
                return f"Error: malformed additional_races line '{line.strip()}'. Expected format: 'YYYY-MM-DD Race name [A/B/C]'."
            raw_date, rest = parts[0], parts[1].strip()
            try:
                validate_date(raw_date)
            except ValueError:
                return f"Error: invalid date '{raw_date}' in additional_races. Use YYYY-MM-DD."
            m = _trailing_priority.match(rest)
            if m:
                name_part, priority = m.group(1).strip(), m.group(2).upper()
            else:
                name_part, priority = rest, "C"
            extra_races.append({"date": raw_date, "name": name_part, "priority": priority})

    # Delete any existing ATP notes in the plan window to avoid duplicates
    plan_end = _monday_of(race_dt) + timedelta(days=6)
    existing = await make_intervals_request(
        url=f"/athlete/{athlete_id_to_use}/events",
        api_key=api_key,
        params={"oldest": today_monday.isoformat(), "newest": plan_end.isoformat()},
    )
    _sem = asyncio.Semaphore(3)

    async def _delete(ev_id: str) -> None:
        async with _sem:
            await make_intervals_request(
                url=f"/athlete/{athlete_id_to_use}/events/{ev_id}",
                api_key=api_key,
                method="DELETE",
            )

    to_delete = [
        ev["id"] for ev in (existing if isinstance(existing, list) else [])
        if _is_atp_note(ev) and isinstance(ev, dict) and ev.get("id")
    ]
    if to_delete:
        await asyncio.gather(*(_delete(ev_id) for ev_id in to_delete))

    # Determine phases
    phase_list = _determine_phases(total_weeks, current_ctl, float(goal_ctl))
    reason = _reason_for_phases(current_ctl, float(goal_ctl), total_weeks)

    # Calculate date ranges (forward from today)
    cursor = today_monday
    phase_ranges: list[dict[str, Any]] = []
    for phase_name, wks in phase_list:
        p_start = cursor
        p_end = cursor + timedelta(weeks=wks) - timedelta(days=1)
        if phase_name == "race":
            p_end = race_monday + timedelta(days=6)
        phase_ranges.append({"name": phase_name, "start": p_start, "end": p_end, "weeks": wks})
        cursor = p_end + timedelta(days=1)

    total_phases = len(phase_ranges)

    # Build all event payloads up front (pure computation, no I/O)
    phase_events: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for i, ph in enumerate(phase_ranges, start=1):
        phase_races = [r for r in extra_races if ph["start"].isoformat() <= r["date"] <= ph["end"].isoformat()]
        description = _build_phase_note(
            phase=ph["name"],
            phase_num=i,
            total_phases=total_phases,
            start=ph["start"],
            end=ph["end"],
            cycle=recovery_cycle,
            goal_tss=goal_weekly_tss,
            race_name=race_name,
            race_date=race_dt,
            phase_races=phase_races,
        )
        event_data: dict[str, Any] = {
            "category": "NOTE",
            "name": f"ATP — {_PHASES[ph['name']]['label']}",
            "description": description,
            "start_date_local": ph["start"].isoformat() + "T00:00:00",
            "end_date_local": (ph["end"] + timedelta(days=1)).isoformat() + "T00:00:00",
            "color": _PHASES[ph["name"]]["color"],
        }
        phase_events.append((ph, event_data))

    async def _post_phase(ph: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        async with _sem:
            return await make_intervals_request(
                url=f"/athlete/{athlete_id_to_use}/events",
                api_key=api_key,
                data=data,
                method="POST",
            )

    post_results = await asyncio.gather(
        *(_post_phase(ph, ev) for ph, ev in phase_events),
        return_exceptions=True,
    )

    # Rollback any created events if a failure occurred
    created_ids = [
        r["id"] for r in post_results
        if isinstance(r, dict) and "id" in r and "error" not in r
    ]
    failed = [r for r in post_results if isinstance(r, Exception) or (isinstance(r, dict) and "error" in r)]
    if failed:
        await asyncio.gather(*(_delete(ev_id) for ev_id in created_ids))
        err_msg = failed[0].get("message") if isinstance(failed[0], dict) else str(failed[0])
        return f"Error creating ATP plan (all changes rolled back): {err_msg}"

    posted_lines: list[str] = []
    for (ph, _), result in zip(phase_events, post_results):
        label = _PHASES[ph["name"]]["label"]
        if isinstance(result, dict) and "error" in result:
            posted_lines.append(f"  ✗ {label} ({ph['start']} – {ph['end']}): {result.get('message')}")
        else:
            posted_lines.append(f"  ✓ {label}: {ph['start']} – {ph['end']} ({ph['weeks']} wk)")

    phase_names = " → ".join(_PHASES[ph["name"]]["label"] for ph in phase_ranges)
    summary = [
        f"ATP created for {race_name} ({race_date})",
        f"Period: {today_monday} – {race_monday + timedelta(days=6)} ({total_weeks} weeks)",
        f"Current CTL: {ctl_source}",
        f"Goal CTL: {goal_ctl}  →  Peak week TSS target: ~{goal_weekly_tss}",
        f"Phase selection: {reason}",
        f"Phases: {phase_names}",
        f"Recovery cycle: every {recovery_cycle} weeks",
        "",
        "Posted to calendar:",
    ] + posted_lines

    return "\n".join(summary)


@mcp.tool()
async def get_atp_plan(
    start_date: str | None = None,
    end_date: str | None = None,
    athlete_id: str | None = None,
    api_key: str | None = None,
) -> str:
    """Read the current ATP plan from the Intervals.icu calendar.

    Returns all ATP phase NOTE events in the given date range with their
    training guidelines — a full season overview.

    Args:
        start_date: Start of range in YYYY-MM-DD format
                    (optional, defaults to 3 months ago)
        end_date: End of range in YYYY-MM-DD format
                  (optional, defaults to 18 months from today)
        athlete_id: The Intervals.icu athlete ID (optional)
        api_key: The Intervals.icu API key (optional)
    """
    athlete_id_to_use, error_msg = resolve_athlete_id(athlete_id, config.athlete_id)
    if error_msg:
        return error_msg

    today = date.today()
    if not start_date:
        start_date = (today - timedelta(days=90)).isoformat()
    if not end_date:
        end_date = (today + timedelta(days=550)).isoformat()

    try:
        validate_date(start_date)
        validate_date(end_date)
    except ValueError as e:
        return f"Error: {e}"

    result = await make_intervals_request(
        url=f"/athlete/{athlete_id_to_use}/events",
        api_key=api_key,
        params={"oldest": start_date, "newest": end_date},
    )

    if isinstance(result, dict) and "error" in result:
        return f"Error fetching ATP plan: {result.get('message')}"

    events = result if isinstance(result, list) else []
    notes = [
        e for e in events
        if _is_atp_note(e)
    ]

    if not notes:
        return (
            f"No ATP notes found between {start_date} and {end_date}. "
            "Use create_atp_plan to create a plan."
        )

    lines = [f"ATP overview ({start_date} – {end_date})\n"]
    for note in notes:
        name = note.get("name", "")
        s = (note.get("start_date_local") or "")[:10]
        e_end_raw = (note.get("end_date_local") or note.get("start_date_local") or "")[:10]
        try:
            e_end = (date.fromisoformat(e_end_raw) - timedelta(days=1)).isoformat() if e_end_raw else ""
        except ValueError:
            e_end = e_end_raw
        desc = note.get("description", "")
        lines.append(f"## {name}  ({s} – {e_end})")
        if desc:
            lines.append(desc)
        lines.append("")

    return "\n".join(lines)


@mcp.tool()
async def get_atp_week_note(
    week_date: str,
    athlete_id: str | None = None,
    api_key: str | None = None,
) -> str:
    """Get the ATP phase note covering the week that contains week_date.

    Returns which phase the week falls in and its training guidelines
    (focus, intensity, TSS target). Use this before planning a week's workouts.

    Args:
        week_date: Any date within the target week in YYYY-MM-DD format
        athlete_id: The Intervals.icu athlete ID (optional)
        api_key: The Intervals.icu API key (optional)
    """
    athlete_id_to_use, error_msg = resolve_athlete_id(athlete_id, config.athlete_id)
    if error_msg:
        return error_msg

    try:
        validate_date(week_date)
    except ValueError as e:
        return f"Error: {e}"

    monday = _monday_of(date.fromisoformat(week_date))
    sunday = monday + timedelta(days=6)

    result = await make_intervals_request(
        url=f"/athlete/{athlete_id_to_use}/events",
        api_key=api_key,
        params={"oldest": monday.isoformat(), "newest": sunday.isoformat()},
    )

    if isinstance(result, dict) and "error" in result:
        return f"Error fetching events: {result.get('message')}"

    events = result if isinstance(result, list) else []
    notes = [
        e for e in events
        if _is_atp_note(e)
    ]

    if not notes:
        return (
            f"No ATP note found for week {monday} – {sunday}. "
            "Use create_atp_plan to structure the season."
        )

    lines = [f"ATP for week {monday} – {sunday}:\n"]
    for note in notes:
        name = note.get("name", "")
        desc = note.get("description", "")
        lines.append(f"### {name}\n{desc}" if name else desc)

    return "\n\n".join(lines)


@mcp.tool()
async def get_planning_context(
    week_date: str,
    athlete_id: str | None = None,
    api_key: str | None = None,
) -> str:
    """Get full planning context for a training week.

    Returns in one call:
    - The ATP phase note for this week (focus, TSS target, key sessions)
    - Current fitness: CTL, ATL, TSB, HRV, ramp rate
    - Workouts already scheduled this week
    - Upcoming races (next 120 days)

    Use this before planning the week's workouts, then use add_or_update_event
    to post each day's workout.

    Args:
        week_date: Any date within the target week in YYYY-MM-DD format
        athlete_id: The Intervals.icu athlete ID (optional)
        api_key: The Intervals.icu API key (optional)
    """
    athlete_id_to_use, error_msg = resolve_athlete_id(athlete_id, config.athlete_id)
    if error_msg:
        return error_msg

    try:
        validate_date(week_date)
    except ValueError as e:
        return f"Error: {e}"

    monday = _monday_of(date.fromisoformat(week_date))
    sunday = monday + timedelta(days=6)
    today = date.today()

    _raw = await asyncio.gather(
        make_intervals_request(
            url=f"/athlete/{athlete_id_to_use}/wellness",
            api_key=api_key,
            params={
                "oldest": (today - timedelta(days=14)).isoformat(),
                "newest": today.isoformat(),
            },
        ),
        make_intervals_request(
            url=f"/athlete/{athlete_id_to_use}/events",
            api_key=api_key,
            params={"oldest": monday.isoformat(), "newest": sunday.isoformat()},
        ),
        make_intervals_request(
            url=f"/athlete/{athlete_id_to_use}/events",
            api_key=api_key,
            params={
                "oldest": today.isoformat(),
                "newest": (today + timedelta(days=120)).isoformat(),
            },
        ),
        return_exceptions=True,
    )
    _fallback: list[Any] = []
    _api_warnings: list[str] = []
    _labels = ("wellness", "week events", "upcoming races")
    for _label, _raw_item in zip(_labels, _raw):
        if isinstance(_raw_item, asyncio.CancelledError):
            raise _raw_item
        if isinstance(_raw_item, Exception):
            _api_warnings.append(f"  • {_label}: {_raw_item}")
    wellness_result: dict[str, Any] | list[dict[str, Any]] = (
        _raw[0] if not isinstance(_raw[0], Exception) else _fallback
    )
    week_events: dict[str, Any] | list[dict[str, Any]] = (
        _raw[1] if not isinstance(_raw[1], Exception) else _fallback
    )
    races_result: dict[str, Any] | list[dict[str, Any]] = (
        _raw[2] if not isinstance(_raw[2], Exception) else _fallback
    )

    week_list = week_events if isinstance(week_events, list) else []
    sections: list[str] = [f"# Planning context: week {monday} – {sunday}\n"]

    # ATP phase note
    sections.append("## ATP phase")
    atp_notes = [
        e for e in week_list
        if _is_atp_note(e)
    ]
    if atp_notes:
        for n in atp_notes:
            name = n.get("name", "")
            desc = n.get("description", "")
            sections.append(f"**{name}**\n{desc}" if name else desc)
    else:
        sections.append(
            "No ATP note for this week. Use create_atp_plan to structure the season."
        )

    # Fitness
    sections.append("\n## Current fitness (last 14 days)")
    if isinstance(wellness_result, list) and wellness_result:
        latest = next(
            (e for e in reversed(wellness_result) if isinstance(e, dict) and e.get("ctl") is not None),
            None,
        )
        if latest:
            ctl = latest.get("ctl")
            atl = latest.get("atl")
            tsb = round(ctl - atl, 1) if ctl is not None and atl is not None else None
            form = ""
            if tsb is not None:
                form = " (fresh)" if tsb > 10 else (" (fatigued)" if tsb < -10 else " (neutral)")
            lines = []
            if ctl is not None:
                lines.append(f"CTL (fitness): {round(ctl, 1)}")
            if atl is not None:
                lines.append(f"ATL (fatigue): {round(atl, 1)}")
            if tsb is not None:
                lines.append(f"TSB (form): {tsb}{form}")
            hrv = latest.get("hrv")
            if hrv is not None:
                lines.append(f"HRV: {hrv}")
            ramp = latest.get("rampRate")
            if ramp is not None:
                lines.append(f"Ramp rate: {round(ramp, 1)} CTL/week")
            sections.append("\n".join(lines))
        else:
            sections.append("No CTL/ATL data available.")
    else:
        sections.append("No wellness data available.")

    # Planned workouts
    sections.append("\n## Planned workouts this week")
    workouts = [e for e in week_list if isinstance(e, dict) and e.get("category") == "WORKOUT"]
    if workouts:
        for w in workouts:
            d = (w.get("start_date_local") or "")[:10]
            name = w.get("name", "workout")
            wtype = w.get("type", "")
            mt = w.get("moving_time")
            dur = f" ({int(mt) // 60} min)" if mt else ""
            sections.append(f"- {d} [{wtype}] {name}{dur}")
    else:
        sections.append("No workouts scheduled yet.")

    # Upcoming races
    sections.append("\n## Upcoming races (120 days)")
    if isinstance(races_result, list) and races_result:
        races = [e for e in races_result if isinstance(e, dict) and e.get("category") in ("RACE_A", "RACE_B", "RACE_C")]
        if races:
            for r in races:
                d = (r.get("start_date_local") or "")[:10]
                name = r.get("name", "race")
                cat = r.get("category", "")
                sections.append(f"- {d}: {name} [{cat}]")
        else:
            sections.append("No races scheduled.")
    else:
        sections.append("No races scheduled.")

    if _api_warnings:
        sections.append("\n## ⚠ API fetch warnings (data may be incomplete)")
        sections.extend(_api_warnings)

    return "\n".join(sections)


@mcp.tool()
async def add_race_event(
    name: str,
    race_date: str,
    priority: str = "A",
    start_time: str = "10:00:00",
    sport: str = "Ride",
    duration_minutes: int | None = None,
    distance_km: float | None = None,
    expected_tss: int | None = None,
    athlete_id: str | None = None,
    api_key: str | None = None,
) -> str:
    """Add a race event to the Intervals.icu calendar.

    Creates an event with category RACE_A, RACE_B, or RACE_C. The priority
    determines how the ATP and HRV coach treat the event:
    - A: most important race — full taper, no heavy training the week before
    - B: important race — reduce load 2 days before
    - C: training race — treat as a hard training day

    Args:
        name: Name of the race (e.g. "Tour of Flanders")
        race_date: Date of the race in YYYY-MM-DD format
        priority: Race priority — A, B, or C (default A)
        start_time: Start time in HH:MM:SS format (default 10:00:00)
        sport: Sport type — Ride, Run, Swim, etc. (default Ride)
        duration_minutes: Expected race duration in minutes (optional)
        distance_km: Expected race distance in km (optional)
        expected_tss: Expected Training Stress Score for the race (optional)
        athlete_id: The Intervals.icu athlete ID (optional)
        api_key: The Intervals.icu API key (optional)
    """
    athlete_id_to_use, error_msg = resolve_athlete_id(athlete_id, config.athlete_id)
    if error_msg:
        return error_msg

    try:
        validate_date(race_date)
    except ValueError as e:
        return f"Error in race_date: {e}"

    priority_upper = priority.strip().upper()
    if priority_upper not in ("A", "B", "C"):
        return "Error: priority must be A, B, or C."

    try:
        datetime.strptime(start_time, "%H:%M:%S")
    except ValueError:
        return "Error: start_time must be in HH:MM:SS format (e.g. '10:00:00')."

    event_data: dict[str, Any] = {
        "category": f"RACE_{priority_upper}",
        "type": sport,
        "name": name,
        "start_date_local": f"{race_date}T{start_time}",
    }

    if duration_minutes is not None:
        event_data["moving_time"] = duration_minutes * 60
    if distance_km is not None:
        event_data["distance"] = round(distance_km * 1000)
    if expected_tss is not None:
        event_data["icu_training_load"] = expected_tss

    result = await make_intervals_request(
        url=f"/athlete/{athlete_id_to_use}/events",
        api_key=api_key,
        data=event_data,
        method="POST",
    )

    if isinstance(result, dict) and "error" in result:
        return f"Error adding race: {result.get('message')}"

    if not isinstance(result, dict):
        return f"Unexpected response adding race: {result}"

    event_id = result.get("id")
    saved_category = result.get("category", "not returned")
    id_str = f" (id: {event_id})" if event_id else ""
    category_ok = saved_category == f"RACE_{priority_upper}"
    category_note = "" if category_ok else f" — WARNING: API saved category as '{saved_category}', expected 'RACE_{priority_upper}'"
    return f"Race '{name}' added on {race_date} [RACE_{priority_upper}]{id_str}.{category_note}"
