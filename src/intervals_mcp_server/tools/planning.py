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
"""

import asyncio
from datetime import date, timedelta
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
        "label": "Voorbereiding",
        "color": "green",
        "focus": "Algemene conditie opbouwen. Aerobe basis leggen, kracht en techniek.",
        "intensity": "80% Z2, 15% sweetspot, 5% drempel.",
        "key_sessions": ["Lange Z2 duurritten", "Krachttraining", "Soepelheid / techniek"],
        "tss_factor": 0.60,
    },
    "base": {
        "label": "Basis",
        "color": "blue",
        "focus": "Aerobe motor versterken. Hoog volume Z2, sweetspot, drempel introductie.",
        "intensity": "65% Z2, 25% sweetspot, 10% drempel.",
        "key_sessions": ["Lange duurritten (3–5 uur)", "2× sweetspot per week", "1× drempelinterval"],
        "tss_factor": 0.80,
    },
    "build": {
        "label": "Opbouw",
        "color": "orange",
        "focus": "Wedstrijdspecifieke kwaliteiten. Drempel, VO2max en anaerobe capaciteit uitbouwen.",
        "intensity": "40% Z2, 30% drempel, 30% VO2max / anaeroob.",
        "key_sessions": ["2× VO2max of drempel per week", "Lange duurrit", "Wedstrijdsimulatie"],
        "tss_factor": 1.00,
    },
    "peak": {
        "label": "Piek",
        "color": "red",
        "focus": "Scherp worden voor de koers. Volume daalt, intensiteit blijft hoog. Tapering.",
        "intensity": "50% Z2, 25% drempel, 25% VO2max — kortere intervallen.",
        "key_sessions": ["Race-pace intervallen", "Activatierit 2 dagen voor koers", "Rustdag voor koers"],
        "tss_factor": 0.65,
    },
    "race": {
        "label": "Koers",
        "color": "red",
        "focus": "Presteren op de A-koers. Minimale training, volle focus op wedstrijd en herstel.",
        "intensity": "Wedstrijd + herstelritten.",
        "key_sessions": ["Activatierit daags voor koers", "Wedstrijd", "Actief herstel erna"],
        "tss_factor": 0.50,
    },
}

# ---------------------------------------------------------------------------
# Phase auto-selection logic (relative to goal CTL, not absolute thresholds)
# ---------------------------------------------------------------------------


def _determine_phases(total_weeks: int, current_ctl: float, goal_ctl: float) -> list[tuple[str, int]]:
    """Return ordered (phase_name, weeks) based on available weeks and CTL gap.

    Uses the ratio current_ctl / goal_ctl so thresholds scale with the event:
    - UCI belofte targeting CTL 110 with current 70 → big gap → needs base
    - Toertocht finisher targeting CTL 60 with current 50 → small gap → straight to build
    """
    peak_wks = min(2, max(1, total_weeks // 7))
    race_wks = 1
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
        if build_wks < 1:
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
    gap = round(goal_ctl - current_ctl)
    if total_weeks <= 5:
        return f"Slechts {total_weeks} weken beschikbaar — direct naar piek-fase."
    if ratio >= 0.90:
        return (
            f"CTL {round(current_ctl)} is al dicht bij doel-CTL {goal_ctl} "
            f"(gap: {gap}) — direct naar opbouwfase."
        )
    if ratio >= 0.70:
        return (
            f"CTL {round(current_ctl)} is {gap} punten onder doel-CTL {goal_ctl} "
            f"({round(ratio * 100)}%) — korte basisblok gevolgd door opbouwfase."
        )
    return (
        f"CTL {round(current_ctl)} is {gap} punten onder doel-CTL {goal_ctl} "
        f"({round(ratio * 100)}%) — voorbereiding en basisblok nodig voor opbouw."
    )


# ---------------------------------------------------------------------------
# Note content builders
# ---------------------------------------------------------------------------

def _monday_of(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _weeks_in(start: date, end: date) -> int:
    return max(1, ((end - start).days + 1 + 6) // 7)


def _recovery_week_numbers(total_weeks: int, cycle: int) -> list[int]:
    return [w for w in range(1, total_weeks + 1) if w % cycle == 0]


def _week_tss(phase: str, goal_tss: int, week: int, total_weeks: int, cycle: int) -> str:
    factor = _PHASES[phase]["tss_factor"]
    if week % cycle == 0:
        return f"~{round(goal_tss * factor * 0.60 / 50) * 50} TSS  ← herstelweek"
    load_weeks = [w for w in range(1, total_weeks + 1) if w % cycle != 0]
    idx = load_weeks.index(week) if week in load_weeks else 0
    prog = 0.85 + 0.15 * (idx / max(len(load_weeks) - 1, 1))
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

    lines = [
        f"Fase {phase_num}/{total_phases}: {p['label']}",
        f"Periode: {start} – {end} ({total_wks} weken)",
        f"Doel: {p['focus']}",
        f"Intensiteit: {p['intensity']}",
        "",
        "Hoofdtrainingen per week:",
        *[f"  • {s}" for s in p["key_sessions"]],
        "",
        "Weekschema:",
    ]
    for w in range(1, total_wks + 1):
        w_start = start + timedelta(weeks=w - 1)
        w_end = min(w_start + timedelta(days=6), end)
        lines.append(f"  Week {w} ({w_start} – {w_end}): {_week_tss(phase, goal_tss, w, total_wks, cycle)}")

    lines += ["", f"A-koers: {race_name} ({race_date})"]
    if phase_races:
        lines.append("Wedstrijden in deze fase:")
        for r in phase_races:
            lines.append(f"  • {r['date']} — {r['name']} [{r.get('priority','C')}]")

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
    - ≥ 90% of goal: already close → Build → Peak → Race
    - 70–90% of goal: moderate gap → Base → Build → Peak → Race
    - < 70% of goal: large gap  → Preparation → Base → Build → Peak → Race

    goal_ctl reference values (CTL needed at race day):
    - Finish a sportive / gran fondo:        50–70
    - Club-level amateur racer:              70–90
    - Competitive amateur / cat. 3–4:        90–110
    - Semi-pro / UCI belofte:               100–130
    - Pro continental / WorldTour:          130+

    Each phase is posted as a NOTE event spanning its full date range on the
    Intervals.icu calendar. The note contains training focus, intensity split,
    weekly TSS targets, key sessions, and recovery-week schedule.

    Args:
        race_date: Date of the A-race in YYYY-MM-DD format
        race_name: Name of the A-race (e.g. "Ronde van Vlaanderen")
        goal_ctl: Target CTL (fitness) needed at race day — see reference above
        recovery_cycle: Recovery week every N weeks — 3 for older/less experienced
                        athletes, 4 for younger/experienced athletes (default 4)
        additional_races: Optional B/C races, one per line as
                          "YYYY-MM-DD Name [A/B/C]"
                          e.g. "2026-03-21 Dwars door V. C"
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
    ctl_source = "onbekend (geen wellness-data)"
    if isinstance(wellness_result, list) and wellness_result:
        for entry in reversed(wellness_result):
            if isinstance(entry, dict) and entry.get("ctl") is not None:
                current_ctl = float(entry["ctl"])
                ctl_source = f"{round(current_ctl, 1)} (uit wellness {entry.get('id', '')})"
                break

    # Weekly TSS in peak build weeks ≈ goal_ctl × 7
    goal_weekly_tss = round(goal_ctl * 7 / 50) * 50

    # Parse additional races
    extra_races: list[dict] = []
    if additional_races:
        for line in additional_races.strip().splitlines():
            parts = line.strip().split(None, 1)   # split on first space only: [date, rest]
            if len(parts) == 2:
                name_part = parts[1].strip()
                priority = "C"
                if "[" in name_part:
                    priority = name_part.split("[")[-1].strip("] ").upper()
                    name_part = name_part.split("[")[0].strip()
                extra_races.append({"date": parts[0], "name": name_part, "priority": priority})

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
    posted_lines: list[str] = []

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
            "end_date_local": ph["end"].isoformat() + "T00:00:00",
            "color": _PHASES[ph["name"]]["color"],
        }

        result = await make_intervals_request(
            url=f"/athlete/{athlete_id_to_use}/events",
            api_key=api_key,
            data=event_data,
            method="POST",
        )

        label = _PHASES[ph["name"]]["label"]
        if isinstance(result, dict) and "error" in result:
            posted_lines.append(f"  ✗ {label} ({ph['start']} – {ph['end']}): {result.get('message')}")
        else:
            posted_lines.append(f"  ✓ {label}: {ph['start']} – {ph['end']} ({ph['weeks']} wk)")

    phase_names = " → ".join(_PHASES[ph["name"]]["label"] for ph in phase_ranges)
    summary = [
        f"ATP aangemaakt voor {race_name} ({race_date})",
        f"Periode: {today_monday} – {race_monday + timedelta(days=6)} ({total_weeks} weken)",
        f"Huidige CTL: {ctl_source}",
        f"Doel-CTL: {goal_ctl}  →  TSS-doel piekweek: ~{goal_weekly_tss}",
        f"Reden fasekeuze: {reason}",
        f"Fasen: {phase_names}",
        f"Herstelcyclus: elke {recovery_cycle} weken",
        "",
        "Gepost op kalender:",
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
    notes = [e for e in events if e.get("category") == "NOTE"]

    if not notes:
        return (
            f"Geen ATP-notities gevonden tussen {start_date} en {end_date}. "
            "Gebruik create_atp_plan om een plan aan te maken."
        )

    lines = [f"ATP-overzicht ({start_date} – {end_date})\n"]
    for note in notes:
        name = note.get("name", "")
        s = (note.get("start_date_local") or "")[:10]
        e_end = (note.get("end_date_local") or note.get("start_date_local") or "")[:10]
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
    notes = [e for e in events if e.get("category") == "NOTE"]

    if not notes:
        return (
            f"Geen ATP-notitie gevonden voor week {monday} – {sunday}. "
            "Gebruik create_atp_plan om het seizoen te structureren."
        )

    lines = [f"ATP voor week {monday} – {sunday}:\n"]
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
    wellness_result: dict[str, Any] | list[dict[str, Any]] = (
        _raw[0] if not isinstance(_raw[0], BaseException) else _fallback
    )
    week_events: dict[str, Any] | list[dict[str, Any]] = (
        _raw[1] if not isinstance(_raw[1], BaseException) else _fallback
    )
    races_result: dict[str, Any] | list[dict[str, Any]] = (
        _raw[2] if not isinstance(_raw[2], BaseException) else _fallback
    )

    week_list = week_events if isinstance(week_events, list) else []
    sections: list[str] = [f"# Planningcontext: week {monday} – {sunday}\n"]

    # ATP phase note
    sections.append("## ATP-fase")
    atp_notes = [e for e in week_list if e.get("category") == "NOTE"]
    if atp_notes:
        for n in atp_notes:
            name = n.get("name", "")
            desc = n.get("description", "")
            sections.append(f"**{name}**\n{desc}" if name else desc)
    else:
        sections.append(
            "Geen ATP-notitie voor deze week. Gebruik create_atp_plan om het seizoen te structureren."
        )

    # Fitness
    sections.append("\n## Conditie (afgelopen 14 dagen)")
    if isinstance(wellness_result, list) and wellness_result:
        latest = next((e for e in reversed(wellness_result) if isinstance(e, dict) and e.get("ctl") is not None), None)
        if latest:
            ctl = latest.get("ctl")
            atl = latest.get("atl")
            tsb = round(ctl - atl, 1) if ctl is not None and atl is not None else None
            form = ""
            if tsb is not None:
                form = " (uitgerust)" if tsb > 10 else (" (vermoeid)" if tsb < -10 else " (neutraal)")
            lines = []
            if ctl is not None:
                lines.append(f"CTL (fitness): {round(ctl, 1)}")
            if atl is not None:
                lines.append(f"ATL (vermoeidheid): {round(atl, 1)}")
            if tsb is not None:
                lines.append(f"TSB (vorm): {tsb}{form}")
            hrv = latest.get("hrv")
            if hrv is not None:
                lines.append(f"HRV: {hrv}")
            ramp = latest.get("rampRate")
            if ramp is not None:
                lines.append(f"Ramp rate: {round(ramp, 1)} CTL/week")
            sections.append("\n".join(lines))
        else:
            sections.append("Geen CTL/ATL beschikbaar.")
    else:
        sections.append("Geen wellness-data beschikbaar.")

    # Planned workouts
    sections.append("\n## Geplande workouts deze week")
    workouts = [e for e in week_list if e.get("category") == "WORKOUT"]
    if workouts:
        for w in workouts:
            d = (w.get("start_date_local") or "")[:10]
            name = w.get("name", "workout")
            wtype = w.get("type", "")
            mt = w.get("moving_time")
            dur = f" ({int(mt) // 60} min)" if mt else ""
            sections.append(f"- {d} [{wtype}] {name}{dur}")
    else:
        sections.append("Nog geen workouts ingepland.")

    # Upcoming races
    sections.append("\n## Komende wedstrijden (120 dagen)")
    if isinstance(races_result, list) and races_result:
        races = [e for e in races_result if e.get("category") in ("RACE", "A", "B", "C")]
        if races:
            for r in races:
                d = (r.get("start_date_local") or "")[:10]
                name = r.get("name", "wedstrijd")
                cat = r.get("category", "")
                sections.append(f"- {d}: {name} [{cat}]")
        else:
            sections.append("Geen wedstrijden gepland.")
    else:
        sections.append("Geen wedstrijden gepland.")

    return "\n".join(sections)
