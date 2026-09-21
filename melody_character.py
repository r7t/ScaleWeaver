"""Soft melodic-shape and sixteenth-cell preferences for the existing engine."""
import math


INTERVAL_TARGETS = (.02, .32, .32, .25, .09)


def shape_cost(recent, candidate, degree, duration=1.):
    """Smoothed local feedback, not forced intervals or copied source pitches."""
    if not recent:
        return 0.
    pitches = list(recent[-25:])
    intervals = [degree(b)-degree(a) for a, b in zip(pitches, pitches[1:])]
    jump = degree(candidate)-degree(pitches[-1])
    category = min(4, abs(jump))
    target = INTERVAL_TARGETS[category]
    observed = (sum(min(4, abs(d))==category for d in intervals)+8*target)/(len(intervals)+8)
    cost = observed-target
    # Repeated attacks are rarer in the reference; ties never enter this call.
    if jump == 0:
        cost += .18
    if intervals and jump and intervals[-1]:
        pairs = [(a, b) for a, b in zip(intervals, intervals[1:]) if a and b]
        turn_rate = (sum(a*b<0 for a, b in pairs)+8*.43)/(len(pairs)+8)
        cost += .35*(turn_rate-.43)*(int(jump*intervals[-1]<0)-.43)
    # Rapid ornaments retain the engine's own small-step preference.
    return cost * (.3 if duration <= .250001 else 1.)


def sixteenth_cells(durations):
    """Count exact 16th-unit 3+1, 1+3 and 1+2+1 duration sequences."""
    ticks = [round(float(d)*4, 6) for d in durations]
    return {name: sum(tuple(ticks[i:i+len(pattern)])==pattern
                      for i in range(len(ticks)-len(pattern)+1))
            for name, pattern in [('3+1', (3, 1)), ('1+3', (1, 3)), ('1+2+1', (1, 2, 1))]}


def rhythm_preference(durations, weights=None):
    if not weights or all(float(v)==1. for v in weights.values()):
        return 1.
    counts = sixteenth_cells(durations)
    log_weight = sum(counts[k]*math.log(float(weights.get(k, 1.))) for k in counts)
    return math.exp(max(-2., min(2., log_weight)))
