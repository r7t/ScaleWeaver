"""Soft melodic-shape and sixteenth-cell preferences for the existing engine."""
import math


INTERVAL_TARGETS = (.02, .32, .32, .25, .09)


def shape_statistics(recent, degree):
    """Candidate-independent history, suitable for a scale-local cache."""
    pitches=list(recent[-25:])
    degrees=[degree(p) for p in pitches]
    intervals=[b-a for a,b in zip(degrees,degrees[1:])]
    counts=[0]*5
    for d in intervals:counts[min(4,abs(d))]+=1
    pairs=[(a,b) for a,b in zip(intervals,intervals[1:]) if a and b]
    turn_rate=(sum(a*b<0 for a,b in pairs)+8*.43)/(len(pairs)+8)
    return (degrees[-1],len(intervals),tuple(counts),
            intervals[-1] if intervals else 0,turn_rate)


def shape_cost(recent, candidate, degree, duration=1., statistics=None):
    """Smoothed local feedback, not forced intervals or copied source pitches."""
    if not recent:
        return 0.
    last,n,counts,previous,turn_rate = (statistics if statistics is not None
                                       else shape_statistics(recent,degree))
    jump = degree(candidate)-last
    category = min(4, abs(jump))
    target = INTERVAL_TARGETS[category]
    observed = (counts[category]+8*target)/(n+8)
    cost = observed-target
    # Repeated attacks are rarer in the reference; ties never enter this call.
    if jump == 0:
        cost += .18
    if n and jump and previous:
        cost += .35*(turn_rate-.43)*(int(jump*previous<0)-.43)
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
