# Simultaneous-attack CSE percentile soft limits

In this revision, **CSE** in an attack limit means the configured composite:

```text
CSE_objective = a*CSE + b*SE + c*CCSE
```

The coefficients are `CSE_2D_A`, `CSE_2D_B`, and `CSE_2D_C` in
`style.cse_weights`. The percentile is recomputed from the complete blended
distribution for the matching 2-, 3-, or 4-note native table. It is not the
pure-CSE percentile and it is not a linear combination of three percentiles.

## Configuration

Add all three fractional percentiles to the style JSON:

```json
"attack_cse_percentile_limits": {
  "2": 0.50,
  "3": 0.50,
  "4": 0.50
}
```

Values range from `0.0` to `1.0`; `0.50` means the 50th percentile. Omitting
the whole object disables the feature for backward compatibility. When the
object exists, all three keys are required.

## Exact behavior

Only notes whose `start_beat` is the same participate. Sustained notes do not.
For a three-note attack the program checks all three dyads and the triad. For a
four-note attack it checks all six dyads, all four triads, and the tetrad.

The limits are implemented as a relaxable gate:

1. If at least one otherwise legal pitch satisfies every newly formed subset,
   only fully compliant pitches proceed to musical scoring and sampling.
2. If no compliant pitch exists, the program retains the candidates with the
   least maximum percentile excess, plus a 0.02 percentile slack, then applies
   a continuous excess cost inside that fallback layer.
3. The soft-limit stage never returns an empty set. Existing hard rules remain
   the only possible source of a rejected generation attempt.
4. A rejected generation attempt is retried deterministically up to eight
   times with recorded rescue seeds, making total failure extremely unlikely.

The saved score contains `simultaneous_attack_cse_limits`, including weights,
limits, counts by cardinality, maxima, and a small list of any unavoidable
final excesses.

## Cleanup default

Cleanup is disabled by default in both the Python API and CLI. Use `--cleanup`
to enable it explicitly. The attack limits are enforced during generation;
they do not depend on cleanup.
