# Manual progression phrase planning

When a chord progression is supplied explicitly, its complete cycle defines one
phrase.  The phrase length is the number of progression entries multiplied by
`bars_per_chord`.

- The requested score length must contain a whole number of cycles.  For
  example, a seven-bar cycle accepts 7, 14, 21, ... bars and rejects 48 bars.
- Each phrase node has one direct child per bar.  Manual mode does not create
  the automatic two- and four-bar binary subdivisions.
- Relations inside a phrase may copy rhythm (`R`) but never melody (`M`).
- Relations between phrases may still copy melody.  Source and target units
  retain the same position within their respective progression cycles.
- The supplied harmony is authoritative: automatic cadence anchoring does not
  replace its final chord.

Automatic harmony mode retains the original eight-bar phrase planning.
