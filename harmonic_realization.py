"""Musical targets separate from the CSE reference voicing (no CSE tables)."""
def tonic_pc(hr):
    return int(hr.SCALE.resolved_harmony()['pitch_roles']['tonic'][0]) % hr.OCT


def musical_root(hr, segment):
    chord = hr.CHORDS[segment['chord_id']]
    overrides = (hr.SCALE.metadata or {}).get('harmony_style', {}).get('root_pcs', {})
    if chord.id in overrides:
        return int(overrides[chord.id]) % hr.OCT
    if 'root_pc' in segment:
        return int(segment['root_pc']) % hr.OCT
    # Recognized natural triads: tertian spelling, not minimum-CSE inversion.
    if hr.NATURAL_SCALE_MODE or chord.id.startswith('MANUAL_TRIAD_'):
        for i, pc in enumerate(hr.PCS):
            if set(chord.pcs) == {hr.PCS[(i+j) % len(hr.PCS)] for j in (0,2,4)}:
                return pc
    # Non-tertian scales retain their declared chord foot unless overridden.
    return chord.foot % hr.OCT


def is_anchor(hr, segment):
    pcs = set(hr.CHORDS[segment['chord_id']].pcs)
    if segment.get('harmony_model') == 'three_tone':
        return (segment.get('phrase_role') in ('cadence', 'establish') and
                tonic_pc(hr) in pcs)
    return any(pcs == set(hr.CHORDS[cid].pcs) for cid in hr.ANCHOR_CHORD_IDS)


def realization_cost(pitches, voices, pcs, root, edo, config):
    """Coverage is primarily accompaniment-owned; Lead is immutable context.

    Capacity-normalized coverage allows tetrads with only three accompanists.
    Joint coverage asks accompanists to fill tones missing from the Lead.
    """
    target = set(pcs)
    accomp = [p % edo for p,v in zip(pitches,voices) if v != 'lead']
    if not target or not accomp:
        return 0.0
    covered = target.intersection(accomp)
    capacity = min(len(target),len(accomp))
    missing = (capacity-len(covered))/capacity
    foreign = sum(p not in target for p in accomp)/len(accomp)
    bass = next((p % edo for p,v in zip(pitches,voices) if v == 'bass'),None)
    union = {p % edo for p in pitches}
    joint_capacity = min(len(target),len(pitches))
    joint_missing = (joint_capacity-len(target.intersection(union)))/joint_capacity
    return (config['bass_root'] * (bass is not None and bass != root)
            + config['accompaniment_coverage'] * missing
            + config['accompaniment_membership'] * foreign
            + config['joint_coverage'] * joint_missing)
