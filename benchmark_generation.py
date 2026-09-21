"""Time generation without audio/export; compare hashes across revisions."""
import argparse
import hashlib
import json
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontend-only', action='store_true')
    parser.add_argument('--bars', type=int, default=48)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--harmony-model', choices=('three-tone', 'legacy'), default='three-tone')
    parser.add_argument('--scale', default='scales/tiangan_72.json')
    parser.add_argument('--rules', default='scales/tiangan_72_5.rules.json')
    parser.add_argument('--style', default='scales/tiangan_72_5_norm.style.json')
    parser.add_argument('--cse-dir', default='CSE_cache')
    args = parser.parse_args()
    import main as generator
    from joint_frontend_generate import generate_frontend_ir
    from accompaniment_generate import generate_from_ir

    start = time.perf_counter()
    spec, _ = generator._configure_adaptive_scale(
        args.scale, args.cse_dir, rules_config=args.rules, style_config=args.style,
        harmony_model=args.harmony_model)
    configured = time.perf_counter()
    frontend = generate_frontend_ir(seed=args.seed, bars=args.bars, spec=spec)
    generated = time.perf_counter()
    result = {'bars': args.bars, 'seed': args.seed, 'harmony_model': args.harmony_model,
              'configure_seconds': configured-start,
              'frontend_seconds': generated-configured,
              'frontend_sha256': digest(frontend)}
    if not args.frontend_only:
        accompaniment_start = time.perf_counter()
        score = generate_from_ir(frontend, spec)
        result['accompaniment_seconds'] = time.perf_counter()-accompaniment_start
        result['voices_sha256'] = digest(score['voices'])
    result['total_seconds'] = time.perf_counter()-start
    print(json.dumps(result, ensure_ascii=False, indent=2))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


if __name__ == '__main__':
    main()
