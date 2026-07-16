#!/usr/bin/env python3
"""Top-3 inferences per battery item from the merged 5-seed gibbs+bd posterior.

Reads the merged item records (evidence-weighted across 5 seeds) and writes a
readable markdown report of the top-3 hypotheses + their posterior probabilities,
labelled with item_id/family/condition and the expected behavior, with markers
for which hypothesis equals the intended / observed sentence.
"""
import json, glob, csv, os

BASE = 'results_nc/calibrationbatteryv0'
BD   = 'lm-pythia-70m__ch-align__rej-gibbsbd__P256__b2__d2__lb6__s0__nseed5'
OUT  = 'planning/calibration_bd_top3.md'


def norm(s):
    return ' '.join(s.strip().split())


def main():
    by_obs = {norm(r['observed']): r
              for r in csv.DictReader(open('planning/calibration_battery_v0.csv'))}

    recs = []
    for f in glob.glob(f'{BASE}/{BD}/results/item_[0-9]*.json'):
        if '_s' in os.path.basename(f):
            continue
        recs.append(json.load(open(f)))

    # order by item_id (family-grouped, stable) via the CSV row order
    csv_order = {norm(r['observed']): i
                 for i, r in enumerate(csv.DictReader(open('planning/calibration_battery_v0.csv')))}
    recs.sort(key=lambda r: csv_order.get(norm(r['observed']), 1e9))

    lines = ['# Battery — top-3 inferences (merged 5-seed gibbs+bd posterior)',
             '',
             f'Channel align, P=256, gibbs+bd indel rejuv, 5 seeds (evidence-weighted merge).  '
             f'{len(recs)} items.',
             '',
             'Per item: **observed** (model input) → **intended** (the gold restoration), '
             'the expected behavior, then the 3 highest-probability hypotheses.  '
             'Markers: `←intended` = matches the gold restoration, `←observed` = equals the input (a no-op/keep).',
             '']

    for r in recs:
        observed = norm(r['observed'])
        meta = by_obs.get(observed, {})
        item_id = meta.get('item_id', f'idx{r["idx"]}')
        family  = meta.get('family', '?')
        cond    = meta.get('condition', '?')
        expected = meta.get('expected', '?')
        intended = norm(meta.get('intended', ''))
        spread = r.get('logZ_stats', {}).get('spread', '?')

        lines.append(f'## {item_id}  ·  {family} / {cond} / expect-{expected}')
        lines.append(f'- observed: `{observed}`')
        lines.append(f'- intended: `{intended}`')
        lines.append(f'- logZ {r.get("logZ", float("nan")):.2f}, seed-spread {spread} nats')
        lines.append('')
        for rank, h in enumerate(r['hypotheses'][:3], 1):
            sent = norm(h['sentence'])
            mark = ''
            if sent == intended:
                mark = '  ←intended'
            elif sent == observed:
                mark = '  ←observed'
            lines.append(f'  {rank}. `{h["prob"]:.3f}`  {sent}{mark}')
        lines.append('')

    with open(OUT, 'w') as fh:
        fh.write('\n'.join(lines))
    print(f'wrote {OUT}  ({len(recs)} items)')


if __name__ == '__main__':
    main()
