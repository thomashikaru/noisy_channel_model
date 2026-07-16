#!/usr/bin/env python3
"""Diff the gibbs+bd nseed5 calibration run against the rej-off nseed5 run.

Open hypothesis (per calibration-coverage-runs-orcd memory): does the indel
rejuvenation (gibbs+bd) shift the DEL_*/INS_* family behavior that rejuv=off
leaves unedited?  Both runs used the SAME 87-item input (idx 0-86, identical
observed strings), so we join the two result sets on idx and label families by
joining each item's `observed` string to planning/calibration_battery_v0.csv.

Treat as a stochastic eval (5-seed evidence-weighted merge): report aggregate
shifts, do NOT overfit individual items.
"""
import json, glob, csv, os, collections

BASE = 'results_nc/calibrationbatteryv0'
OFF  = 'lm-pythia-70m__ch-align__rej-off__P256__b2__d2__lb6__s0__nseed5'
BD   = 'lm-pythia-70m__ch-align__rej-gibbsbd__P256__b2__d2__lb6__s0__nseed5'


def load_merged(slug):
    recs = {}
    for f in glob.glob(f'{BASE}/{slug}/results/item_[0-9]*.json'):
        if '_s' in os.path.basename(f):      # skip per-seed records
            continue
        r = json.load(open(f))
        recs[r['idx']] = r
    return recs


def norm(s):
    return ' '.join(s.strip().split())


def main():
    off = load_merged(OFF)
    bd  = load_merged(BD)
    csv_rows = list(csv.DictReader(open('planning/calibration_battery_v0.csv')))
    by_obs = {norm(r['observed']): r for r in csv_rows}

    rows = []
    for idx in sorted(set(off) & set(bd)):
        o, b = off[idx], bd[idx]
        meta = by_obs.get(norm(o['observed']))
        if meta is None:
            continue
        observed = norm(o['observed'])
        intended = norm(meta['intended'])
        expected = meta['expected']                     # 'edit' or 'keep'
        target   = intended if expected == 'edit' else observed
        off_map, bd_map = norm(o['map']), norm(b['map'])

        def ok(m):                                       # matches expected behavior
            return int(m == target)

        def edited(m):                                   # MAP departs from observed
            return int(m != observed)

        rows.append(dict(
            item_id=meta['item_id'], family=meta['family'],
            condition=meta['condition'], expected=expected,
            observed=observed, intended=intended,
            off_map=off_map, bd_map=bd_map,
            off_edited=edited(off_map), bd_edited=edited(bd_map),
            off_ok=ok(off_map), bd_ok=ok(bd_map),
            changed=int(off_map != bd_map),
        ))

    out = 'planning/calibration_bd_vs_off.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'wrote {out}  ({len(rows)} items)\n')

    # ---- overall ----
    n = len(rows)
    off_ok = sum(r['off_ok'] for r in rows)
    bd_ok  = sum(r['bd_ok']  for r in rows)
    changed = sum(r['changed'] for r in rows)
    print(f'OVERALL  n={n}')
    print(f'  rej-off  matches-expected: {off_ok}/{n}')
    print(f'  gibbs+bd matches-expected: {bd_ok}/{n}   (delta {bd_ok-off_ok:+d})')
    print(f'  MAP changed by gibbs+bd:   {changed}/{n}')
    # net correctness flips
    gained = [r for r in rows if r['bd_ok'] and not r['off_ok']]
    lost   = [r for r in rows if r['off_ok'] and not r['bd_ok']]
    print(f'  flips: {len(gained)} newly-correct, {len(lost)} newly-wrong')

    # ---- by family (focus on edit-expected families) ----
    fam = collections.defaultdict(lambda: dict(n=0, off_ok=0, bd_ok=0,
                                               off_ed=0, bd_ed=0, chg=0))
    for r in rows:
        f = fam[r['family']]
        f['n'] += 1
        f['off_ok'] += r['off_ok']; f['bd_ok'] += r['bd_ok']
        f['off_ed'] += r['off_edited']; f['bd_ed'] += r['bd_edited']
        f['chg'] += r['changed']
    print('\nBY FAMILY  (ok = matches expected; ed = edited away from observed)')
    print(f'  {"family":<10} {"n":>2}  {"off_ok":>6} {"bd_ok":>6}   {"off_ed":>6} {"bd_ed":>6}  {"chg":>3}')
    for k in sorted(fam):
        f = fam[k]
        print(f'  {k:<10} {f["n"]:>2}  {f["off_ok"]:>6} {f["bd_ok"]:>6}   '
              f'{f["off_ed"]:>6} {f["bd_ed"]:>6}  {f["chg"]:>3}')

    # ---- the open hypothesis: DEL_*/INS_* edit-expected items ----
    print('\nINDEL FAMILIES (DEL*/INS*), edit-expected items where behavior changed:')
    for r in rows:
        if not (r['family'].startswith('DEL') or r['family'].startswith('INS')):
            continue
        if r['expected'] != 'edit':
            continue
        if r['changed'] or r['off_edited'] != r['bd_edited']:
            tag = 'GAIN' if (r['bd_ok'] and not r['off_ok']) else \
                  ('LOSS' if (r['off_ok'] and not r['bd_ok']) else 'shift')
            print(f'  [{tag}] {r["item_id"]:<10} obs: {r["observed"]}')
            print(f'         off -> {r["off_map"]}')
            print(f'         bd  -> {r["bd_map"]}')
            print(f'         intended: {r["intended"]}')


if __name__ == '__main__':
    main()
