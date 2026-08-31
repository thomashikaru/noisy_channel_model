#!/usr/bin/env python3
"""Diff the lookahead-charge battery run against the plain off-arm run (gate 7 of
planning/LOOKAHEAD_CHARGE_PLAN.md — "did anything else move").

Both runs: 87-item battery, align, rejuv=off, P=64, N_SEEDS=2, band 2 — identical except
LOOKAHEAD. Join on idx (same input file), label via planning/calibration_battery_v0.csv, using
the SAME matching conventions as planning/bd_vs_off_diff.py (target = intended when
expected=='edit' else observed; exact + case-insensitive match). Adds the lookahead-specific
columns: unit-0 del_before (the artifact detector), p_literal, logZ, and the 2-seed logZ spread
(the heavier-tails concern). Stochastic eval — report aggregate shifts, don't overfit items.

Writes planning/calibration_la_vs_off.csv and prints the summary.
"""
import json, glob, csv, os, collections

BASE = 'results_nc/calibrationbatteryv0'
OFF = 'lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__nseed2'
LA = 'lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__la__nseed2'


def load_merged(slug):
    recs = {}
    for f in glob.glob(f'{BASE}/{slug}/results/item_[0-9]*.json'):
        if '_s' in os.path.basename(f):
            continue
        r = json.load(open(f))
        recs[r['idx']] = r
    return recs


def norm(s):
    return ' '.join(s.strip().split())


def del0(r):
    w = r.get('words') or {}
    us = w.get('units') or []
    v = us[0].get('del_before') if us else None
    return float(v) if v is not None else None


def main():
    off = load_merged(OFF)
    la = load_merged(LA)
    csv_rows = list(csv.DictReader(open('planning/calibration_battery_v0.csv')))
    by_obs = {norm(r['observed']): r for r in csv_rows}

    rows = []
    for idx in sorted(set(off) & set(la)):
        o, l = off[idx], la[idx]
        meta = by_obs.get(norm(o['observed']))
        if meta is None or o.get('status') != 'ok' or l.get('status') != 'ok':
            continue
        observed = norm(o['observed'])
        target = norm(meta['intended']) if meta['expected'] == 'edit' else observed
        om, lm = norm(o['map']), norm(l['map'])
        rows.append(dict(
            item_id=meta['item_id'], family=meta['family'], expected=meta['expected'],
            observed=observed, target=target, off_map=om, la_map=lm,
            off_ok=int(om == target), la_ok=int(lm == target),
            off_ok_ci=int(om.lower() == target.lower()), la_ok_ci=int(lm.lower() == target.lower()),
            off_edited=int(om != observed), la_edited=int(lm != observed),
            off_del0=del0(o), la_del0=del0(l),
            off_plit=o.get('p_literal'), la_plit=l.get('p_literal'),
            off_logZ=o['logZ'], la_logZ=l['logZ'], dlogZ=l['logZ'] - o['logZ'],
            off_spread=(o.get('logZ_stats') or {}).get('spread'),
            la_spread=(l.get('logZ_stats') or {}).get('spread'),
        ))

    n = len(rows)
    print(f'joined items: {n} (off {len(off)}, la {len(la)})')

    def tot(k):
        return sum(r[k] for r in rows)

    print(f'\nmatches-expected      off {tot("off_ok")}/{n}   la {tot("la_ok")}/{n}')
    print(f'matches-expected (ci) off {tot("off_ok_ci")}/{n}   la {tot("la_ok_ci")}/{n}')
    print(f'edited (MAP != observed) off {tot("off_edited")}/{n}   la {tot("la_edited")}/{n}')

    art_off = [r for r in rows if (r['off_del0'] or 0) > 0.5]
    art_la = [r for r in rows if (r['la_del0'] or 0) > 0.5]
    print(f'\nunit-0 del_before > 0.5 (leading-deletion artifact): off {len(art_off)}  la {len(art_la)}')
    for r in art_la:
        print(f'  la still artifacted: {r["item_id"]} del0 {r["la_del0"]:.2f} (off {r["off_del0"]:.2f}) map {r["la_map"][:50]!r}')

    dz = sorted(r['dlogZ'] for r in rows)
    mean = sum(dz) / n
    print(f'\nlogZ shift (la - off): mean {mean:+.2f}  median {dz[n // 2]:+.2f}  '
          f'up {sum(1 for d in dz if d > 0.5)}  down {sum(1 for d in dz if d < -0.5)}  ~flat {sum(1 for d in dz if abs(d) <= 0.5)}')
    movers = sorted(rows, key=lambda r: -abs(r['dlogZ']))[:8]
    for r in movers:
        print(f'  {r["item_id"]:>9} dlogZ {r["dlogZ"]:+7.2f}  del0 {r["off_del0"]:.2f}->{r["la_del0"]:.2f}  '
              f'ok {r["off_ok"]}->{r["la_ok"]}')

    sp = [(r['off_spread'], r['la_spread']) for r in rows if r['off_spread'] is not None and r['la_spread'] is not None]
    print(f'\n2-seed logZ spread: mean off {sum(a for a, _ in sp) / len(sp):.2f}  la {sum(b for _, b in sp) / len(sp):.2f}  '
          f'la>off on {sum(1 for a, b in sp if b > a)}/{len(sp)}')

    changed = [r for r in rows if r['off_map'] != r['la_map']]
    gained = [r for r in changed if r['la_ok'] and not r['off_ok']]
    lost = [r for r in changed if r['off_ok'] and not r['la_ok']]
    print(f'\nMAP changed on {len(changed)} items: newly-correct {len(gained)}, newly-wrong {len(lost)}')
    fam = collections.Counter()
    for r in changed:
        fam[r['family']] += 1
    print('  changes by family:', dict(sorted(fam.items())))
    for tag, group in (('GAINED', gained), ('LOST', lost)):
        for r in group:
            print(f'  {tag} {r["item_id"]:>9} [{r["expected"]}] off {r["off_map"][:44]!r} -> la {r["la_map"][:44]!r}')

    out = 'planning/calibration_la_vs_off.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
