#!/usr/bin/env python3
"""Diff the lookahead-IN-PROPOSAL battery run against the lookahead-at-resampling-only run
(planning/OFF_ARM_INFERENCE_FIX.md sec 6 decision 1 — the pre-committed A/B).

Both runs: 87-item battery, align, rejuv=off, P=64, N_SEEDS=4, band 2 — identical except
LA_PROPOSAL. Join on idx (same input file), label via planning/calibration_battery_v0.csv, using
the SAME matching conventions as planning/la_vs_off_diff.py (target = intended when
expected=='edit' else observed; exact + case-insensitive match). The pre-committed criteria:
(a) artifact items (unit-0 del_before > 0.5) cleared, (b) genuine repairs retained
(matches-expected on expected=='edit'), (c) overall edit rate not worse, (d) logZ up on the
same model. Stochastic eval — report aggregate shifts, don't overfit items.

Writes planning/calibration_lap_vs_la.csv and prints the summary.
"""
import json, glob, csv, os, collections

BASE = 'results_nc/calibrationbatteryv0'
LA = 'lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__la__nseed4'
LAP = 'lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__la__lap__nseed4'


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


def del_any(r):
    w = r.get('words') or {}
    us = w.get('units') or []
    vs = [u.get('del_before') for u in us if u.get('del_before') is not None]
    return max((float(v) for v in vs), default=None)


def main():
    la = load_merged(LA)
    lap = load_merged(LAP)
    csv_rows = list(csv.DictReader(open('planning/calibration_battery_v0.csv')))
    by_obs = {norm(r['observed']): r for r in csv_rows}

    rows = []
    for idx in sorted(set(la) & set(lap)):
        o, l = la[idx], lap[idx]
        meta = by_obs.get(norm(o['observed']))
        if meta is None or o.get('status') != 'ok' or l.get('status') != 'ok':
            continue
        observed = norm(o['observed'])
        target = norm(meta['intended']) if meta['expected'] == 'edit' else observed
        om, lm = norm(o['map']), norm(l['map'])
        rows.append(dict(
            item_id=meta['item_id'], family=meta['family'], expected=meta['expected'],
            observed=observed, target=target, la_map=om, lap_map=lm,
            la_ok=int(om == target), lap_ok=int(lm == target),
            la_ok_ci=int(om.lower() == target.lower()), lap_ok_ci=int(lm.lower() == target.lower()),
            la_edited=int(om != observed), lap_edited=int(lm != observed),
            la_del0=del0(o), lap_del0=del0(l),
            la_delany=del_any(o), lap_delany=del_any(l),
            la_plit=o.get('p_literal'), lap_plit=l.get('p_literal'),
            la_logZ=o['logZ'], lap_logZ=l['logZ'], dlogZ=l['logZ'] - o['logZ'],
            la_spread=(o.get('logZ_stats') or {}).get('spread'),
            lap_spread=(l.get('logZ_stats') or {}).get('spread'),
        ))

    n = len(rows)
    print(f'joined items: {n} (la {len(la)}, lap {len(lap)})')

    def tot(k):
        return sum(r[k] for r in rows)

    print(f'\nmatches-expected      la {tot("la_ok")}/{n}   lap {tot("lap_ok")}/{n}')
    print(f'matches-expected (ci) la {tot("la_ok_ci")}/{n}   lap {tot("lap_ok_ci")}/{n}')
    print(f'edited (MAP != observed) la {tot("la_edited")}/{n}   lap {tot("lap_edited")}/{n}   (criterion c)')

    ed = [r for r in rows if r['expected'] == 'edit']
    print(f'\ncriterion (b) genuine repairs, expected==edit (n={len(ed)}): '
          f'la {sum(r["la_ok_ci"] for r in ed)}/{len(ed)}   lap {sum(r["lap_ok_ci"] for r in ed)}/{len(ed)}')

    art_la = [r for r in rows if (r['la_del0'] or 0) > 0.5]
    art_lap = [r for r in rows if (r['lap_del0'] or 0) > 0.5]
    print(f'\ncriterion (a) unit-0 del_before > 0.5 (leading-deletion artifact): '
          f'la {len(art_la)}  lap {len(art_lap)}')
    for r in art_lap:
        print(f'  lap still artifacted: {r["item_id"]} del0 {r["lap_del0"]:.2f} '
              f'(la {r["la_del0"]:.2f}) map {r["lap_map"][:50]!r}')
    any_la = [r for r in rows if (r['la_delany'] or 0) > 0.5]
    any_lap = [r for r in rows if (r['lap_delany'] or 0) > 0.5]
    print(f'ANY-unit del_before > 0.5 (the sec-3.4 affected signature): la {len(any_la)}  lap {len(any_lap)}')

    dz = sorted(r['dlogZ'] for r in rows)
    mean = sum(dz) / n
    print(f'\ncriterion (d) logZ shift (lap - la): mean {mean:+.2f}  median {dz[n // 2]:+.2f}  '
          f'up {sum(1 for d in dz if d > 0.5)}  down {sum(1 for d in dz if d < -0.5)}  ~flat {sum(1 for d in dz if abs(d) <= 0.5)}')
    movers = sorted(rows, key=lambda r: -abs(r['dlogZ']))[:8]
    for r in movers:
        print(f'  {r["item_id"]:>9} dlogZ {r["dlogZ"]:+7.2f}  del0 {(r["la_del0"] or 0):.2f}->{(r["lap_del0"] or 0):.2f}  '
              f'ok {r["la_ok"]}->{r["lap_ok"]}')

    sp = [(r['la_spread'], r['lap_spread']) for r in rows
          if r['la_spread'] is not None and r['lap_spread'] is not None]
    if sp:
        print(f'\n4-seed logZ spread: mean la {sum(a for a, _ in sp) / len(sp):.2f}  '
              f'lap {sum(b for _, b in sp) / len(sp):.2f}  '
              f'lap>la on {sum(1 for a, b in sp if b > a)}/{len(sp)}')

    changed = [r for r in rows if r['la_map'] != r['lap_map']]
    gained = [r for r in changed if r['lap_ok'] and not r['la_ok']]
    lost = [r for r in changed if r['la_ok'] and not r['lap_ok']]
    print(f'\nMAP changed on {len(changed)} items: newly-correct {len(gained)}, newly-wrong {len(lost)}')
    fam = collections.Counter()
    for r in changed:
        fam[r['family']] += 1
    print('  changes by family:', dict(sorted(fam.items())))
    for tag, group in (('GAINED', gained), ('LOST', lost)):
        for r in group:
            print(f'  {tag} {r["item_id"]:>9} [{r["expected"]}] la {r["la_map"][:44]!r} -> lap {r["lap_map"][:44]!r}')

    out = 'planning/calibration_lap_vs_la.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
