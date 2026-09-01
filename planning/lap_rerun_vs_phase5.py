#!/usr/bin/env python
"""Compare the Phase-5 main_off RE-RUN under LA_PROPOSAL=1 (slug ...__la__lap__nseed4) against the
original Phase-5 main_off outputs (slug ...__la__nseed4), from the collected tables in
experiments/outputs/ (planning/OFF_ARM_INFERENCE_FIX.md sec 6 decision 4).

Two jobs: (1) re-run the Phase-5 verification gates on the NEW outputs (ok counts, finite
surprisals, p_copy+p_sub+p_ins == 1, sum(S_k)+S_end == -logZ on merged records); (2) report what
moved, per dataset and overall: the posited-deletion signatures (unit-0 and any-unit del_before >
0.5), edit rate, p_literal == 0 count, 4-seed logZ spread, logZ shift on joined items, MAP changes,
and the gibson2013 grammaticality contrast. Writes planning/phase5_lap_vs_la_summary.csv.

Usage: python planning/lap_rerun_vs_phase5.py [OLD_SLUG NEW_SLUG]   (defaults below; a dataset whose
new-slug tables are not collected yet is reported as pending and skipped).
"""
import os, sys, gzip
import numpy as np
import pandas as pd

OUT = 'experiments/outputs'
OLD = 'lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__la__nseed4'
NEW = 'lm-pythia-70m__ch-align__rej-off__P64__b2__d2__lb6__s0__la__lap__nseed4'
DATASETS = ['moses', 'tabor2004', 'huang2024', 'gibson2013', 'clark2026', 'qian2023', 'ryskin2021', 'chen2023']
REAL = [d for d in DATASETS if d != 'moses']


def load(slug, ds):
    d = os.path.join(OUT, slug, ds)
    if not os.path.exists(os.path.join(d, 'sentences.csv.gz')):
        return None, None
    s = pd.read_csv(os.path.join(d, 'sentences.csv.gz'), keep_default_na=False)
    w = pd.read_csv(os.path.join(d, 'words.csv.gz'))
    return s, w


def num(x):
    return pd.to_numeric(x, errors='coerce')


def gates(s, w, label):
    ok = s[s.status == 'ok']
    n_err = int((s.status == 'error').sum()); n_miss = int((s.status == 'missing').sum())
    fin = int(w.surprisal_nc.notna().sum()) == len(w) and int(w.surprisal_lm.notna().sum()) == len(w)
    psum = (w.p_copy + w.p_sub + w.p_ins).abs()
    pdev = float((psum - 1).abs().max()) if len(w) else float('nan')
    m = w[w.seed.astype(str) == 'merged']
    S = m.groupby('sentence_id').agg(S=('surprisal_nc', 'sum'), Send=('surprisal_end_nc', 'first'))
    lz = ok.set_index('sentence_id')['logZ'].pipe(num)
    j = S.join(lz, how='inner')
    ident = float((j.S + j.Send + j.logZ).abs().max()) if len(j) else float('nan')
    return dict(label=label, ok=len(ok), error=n_err, missing=n_miss, word_rows=len(w),
                all_finite=fin, psum_maxdev=pdev, identity_maxdev=ident)


def arm_metrics(s, w):
    # one row per model input: qian2023 maps several stimulus rows onto one sentence_id (480 stim
    # rows, 472 inputs), and the join below is keyed by sentence_id
    ok = s[s.status == 'ok'].drop_duplicates('sentence_id').copy()
    ok['edited'] = ok['map'].astype(str).str.strip() != ok['model_input'].astype(str).str.strip()
    ok['plit0'] = num(ok.p_literal) == 0
    m = w[w.seed.astype(str) == 'merged']
    d0 = m[m.unit_idx == 0].set_index('sentence_id')['del_before'].pipe(num)
    dany = m.groupby('sentence_id')['del_before'].max().pipe(num)
    ok = ok.set_index('sentence_id')
    ok['del0'] = d0.reindex(ok.index)
    ok['delany'] = dany.reindex(ok.index)
    return ok


def summarize(ok):
    return dict(n=len(ok), edited=int(ok.edited.sum()), edit_rate=ok.edited.mean(),
                plit0=int(ok.plit0.sum()), art0=int((ok.del0 > 0.5).sum()),
                artany=int((ok.delany > 0.5).sum()),
                spread_median=float(num(ok.logZ_spread).median()),
                spread_mean=float(num(ok.logZ_spread).mean()))


def main():
    old_slug, new_slug = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else (OLD, NEW)
    print(f'OLD: {old_slug}\nNEW: {new_slug}\n')
    rows, gate_rows, joined_all = [], [], []
    for ds in DATASETS:
        so, wo = load(old_slug, ds)
        sn, wn = load(new_slug, ds)
        if sn is None:
            print(f'{ds:>10}: NEW not collected yet -- pending'); continue
        if so is None:
            print(f'{ds:>10}: OLD missing'); continue
        gate_rows.append(dict(dataset=ds, **gates(sn, wn, 'new')))
        o, n = arm_metrics(so, wo), arm_metrics(sn, wn)
        j = o.join(n, how='inner', lsuffix='_old', rsuffix='_new')
        j['dlogZ'] = num(j.logZ_new) - num(j.logZ_old)
        j['map_changed'] = j.map_old.astype(str).str.strip() != j.map_new.astype(str).str.strip()
        j['dataset'] = ds
        joined_all.append(j)
        mo, mn = summarize(o), summarize(n)
        rows.append(dict(dataset=ds, n=mn['n'],
                         edited_old=mo['edited'], edited_new=mn['edited'],
                         plit0_old=mo['plit0'], plit0_new=mn['plit0'],
                         art0_old=mo['art0'], art0_new=mn['art0'],
                         artany_old=mo['artany'], artany_new=mn['artany'],
                         spread_med_old=mo['spread_median'], spread_med_new=mn['spread_median'],
                         dlogZ_mean=float(j.dlogZ.mean()), dlogZ_median=float(j.dlogZ.median()),
                         up=int((j.dlogZ > 0.5).sum()), down=int((j.dlogZ < -0.5).sum()),
                         map_changed=int(j.map_changed.sum()),
                         became_literal=int((j.map_changed & ~j.edited_new & j.edited_old).sum()),
                         became_edited=int((j.map_changed & j.edited_new & ~j.edited_old).sum())))

    if not rows:
        print('nothing to compare yet'); return
    print('== verification gates on the NEW outputs ==')
    g = pd.DataFrame(gate_rows)
    print(g.to_string(index=False))
    print(f'  TOTAL ok={g.ok.sum()} error={g.error.sum()} missing={g.missing.sum()} word_rows={g.word_rows.sum()} '
          f'all_finite={bool(g.all_finite.all())} psum_maxdev={g.psum_maxdev.max():.2e} '
          f'identity_maxdev={g.identity_maxdev.max():.2e}')

    print('\n== per dataset, old (la) -> new (la+lap) ==')
    t = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    print(t.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

    J = pd.concat(joined_all)
    R = J[J.dataset.isin(REAL)]
    n = len(R)
    print(f'\n== overall, the {len(set(R.dataset))} real datasets ({n} joined items) ==')
    for k, name in (('edited', 'edited MAPs'), ('plit0', 'p_literal == 0'),
                    ):
        a, b = int(R[f'{k}_old'].sum()), int(R[f'{k}_new'].sum())
        print(f'  {name:>28}: {a} ({100*a/n:.1f}%) -> {b} ({100*b/n:.1f}%)')
    a0, b0 = int((R.del0_old > 0.5).sum()), int((R.del0_new > 0.5).sum())
    aa, ba = int((R.delany_old > 0.5).sum()), int((R.delany_new > 0.5).sum())
    print(f'  {"unit-0 del_before > 0.5":>28}: {a0} ({100*a0/n:.1f}%) -> {b0} ({100*b0/n:.1f}%)')
    print(f'  {"any-unit del_before > 0.5":>28}: {aa} ({100*aa/n:.1f}%) -> {ba} ({100*ba/n:.1f}%)')
    print(f'  {"median 4-seed logZ spread":>28}: {num(R.logZ_spread_old).median():.2f} -> {num(R.logZ_spread_new).median():.2f}')
    print(f'  {"logZ shift new-old":>28}: mean {R.dlogZ.mean():+.2f} median {R.dlogZ.median():+.2f} '
          f'up {(R.dlogZ > 0.5).sum()} down {(R.dlogZ < -0.5).sum()} flat {((R.dlogZ.abs() <= 0.5)).sum()}')
    print(f'  {"MAP changed":>28}: {int(R.map_changed.sum())} '
          f'(became literal {int((R.map_changed & ~R.edited_new & R.edited_old).sum())}, '
          f'became edited {int((R.map_changed & R.edited_new & ~R.edited_old).sum())}, '
          f'edit -> different edit {int((R.map_changed & R.edited_new & R.edited_old).sum())})')
    # affected-class view (sec 3.4): items the OLD run flagged with any-unit del_before > 0.5
    aff = R[R.delany_old > 0.5]
    if len(aff):
        print(f'  {"old-affected items":>28}: {len(aff)}; of these now edited {int(aff.edited_new.sum())} '
              f'(was {int(aff.edited_old.sum())}), dlogZ mean {aff.dlogZ.mean():+.2f}, '
              f'still any-unit del>0.5: {int((aff.delany_new > 0.5).sum())}')
    g13 = R[R.dataset == 'gibson2013']
    if len(g13):   # directional sanity check: implausible (noisy) inputs should be edited more often
        for arm in ('old', 'new'):
            pl = g13[f'plausibility_{arm}'].astype(str)
            imp = g13[pl == 'implausible'][f'edited_{arm}'].mean()
            pla = g13[pl == 'plausible'][f'edited_{arm}'].mean()
            print(f'  gibson2013 edited rate {arm}: implausible {100*imp:.0f}% vs plausible {100*pla:.0f}%')

    out = 'planning/phase5_lap_vs_la_summary.csv'
    t.to_csv(out, index=False)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
