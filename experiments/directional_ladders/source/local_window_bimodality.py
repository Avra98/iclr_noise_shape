#!/usr/bin/env python3
"""Short-window occupancy in genuinely fixed parameter coordinates.

Inputs are branches/*/{meta.json,trace.npz}. trace.npz supplies step[T] and
coordinates[T,K] (projection is an alias). axis_names and eta can be in the trace
or metadata. Optional scalar_s and fixed_rayleigh describe realized minibatches,
not independent conditional probes. Fresh conditional draws belong in
fresh_probes.npz: probe_step[N], fixed_gain[N,K], scalar_gain[N].

Run --root CAMPAIGN --out CAMPAIGN/analysis. After discovery is complete,
--freeze-selection writes a locked, discovery-only axis/time-window choice for
subsequent exact-window confirmation. All other scans remain exploratory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

WINDOWS = (64, 128, 256, 512, 1024)
BANDWIDTHS = (.7, 1., 1.4)
METHOD = {
    'version': 1,
    'window_lengths_observations': list(WINDOWS),
    'overlap': '50%; every eligible window and axis enters the denominator',
    'coordinate': 'One fixed parameter reference and direction per branch. Raw coordinates are primary. No tail clipping, moving-axis histograms, or pooling of distinct branches.',
    'density': 'Gaussian-smoothed full-support histogram (2048 cells), Scott bandwidth SD*n^(-1/5) times 0.7, 1.0 and 1.4; four bandwidths of padding.',
    'two_lobes': 'Exactly two prominent density maxima at each of the three bandwidths; each prominence >=15% maximum density; separation >=0.8 SD; valley <=80% smaller peak; each side >=10% empirical occupancy. A descriptive screen, not a test.',
    'drift_control': 'Four contiguous block means span >0.8 SD or first/second half Wasserstein distance >0.5 SD flags drift. Linear detrending is shown separately and never supplies the primary result.',
    'switching_control': 'Report all crossings of the central valley and core-to-core switches (cores halfway from valley to each peak). At least four core switches and both cores visited in both halves is recurring switching.',
    'parity_control': 'Lag-1/lag-2 correlation, even-odd mean gap/SD, separate parity densities. Lag1 < -0.5 and parity gap >1.2 SD flags an alternating two-cycle pattern, which is distinct from random switching.',
    'moment': 'H=1/mean(1/abs(a)), Lambda=mean(log(abs(a))), a=1-eta*s. Exact zeros retained. Realized trajectory windows and fresh checkpoint-conditional probes are reported separately. Inverse-moment existence is not established.',
    'confirmation': 'A choice frozen from discovery uses the same axis name, relative start index and width in each confirmation stream. Whole confirmation scans are still exploratory; neither scan counts nor selected examples provide iid p-values.',
    'limits': 'Overlap and serial dependence prevent treating windows as independent replications. Finite-window two lobes do not establish stationarity, an invariant bimodal law, a full network Lyapunov exponent, or harmonic-threshold causality.',
}


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else ('+inf' if value > 0 else '-inf')
    return value


def dump(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(clean(obj), indent=2, allow_nan=False))
    temporary.replace(path)


def density(x, factor=1.):
    x = np.asarray(x, float)
    sd = float(x.std()) if len(x) else 0.
    if len(x) < 2 or sd == 0: return np.array([x[0] if len(x) else 0.]), np.array([1.]), 0.
    h = sd * len(x) ** (-.2) * factor
    edges = np.linspace(x.min() - 4*h, x.max() + 4*h, 2049)
    dx = edges[1] - edges[0]
    counts, _ = np.histogram(x, edges)
    y = gaussian_filter1d(counts.astype(float) / (len(x)*dx), max(h/dx, .5), mode='constant')
    return (edges[:-1] + edges[1:]) / 2, y, h


def shape(x, factors=BANDWIDTHS):
    x = np.asarray(x, float)
    result = {'n': len(x), 'passes': False, 'by_bandwidth': {}}
    if len(x) < 24 or not np.isfinite(x).all(): return result
    sd = float(x.std()); result['sd'] = sd
    if sd <= np.finfo(float).eps * max(1., float(np.max(np.abs(x)))) * 32: return result
    for factor in factors:
        grid, y, h = density(x, factor)
        peaks, props = find_peaks(y, prominence=.15*y.max(), distance=max(1, int(.8*sd/(grid[1]-grid[0]))))
        r = {'peak_count': len(peaks), 'h': h, 'passes': False, 'peaks': grid[peaks].tolist()}
        if len(peaks) == 2:
            left, right = peaks
            valley = int(left + np.argmin(y[left:right+1])); cut = float(grid[valley])
            mass = float(np.mean(x <= cut)); ratio = float(y[valley]/min(y[left], y[right]))
            separation = float((grid[right] - grid[left])/sd)
            resolution_ok = bool(h >= 2*(grid[1]-grid[0]))
            r.update(valley=cut, left_mass=mass, valley_ratio=ratio,
                     separation_sd=separation, prominence_fraction=(props['prominences']/y.max()).tolist(),
                     resolution_ok=resolution_ok,
                     passes=bool(separation >= .8 and ratio <= .8 and .1 <= mass <= .9 and resolution_ok))
        result['by_bandwidth'][str(factor)] = r
    result['passes'] = all(r['passes'] for r in result['by_bandwidth'].values())
    result['bandwidth_pass_count'] = sum(r['passes'] for r in result['by_bandwidth'].values())
    return result


def correlation(x, lag):
    if len(x) <= lag+2 or x[:-lag].std() == 0 or x[lag:].std() == 0: return float('nan')
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def detrend(x):
    t = np.linspace(-1., 1., len(x))
    fit = np.polyval(np.polyfit(t, x, 1), t)
    return x-fit, fit


def controls(x, raw):
    sd = float(x.std())
    if sd == 0: return {'drift_flag': False, 'recurring_switching': False, 'alternation_flag': False}
    block_means = np.array([part.mean() for part in np.array_split(x, 4)])
    drift = float(np.ptp(block_means)/sd)
    split = float(wasserstein_distance(x[:len(x)//2], x[len(x)//2:])/sd)
    lag1 = correlation(x, 1); parity = float(abs(x[::2].mean()-x[1::2].mean())/sd)
    result = {'lag1': lag1, 'lag2': correlation(x, 2), 'parity_gap_sd': parity,
              'block_mean_range_sd': drift, 'split_wasserstein_sd': split,
              'drift_flag': bool(drift > .8 or split > .5),
              'alternation_flag': bool(lag1 < -.5 and parity > 1.2),
              'recurring_switching': False, 'valley_crossings': 0, 'core_switches': 0}
    mode = raw['by_bandwidth'].get('1.0', {})
    if mode.get('passes'):
        cut = mode['valley']; left, right = mode['peaks']
        sides = x > cut
        cores = np.where(x < (left+cut)/2, -1, np.where(x > (right+cut)/2, 1, 0))
        nonzero = cores[cores != 0]
        switches = int(np.sum(nonzero[1:] != nonzero[:-1]))
        both_halves = all(len(np.unique(part[part != 0])) == 2 for part in np.array_split(cores, 2))
        result.update(valley_crossings=int(np.sum(sides[1:] != sides[:-1])), core_switches=switches,
                      both_cores_in_both_halves=both_halves,
                      recurring_switching=bool(switches >= 4 and both_halves))
    return result


def moments(a):
    a = np.asarray(a, float); r = {'n': len(a), 'valid': bool(len(a) and np.isfinite(a).all())}
    if not r['valid']: return r
    aa = np.abs(a)
    with np.errstate(divide='ignore'):
        logs = np.log(aa); log_inv_sum = logsumexp(-logs)
    r.update(Lambda=float(logs.mean()), H=float(np.exp(np.log(len(a))-log_inv_sum)),
             expanding_fraction=float(np.mean(aa > 1)), exact_zero_count=int(np.sum(aa == 0)),
             min_abs_a=float(aa.min()))
    if r['exact_zero_count']:
        r['largest_inverse_weight_share'] = None
    else:
        r['largest_inverse_weight_share'] = float(np.exp(np.max(-logs)-log_inv_sum))
    return r


def load_branch(folder):
    meta = json.loads((folder/'meta.json').read_text())
    with np.load(folder/'trace.npz', allow_pickle=False) as z: trace = {k: z[k] for k in z.files}
    x = np.asarray(trace.get('coordinates', trace.get('projection')), float)
    steps = np.asarray(trace['step'])
    if x.ndim != 2 or len(x) != len(steps): raise ValueError('Expected coordinates[T,K] and step[T]')
    if len(steps) > 1 and np.any(np.diff(steps) <= 0): raise ValueError('step must increase strictly')
    names = trace.get('axis_names', meta.get('axis_names'))
    if names is None: names = [f'fixed_axis_{i+1}' for i in range(x.shape[1])]
    names = [str(v) for v in names]
    if len(names) != x.shape[1]: raise ValueError('axis_names must match coordinate columns')
    eta = float(np.asarray(trace.get('eta', meta.get('eta', meta.get('learning_rate', np.nan)))))
    trace.update(coordinates=x, step=steps, axis_names=names, eta=eta)
    fresh = {}
    for name in ('fresh_probes.npz', 'same_axis_gains.npz', 'probes.npz'):
        if (folder/name).exists():
            with np.load(folder/name, allow_pickle=False) as z: fresh = {k: z[k] for k in z.files}
            break
    if 'fixed_gain' not in fresh:
        if 'a' in fresh: fresh['fixed_gain'] = fresh['a']
        elif 'fixed_rayleigh' in fresh: fresh['fixed_gain'] = 1-eta*fresh['fixed_rayleigh']
    if 'scalar_gain' not in fresh and 'scalar_s' in fresh: fresh['scalar_gain'] = 1-eta*fresh['scalar_s']
    return meta, trace, fresh


def moment_window(trace, lo, hi, axis):
    result = {}
    eta = trace['eta']; n = len(trace['step'])
    if 'scalar_s' in trace and len(trace['scalar_s']) == n:
        result['gradient_direction_realized'] = moments(1-eta*np.asarray(trace['scalar_s'])[lo:hi])
    if 'fixed_rayleigh' in trace and np.asarray(trace['fixed_rayleigh']).shape == trace['coordinates'].shape:
        result['same_fixed_axis_realized'] = moments(1-eta*np.asarray(trace['fixed_rayleigh'])[lo:hi, axis])
    if 'moving_log_gain' in trace and len(trace['moving_log_gain']) == n:
        logs = np.asarray(trace['moving_log_gain'])[lo:hi]
        result['moving_vector_mean_log_norm_gain'] = np.mean(logs, axis=0).tolist()
    if 'fixed_norm_log_gain' in trace and np.asarray(trace['fixed_norm_log_gain']).shape == trace['coordinates'].shape:
        result['same_fixed_axis_reset_mean_log_norm_gain'] = float(np.mean(trace['fixed_norm_log_gain'][lo:hi, axis]))
    if 'fixed_orthogonal_gain_sq' in trace and np.asarray(trace['fixed_orthogonal_gain_sq']).shape == trace['coordinates'].shape:
        result['same_fixed_axis_mean_orthogonal_gain_sq'] = float(np.mean(trace['fixed_orthogonal_gain_sq'][lo:hi, axis]))
    return result


def evaluate_window(trace, axis, lo, width):
    hi = lo+width; x = trace['coordinates'][lo:hi, axis]; steps = trace['step']
    row = {'axis_index': axis, 'axis_name': trace['axis_names'][axis], 'width': width,
           'start_index': lo, 'stop_index_exclusive': hi, 'start_step': float(steps[lo]),
           'end_step': float(steps[hi-1]), 'finite': bool(np.isfinite(x).all()),
           'raw_pass': False, 'detrended_pass': False, 'category': 'invalid',
           'raw': {}, 'detrended': {}, 'controls': {}, 'moments': moment_window(trace, lo, hi, axis)}
    if not row['finite']: return row
    raw = shape(x); residual, _ = detrend(x); residual_shape = shape(residual)
    control = controls(x, raw)
    control['even_raw_shape'] = shape(x[::2], factors=(1.,))
    control['odd_raw_shape'] = shape(x[1::2], factors=(1.,))
    category = 'no_robust_raw_two_lobes'
    if raw['passes']:
        category = ('alternating_two_lobes' if control['alternation_flag'] else
                    'drifting_two_lobes' if control['drift_flag'] else
                    'recurring_two_lobes' if control['recurring_switching'] else
                    'raw_two_lobes_without_recurring_switching')
    elif residual_shape['passes']: category = 'detrended_only_two_lobes'
    row.update(raw_pass=raw['passes'], detrended_pass=residual_shape['passes'],
               raw=raw, detrended=residual_shape, controls=control, category=category)
    return row


def rank(row):
    raw = row['raw']; control = row['controls']
    central = raw.get('by_bandwidth', {}).get('1.0', {})
    return (int(row['raw_pass']), int(control.get('recurring_switching', False)),
            int(not control.get('drift_flag', True)), int(not control.get('alternation_flag', True)),
            raw.get('bandwidth_pass_count', 0), -central.get('valley_ratio', 1.), row['width'])


def select_examples(rows, count=4):
    selected = []
    for row in sorted(rows, key=rank, reverse=True):
        if not row['finite']: continue
        if any(row['axis_index'] == r['axis_index'] and
               max(row['start_index'], r['start_index']) < min(row['stop_index_exclusive'], r['stop_index_exclusive'])
               for r in selected): continue
        selected.append(row)
        if len(selected) == count: break
    return selected


def savefig(fig, path):
    fig.tight_layout(rect=(0, 0, 1, .955)); fig.savefig(path, dpi=150); plt.close(fig)


def figures(folder, meta, trace, rows, selected, fresh_summary):
    x = trace['coordinates']; steps = trace['step']; names = trace['axis_names']; k = x.shape[1]
    title = f"{meta.get('architecture', 'CNN')} | {meta.get('role', meta.get('split', 'exploratory'))} | finite-window fixed-coordinate occupancy"
    colors = ListedColormap(['#eeeeee', '#efb851', '#7049a8'])
    fig, axs = plt.subplots(k, 2, figsize=(15, 2.8*k), squeeze=False)
    for j, name in enumerate(names):
        axs[j, 0].plot(steps, x[:, j], lw=.5)
        for r in selected:
            if r['axis_index'] == j: axs[j, 0].axvspan(r['start_step'], r['end_step'], color='C1', alpha=.17)
        axs[j, 0].set(title=name, xlabel='continuation optimizer step', ylabel='fixed coordinate')
        for wi, width in enumerate(WINDOWS):
            subset = [r for r in rows if r['axis_index'] == j and r['width'] == width]
            for r in subset:
                value = 2 if r['raw_pass'] else 1 if r['detrended_pass'] else 0
                center = (r['start_step']+r['end_step'])/2
                axs[j, 1].scatter(center, wi, color=colors(value), marker='s', s=17, edgecolors='none')
        axs[j, 1].set(yticks=range(len(WINDOWS)), yticklabels=WINDOWS, xlabel='window center step', ylabel='window observations', title='Purple: raw passes | amber: detrended only | gray: neither')
    fig.suptitle(title, fontsize=11); savefig(fig, folder/'fig1_trace_and_all_windows.png')

    if selected:
        fig, axs = plt.subplots(len(selected), 4, figsize=(18, 3.3*len(selected)), squeeze=False)
        for i, row in enumerate(selected):
            lo, hi, j = row['start_index'], row['stop_index_exclusive'], row['axis_index']
            v = x[lo:hi, j]; residual, fit = detrend(v); ss = steps[lo:hi]; c = row['controls']
            axs[i, 0].plot(ss, v, lw=.7); axs[i, 0].plot(ss, fit, '--', c='C3', lw=.9)
            axs[i, 0].set(title=f"{names[j]} | n={len(v)}\n{row['category'].replace('_', ' ')}", xlabel='step', ylabel='raw fixed coordinate')
            for factor in BANDWIDTHS:
                grid, y, _ = density(v, factor)
                axs[i, 1].plot(grid, y, label=f'Scott ×{factor}', lw=1.)
                grid, y, _ = density(residual, factor)
                axs[i, 3].plot(grid, y, label=f'×{factor}', lw=.8)
            axs[i, 1].hist(v, bins=min(32, int(np.sqrt(len(v))*2)), density=True, alpha=.15, color='k')
            axs[i, 1].set(title='Raw occupancy: primary', xlabel='raw fixed coordinate', ylabel='density'); axs[i, 1].legend(fontsize=7)
            for vv, label in ((v[::2], 'even observations'), (v[1::2], 'odd observations')):
                grid, y, _ = density(vv, 1.); axs[i, 2].plot(grid, y, label=label)
            axs[i, 2].set(title=f"lag1={c.get('lag1', np.nan):.2f}; core switches={c.get('core_switches', 0)}\nparity gap/SD={c.get('parity_gap_sd', np.nan):.2f}", xlabel='raw fixed coordinate', ylabel='parity density'); axs[i, 2].legend(fontsize=7)
            axs[i, 3].set(title=f"Detrended control only; passes={row['detrended_pass']}\ndrift flag={c.get('drift_flag', False)}", xlabel='linear-fit residual', ylabel='density')
        fig.suptitle(title+'\nExamples selected by the declared screen; no significance claim', fontsize=11)
        savefig(fig, folder/'fig2_selected_short_windows.png')

    fig, axs = plt.subplots(k, 2, figsize=(14, 2.6*k), squeeze=False)
    for j, name in enumerate(names):
        axisrows = [r for r in rows if r['axis_index'] == j and r['width'] == 256]
        for col, key in enumerate(('gradient_direction_realized', 'same_fixed_axis_realized')):
            ax = axs[j, col]; present = [r for r in axisrows if r['moments'].get(key, {}).get('valid')]
            for r in present:
                m = r['moments'][key]
                if np.isfinite(m['Lambda']): ax.scatter(m['Lambda'], m['H'], c='C3' if r['raw_pass'] else 'C0', s=18, alpha=.65)
            ax.axhline(1, ls='--', c='k', lw=.6); ax.axvline(0, ls='--', c='k', lw=.6)
            ax.set(title=f"{name}\n{key.replace('_', ' ')}", xlabel='window mean log scalar gain', ylabel='window harmonic gain')
            if not present: ax.text(.05, .6, 'Dense realized gains unavailable;\nsee checkpoint-conditional probes in JSON.', transform=ax.transAxes)
    fig.suptitle('256-observation windows | red: raw two-lobe screen | blue: other\nOverlap is retained; realized path moments are not frozen-checkpoint expectations', fontsize=11)
    savefig(fig, folder/'fig3_window_moments.png')


def analyze_branch(source, out, render=True):
    meta, trace, fresh = load_branch(source)
    steps = trace['step']; rows = []
    for axis in range(trace['coordinates'].shape[1]):
        for width in WINDOWS:
            for lo in range(0, len(steps)-width+1, width//2): rows.append(evaluate_window(trace, axis, lo, width))
    selected = select_examples(rows)
    fresh_summary = {}
    if fresh and 'probe_step' in fresh:
        probe_steps = np.asarray(fresh['probe_step'])
        for step in np.unique(probe_steps):
            use = probe_steps == step; one = {}
            if 'scalar_gain' in fresh: one['gradient_direction_conditional'] = moments(np.asarray(fresh['scalar_gain'])[use])
            if 'fixed_gain' in fresh:
                a = np.asarray(fresh['fixed_gain'])
                one['fixed_axis_conditional'] = {name: moments(a[use, j]) for j, name in enumerate(trace['axis_names'])}
            if 'fixed_norm_log_gain' in fresh:
                one['fixed_axis_reset_conditional_mean_log_norm_gain'] = {
                    name: float(np.mean(np.asarray(fresh['fixed_norm_log_gain'])[use, j]))
                    for j, name in enumerate(trace['axis_names'])}
            fresh_summary[str(float(step))] = one
    denominators = []
    for axis, name in enumerate(trace['axis_names']):
        for width in WINDOWS:
            rr = [r for r in rows if r['axis_index'] == axis and r['width'] == width]
            denominators.append({'axis_name': name, 'width': width, 'scanned_windows': len(rr),
                                 'finite_windows': sum(r['finite'] for r in rr),
                                 'raw_pass_windows': sum(r['raw_pass'] for r in rr),
                                 'detrended_only_windows': sum(r['detrended_pass'] and not r['raw_pass'] for r in rr),
                                 'recurring_nondrift_nonalternating_windows': sum(r['category'] == 'recurring_two_lobes' for r in rr),
                                 'alternating_windows': sum(r['category'] == 'alternating_two_lobes' for r in rr)})
    result = {'source': str(source.resolve()), 'meta': meta, 'observations': len(steps),
              'branch_complete': str(meta.get('status', '')).lower() in ('complete', 'completed'),
              'step_stride': sorted(set(np.diff(steps).tolist())) if len(steps)>1 else [],
              'method': METHOD, 'windows': rows, 'denominators': denominators,
              'selected_examples': selected, 'fresh_checkpoint_moments': fresh_summary,
              'trace_sha256': hashlib.sha256((source/'trace.npz').read_bytes()).hexdigest(),
              'analysis_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.mkdir(parents=True, exist_ok=True); dump(out/'analysis.json', result)
    with (out/'all_windows.csv').open('w', newline='') as handle:
        keys = ('axis_name', 'width', 'start_index', 'start_step', 'end_step', 'finite', 'raw_pass', 'detrended_pass', 'category')
        control_keys = ('lag1', 'lag2', 'parity_gap_sd', 'drift_flag', 'core_switches', 'recurring_switching')
        writer = csv.DictWriter(handle, fieldnames=keys+control_keys); writer.writeheader()
        for row in rows: writer.writerow(clean({**{key: row[key] for key in keys}, **{key: row['controls'].get(key) for key in control_keys}}))
    if render: figures(out, meta, trace, rows, selected, fresh_summary)
    return result


def role(meta): return str(meta.get('role', meta.get('split', 'exploratory'))).lower()


def confirmation(results, out, freeze):
    selection_file = out/'discovery_selection.json'
    discovery = [r for r in results if role(r['meta']) == 'discovery']
    if freeze and not selection_file.exists():
        if any(not r['branch_complete'] for r in discovery):
            raise ValueError('Cannot freeze selection before every discovered discovery branch is completed')
        choices = [(r, w) for r in discovery for w in r['windows'] if w['finite']]
        if not choices: raise ValueError('Cannot freeze selection: no discovery windows')
        r, w = max(choices, key=lambda pair: rank(pair[1]))
        dump(selection_file, {'selection_source': 'discovery_only', 'source': r['source'],
                             'trace_sha256': r['trace_sha256'], 'axis_name': w['axis_name'],
                             'checkpoint_sha256': r['meta'].get('checkpoint_sha256'),
                             'frame': r['meta'].get('frame'),
                             'analysis_code_sha256': r['analysis_code_sha256'],
                             'start_index': w['start_index'], 'width': w['width'],
                             'discovery_category': w['category'], 'raw_pass': w['raw_pass'],
                             'note': 'Locked relative window and axis; continuation streams share a parent checkpoint. Full scans do not constitute confirmation.'})
    if not selection_file.exists(): return {'locked': False, 'note': 'No frozen discovery-only selection. All displayed scans are exploratory.'}
    selection = json.loads(selection_file.read_text()); checked = []
    fig, axs = plt.subplots(max(1, sum(role(r['meta']) == 'confirmation' for r in results)), 2, figsize=(12, max(3.3, 3.3*sum(role(r['meta']) == 'confirmation' for r in results))), squeeze=False)
    for result in results:
        if role(result['meta']) != 'confirmation': continue
        meta, trace, _ = load_branch(Path(result['source'])); lo = selection['start_index']; width = selection['width']
        mismatched = [key for key in ('checkpoint_sha256', 'frame')
                      if selection.get(key) is not None and selection[key] != meta.get(key)]
        if mismatched:
            checked.append({'source': result['source'], 'available': False,
                            'reason': 'Fixed-frame provenance mismatch: '+', '.join(mismatched)})
            continue
        if selection['axis_name'] not in trace['axis_names'] or len(trace['step']) < lo+width:
            checked.append({'source': result['source'], 'available': False}); continue
        axis = trace['axis_names'].index(selection['axis_name']); row = evaluate_window(trace, axis, lo, width)
        i = len(checked); checked.append({'source': result['source'], 'available': True, 'window': row})
        x = trace['coordinates'][lo:lo+width, axis]; steps = trace['step'][lo:lo+width]
        axs[i, 0].plot(steps, x, lw=.6); axs[i, 0].set(title=f"Confirmation stream {i+1}: {row['category'].replace('_', ' ')}", xlabel='step', ylabel='raw fixed coordinate')
        for f in BANDWIDTHS:
            grid, y, _ = density(x, f); axs[i, 1].plot(grid, y, label=f'Scott ×{f}')
        axs[i, 1].set(xlabel='raw fixed coordinate', ylabel='density'); axs[i, 1].legend(fontsize=8)
    if checked:
        fig.suptitle(f"Locked discovery choice: {selection['axis_name']}, start index {lo}, {width} observations\nExact-window confirmation; no alternative-window selection", fontsize=11)
        savefig(fig, out/'fig4_locked_window_confirmation.png')
    else: plt.close(fig)
    output = {'locked': True, 'selection': selection, 'confirmation_streams': checked}
    dump(out/'confirmation.json', output); return output


def aggregate(results, errors, out, confirm):
    denominator = sum(len(r['windows']) for r in results)
    raw = sum(sum(w['raw_pass'] for w in r['windows']) for r in results)
    recurring = sum(sum(w['category'] == 'recurring_two_lobes' for w in r['windows']) for r in results)
    summary = {'branches': len(results), 'scanned_axis_windows': denominator, 'raw_pass_windows': raw,
               'recurring_nondrift_nonalternating_windows': recurring, 'errors': errors,
               'denominators_by_branch': {Path(r['source']).name: r['denominators'] for r in results},
               'confirmation': confirm, 'method': METHOD}
    dump(out/'summary.json', summary)
    text = ['# Short-window fixed-direction analysis', '',
            f'{len(results)} branches; {denominator} axis/windows scanned, including overlapping windows. '
            f'{raw} pass the raw three-bandwidth two-lobe screen; {recurring} also show recurring core switching without the defined drift or alternation flags.', '',
            'These are descriptive counts. A passing scanned window is not a significant discovery, an independent replication, or evidence of a stationary bimodal distribution.', '']
    text += [f'**{key}:** {value}\n' for key, value in METHOD.items()]
    (out/'README.md').write_text('\n'.join(text))
    cards = []
    for r in results:
        folder = Path(r['source']).name; name = html.escape(folder)
        figs = ''.join(f'<a href="{folder}/{f.name}"><img loading="lazy" src="{folder}/{f.name}"></a>' for f in sorted((out/folder).glob('fig*.png')))
        cards.append(f'<section><h2>{name} | {html.escape(role(r["meta"]))}</h2><a href="{folder}/analysis.json">All windows, controls and moments</a> · <a href="{folder}/all_windows.csv">Window CSV</a>{figs}</section>')
    conf_html = '<a href="fig4_locked_window_confirmation.png"><img src="fig4_locked_window_confirmation.png"></a>' if (out/'fig4_locked_window_confirmation.png').exists() else ''
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Short-window fixed-direction occupancy</title><style>body{font:16px system-ui;max-width:1500px;margin:2rem auto}img{max-width:100%;height:auto}section{border-top:1px solid #aaa;margin-top:2rem}p{max-width:1100px}</style><h1>Short-window fixed-direction occupancy</h1>'
        f'<p>{denominator} scanned axis/windows; {raw} raw two-lobe screens passed. Overlap and serial dependence are retained; counts are descriptive.</p>'
        '<p>Raw fixed-coordinate occupancy is primary. Detrending, recurring switching, parity and drift are separate controls. Moment values alone do not establish bimodality.</p><p><a href="README.md">Methods</a> · <a href="summary.json">All scan denominators</a></p>'+conf_html+''.join(cards))


def self_test():
    rng = np.random.default_rng(1621); n = 1024; t = np.arange(n)
    fixtures = {'normal': rng.normal(size=n),
                'mixture': rng.choice([-2., 2.], size=n)+rng.normal(scale=.22, size=n),
                'alternate': 2*(2*(t%2)-1)+rng.normal(scale=.15, size=n),
                'two_plateaus': np.where(t<n//2, -2., 2.)+rng.normal(scale=.15, size=n),
                'linear_drift': np.linspace(-4, 4, n)+rng.normal(scale=.2, size=n),
                'constant': np.zeros(n)}
    result = {}
    for name, x in fixtures.items():
        trace = {'coordinates': x[:, None], 'step': t, 'axis_names': ['fixed'], 'eta': .1}
        result[name] = evaluate_window(trace, 0, 0, n)
    assert not result['normal']['raw_pass']
    assert result['mixture']['category'] == 'recurring_two_lobes'
    assert result['alternate']['category'] == 'alternating_two_lobes'
    assert result['two_plateaus']['category'] == 'drifting_two_lobes'
    assert not result['linear_drift']['raw_pass']
    assert not result['constant']['raw_pass']
    assert moments(np.array([0., 2.]))['H'] == 0.
    assert moments(np.array([0., 2.]))['Lambda'] == -float('inf')
    assert moments(np.array([2., 2.]))['H'] == 2.
    assert not moments(np.array([np.nan, 2.]))['valid']
    print(json.dumps({name: r['category'] for name, r in result.items()}, indent=2))
    print('Self-test passed: separated mixtures, drift, alternation, normal occupancy, exact zeros and invalid gains.')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--no-render', action='store_true')
    p.add_argument('--freeze-selection', action='store_true')
    p.add_argument('--branch', action='append', default=[], help='Analyze only these branch IDs; repeat as needed')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test: self_test()
    if not args.root:
        if args.self_test: return
        p.error('--root required')
    out = args.out or args.root/'analysis'; out.mkdir(parents=True, exist_ok=True)
    results, errors = [], []
    for source in sorted(args.root.glob('branches/*/trace.npz')):
        if args.branch and source.parent.name not in args.branch: continue
        if not args.branch and source.parent.name.startswith('smoke'): continue
        if not (source.parent/'meta.json').exists(): continue
        try:
            r = analyze_branch(source.parent, out/source.parent.name, not args.no_render); results.append(r)
            print(f"{source.parent.name}: {len(r['windows'])} axis/windows; {sum(w['raw_pass'] for w in r['windows'])} raw screen passes", flush=True)
        except Exception as exc:
            errors.append({'source': str(source), 'error': repr(exc)}); print(f'ERROR {source}: {exc}', flush=True)
    confirm = confirmation(results, out, args.freeze_selection)
    aggregate(results, errors, out, confirm)
    print(out/'index.html')
    if errors: raise SystemExit(1)


if __name__ == '__main__': main()
