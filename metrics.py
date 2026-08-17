"""Metrics computed from the flat list of phase records.

Per-anomaly precision/recall/F1 and macro F1 come from the four outcome
classes. The binary scores (AUC, Brier, balanced accuracy, Cohen's kappa, MCC)
use y_true = 1 if coverage_true >= 0.5 and y_pred = 1 if coverage_obs >= 0.5,
i.e. the same threshold on both sides.
"""
import numpy as np

ANOMALY_NAMES = {
    1: 'water_stress', 2: 'pest_infestation', 3: 'fungal_disease',
    4: 'planting_failure', 5: 'atmospheric_contamination', 6: 'mixed_A1_A4',
}
POS_THRESHOLD = 0.5


def _per_anomaly(records):
    out = {}
    for aid in range(1, 7):
        rs = [r for r in records if r['anomaly_id'] == aid]
        if not rs:
            continue
        tp = sum(1 for r in rs if r['result'] == 'true_positive')
        pd = sum(1 for r in rs if r['result'] == 'partial_detection')
        fp = sum(1 for r in rs if r['result'] == 'false_positive')
        fn = sum(1 for r in rs if r['result'] == 'false_negative')
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[aid] = {'anomaly_name': ANOMALY_NAMES[aid], 'n_phases': len(rs),
                    'tp': tp, 'pd': pd, 'fp': fp, 'fn': fn,
                    'precision': round(prec, 4), 'recall': round(rec, 4),
                    'f1': round(f1, 4)}
    return out


def _auc(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float('nan'), n_pos, n_neg
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average the ranks of tied scores.
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    sum_pos = ranks[labels == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc), n_pos, n_neg


def _binary_labels(records, real_thr=POS_THRESHOLD, obs_thr=POS_THRESHOLD):
    y_true = np.array([1 if r['coverage_true'] >= real_thr else 0 for r in records])
    y_pred = np.array([1 if r['coverage_obs'] >= obs_thr else 0 for r in records])
    return y_true, y_pred


def _confusion_counts(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def balanced_accuracy(y_true, y_pred):
    tp, tn, fp, fn = _confusion_counts(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return 0.5 * (tpr + tnr)


def cohen_kappa(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return 0.0
    tp, tn, fp, fn = _confusion_counts(y_true, y_pred)
    po = (tp + tn) / n
    p_pred_pos = (tp + fp) / n
    p_true_pos = (tp + fn) / n
    pe = p_pred_pos * p_true_pos + (1 - p_pred_pos) * (1 - p_true_pos)
    return (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else 1.0


def mcc(y_true, y_pred):
    tp, tn, fp, fn = _confusion_counts(y_true, y_pred)
    num = tp * tn - fp * fn
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float(num / den) if den > 0 else 0.0


def brier(records):
    y_true, _ = _binary_labels(records)
    obs = np.array([r['coverage_obs'] for r in records])
    return float(np.mean((obs - y_true) ** 2))


def f1_macro(records):
    pa = _per_anomaly(records)
    if not pa:
        return 0.0
    return float(np.mean([v['f1'] for v in pa.values()]))


def _bootstrap_ci(records, fn, n_boot=500, seed=12345):
    """95% percentile bootstrap interval over resamples of the phase records."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    vals = []
    for _ in range(n_boot):
        samp = [records[i] for i in rng.choice(idx, size=len(idx), replace=True)]
        try:
            vals.append(fn(samp))
        except Exception:
            pass
    if not vals:
        return {'low': float('nan'), 'high': float('nan')}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {'low': round(float(lo), 4), 'high': round(float(hi), 4)}


def summarize(records, mission_aggregate):
    y_true, y_pred = _binary_labels(records)
    obs = [r['coverage_obs'] for r in records]
    conf = {k: sum(1 for r in records if r['result'] == k)
            for k in ('true_positive', 'partial_detection',
                      'false_positive', 'false_negative')}

    auc, n_pos, n_neg = _auc(obs, y_true)
    y_true_strict = np.array([1 if r['coverage_true'] >= 0.7 else 0 for r in records])
    auc7, _, _ = _auc(obs, y_true_strict)

    rewards = [r['reward'] for r in records]
    return {
        'n_records': len(records),
        'reward_mean': round(float(np.mean(rewards)), 4) if rewards else 0.0,
        'reward_total': round(float(np.sum(rewards)), 4) if rewards else 0.0,
        'confusion_matrix': conf,
        'per_anomaly': _per_anomaly(records),
        'f1_macro': round(f1_macro(records), 4),
        'auc_roc_binary': {'auc': round(auc, 4), 'n_pos': n_pos, 'n_neg': n_neg},
        'auc_strict_07': round(auc7, 4),
        'brier_score': round(brier(records), 4),
        'balanced_accuracy': round(balanced_accuracy(y_true, y_pred), 4),
        'cohen_kappa': round(cohen_kappa(y_true, y_pred), 4),
        'mcc': round(mcc(y_true, y_pred), 4),
        'mission_resolution': mission_aggregate,
        'ci_95': {
            'f1_macro': _bootstrap_ci(records, f1_macro),
            'balanced_accuracy': _bootstrap_ci(
                records, lambda rs: balanced_accuracy(*_binary_labels(rs))),
            'cohen_kappa': _bootstrap_ci(
                records, lambda rs: cohen_kappa(*_binary_labels(rs))),
            'mcc': _bootstrap_ci(records, lambda rs: mcc(*_binary_labels(rs))),
        },
    }
