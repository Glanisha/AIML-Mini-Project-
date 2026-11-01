import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, ScrollView, Text, View } from 'react-native';

import { Banner } from '@/components/Banner';
import { Button } from '@/components/Button';
import { Card } from '@/components/Card';
import { LineChart } from '@/components/LineChart';
import { MetricTile } from '@/components/MetricTile';
import { getMetrics } from '@/lib/api';
import type { MetricsResponse, RobustnessRow } from '@/lib/types';

const TEAL = '#0f766e';
const GREY = '#94a3b8';
const RED = '#dc2626';
const AMBER = '#d97706';
const pct = (v: number) => `${Math.round(v * 100)}%`;

export default function Evidence() {
  const [m, setM] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = useCallback(() => {
    getMetrics()
      .then(setM)
      .catch((e: Error) => setError(e.message));
  }, []);
  useEffect(fetchMetrics, [fetchMetrics]);

  if (error) {
    return (
      <View className="flex-1 gap-3 p-5">
        <Banner tone="danger" title="Could not load metrics" lines={[error]} />
        <Button
          title="Retry"
          onPress={() => {
            setError(null);
            fetchMetrics();
          }}
        />
      </View>
    );
  }
  if (!m) {
    return (
      <View className="flex-1 items-center justify-center">
        <ActivityIndicator color={TEAL} />
      </View>
    );
  }

  const h = m.headline;
  const { labels, matrix } = m.confusion_matrix;

  return (
    <ScrollView className="flex-1" contentContainerClassName="gap-4 p-5 pb-12">
      {m.is_mock ? <Banner tone="info" title="Placeholder metrics" lines={['These are not real results.']} /> : null}

      <Text className="text-sm leading-5 text-slate-600">
        Results on {m.test_size} held-out test images never used for training, tuning or model selection. Every figure here comes from the saved evaluation of the deployed model
        ({m.model_version}).
      </Text>

      {h ? (
        <Card title="Headline performance">
          <View className="flex-row flex-wrap gap-2">
            <MetricTile label="ROC-AUC" metric={h.roc_auc} format="dec" />
            <MetricTile label="Sensitivity (malignant)" metric={h.sensitivity} hint="Share of malignant lesions flagged" />
            <MetricTile label="Specificity" metric={h.specificity} hint="Share of benign lesions not flagged" />
            <MetricTile label="Balanced accuracy" metric={h.balanced_accuracy} />
            <MetricTile label="NPV" metric={h.npv} />
            <MetricTile label="PR-AUC" metric={h.pr_auc} format="dec" />
          </View>
          {m.operating_threshold != null ? (
            <Text className="mt-3 text-xs text-slate-500">
              At the deployed referral threshold of {pct(m.operating_threshold)}, chosen on cross-validation to reach ≥90% sensitivity. 95% CIs from 2,000 stratified bootstrap
              resamples.
            </Text>
          ) : null}
        </Card>
      ) : null}

      <Card title="Confusion matrix (test set)">
        <Text className="mb-2 text-xs text-slate-500">Rows: true class · Columns: predicted class</Text>
        <View className="flex-row">
          <View className="flex-1" />
          {labels.map((l) => (
            <Text key={l} className="flex-1 text-center text-xs font-semibold capitalize text-slate-600">
              {l}
            </Text>
          ))}
        </View>
        {matrix.map((row, i) => (
          <View key={labels[i]} className="mt-1 flex-row items-center">
            <Text className="flex-1 text-xs font-semibold capitalize text-slate-600">{labels[i]}</Text>
            {row.map((count, j) => (
              <View key={j} className={`mx-0.5 flex-1 items-center rounded-md py-3 ${i === j ? 'bg-teal-100' : 'bg-red-50'}`}>
                <Text className="text-lg font-semibold text-slate-900">{count}</Text>
              </View>
            ))}
          </View>
        ))}
      </Card>

      {m.curves && h ? (
        <Card title="ROC curve vs baseline">
          <LineChart
            series={[
              { label: `OralSight (AUC ${h.roc_auc.value.toFixed(2)})`, color: TEAL, points: m.curves.roc },
              { label: 'Frozen-feature baseline', color: GREY, points: m.curves.roc_baseline, dashed: true },
            ]}
            diagonal
            markers={[{ x: 1 - h.specificity.value, y: h.sensitivity.value, color: RED, label: 'Operating point' }]}
            xLabel="1 − specificity"
            yLabel="Sensitivity"
          />
          {m.deployed_vs_baseline_auc ? (
            <Text className="mt-2 text-xs text-slate-500">
              AUC gain over baseline: {m.deployed_vs_baseline_auc.difference >= 0 ? '+' : ''}
              {m.deployed_vs_baseline_auc.difference.toFixed(3)} (95% CI {m.deployed_vs_baseline_auc.ci95[0].toFixed(3)} to {m.deployed_vs_baseline_auc.ci95[1].toFixed(3)}, paired
              bootstrap)
            </Text>
          ) : null}
        </Card>
      ) : null}

      {m.comparison ? (
        <Card title="Model comparison (test set)">
          {m.comparison.map((c) => (
            <View key={c.key} className={`mb-2 rounded-xl p-3 ${c.key === 'deployed' ? 'bg-teal-50' : 'bg-slate-50'}`}>
              <Text className="text-sm font-medium text-slate-900">{c.name}</Text>
              <Text className="mt-1 text-xs text-slate-600">
                AUC {c.roc_auc.toFixed(3)} · Sens {pct(c.sensitivity)} · Spec {pct(c.specificity)} · Acc {pct(c.accuracy)}
              </Text>
            </View>
          ))}
          <Text className="text-xs text-slate-500">Architecture chosen on cross-validation AUC, not on these test numbers.</Text>
        </Card>
      ) : null}

      {m.calibration ? (
        <Card title="Calibration: can the scores be trusted as probabilities?">
          <LineChart
            series={[
              {
                label: `Before (ECE ${m.calibration.ece_before.toFixed(3)})`,
                color: GREY,
                points: m.calibration.reliability_before.map((b) => [b.mean_predicted, b.observed_rate]),
                dashed: true,
                dots: true,
              },
              {
                label: `After temperature scaling (ECE ${m.calibration.ece_after.toFixed(3)})`,
                color: TEAL,
                points: m.calibration.reliability_after.map((b) => [b.mean_predicted, b.observed_rate]),
                dots: true,
              },
            ]}
            diagonal
            xLabel="Predicted P(malignant)"
            yLabel="Observed rate"
          />
          <Text className="mt-2 text-xs text-slate-500">
            Points on the diagonal mean a score of 70% really is malignant ~70% of the time. Brier score {m.calibration.brier_before.toFixed(3)} → {m.calibration.brier_after.toFixed(3)}.
          </Text>
        </Card>
      ) : null}

      {m.risk_coverage && m.selective ? (
        <Card title="Knowing when not to answer">
          <LineChart
            series={[{ label: `Error rate (AURC ${m.risk_coverage.aurc.toFixed(3)})`, color: TEAL, points: m.risk_coverage.points.map((p) => [p.coverage, p.risk]) }]}
            verticalLine={{ x: m.selective.coverage, color: AMBER, label: `Deployed: answers ${pct(m.selective.coverage)}` }}
            xLabel="Share of cases answered"
            yLabel="Error rate"
            yDomain={[0, Math.max(0.5, ...m.risk_coverage.points.map((p) => p.risk))]}
          />
          <Text className="mt-2 text-xs text-slate-500">
            The model abstains on {m.selective.abstained} of {m.test_size} test images. On the cases it does answer, sensitivity is{' '}
            {m.selective.sensitivity != null ? pct(m.selective.sensitivity) : 'n/a'} and specificity {m.selective.specificity != null ? pct(m.selective.specificity) : 'n/a'}.
          </Text>
        </Card>
      ) : null}

      {m.decision_curve ? (
        <Card title="Decision curve: is it clinically useful?">
          <LineChart
            series={[
              { label: 'OralSight', color: TEAL, points: m.decision_curve.map((r) => [r.threshold, r.model]) },
              { label: 'Refer everyone', color: GREY, points: m.decision_curve.map((r) => [r.threshold, r.treat_all]), dashed: true },
              { label: 'Refer no one', color: '#0f172a', points: m.decision_curve.map((r) => [r.threshold, 0]) },
            ]}
            xDomain={[0.05, 0.8]}
            yDomain={[-0.1, 0.6]}
            xLabel="Threshold probability"
            yLabel="Net benefit"
          />
          <Text className="mt-2 text-xs text-slate-500">
            Net benefit weighs true referrals against unnecessary ones. A curve above both reference lines means using the model beats referring everyone or no one at that
            threshold, for this dataset&apos;s ~50% prevalence.
          </Text>
        </Card>
      ) : null}

      {m.robustness ? <RobustnessCard robustness={m.robustness} /> : null}

      {m.ood ? (
        <Card title="Rejecting non-oral images">
          {Object.entries(m.ood.sets).map(([name, s]) => (
            <View key={name} className="mb-2 flex-row justify-between">
              <Text className="flex-1 text-sm capitalize text-slate-700">{name.replace(/_/g, ' ')}</Text>
              <Text className="text-sm text-slate-900">
                {pct(s.detected_rate)} rejected · AUROC {s.auroc.toFixed(2)}
              </Text>
            </View>
          ))}
          <Text className="mt-1 text-xs text-slate-500">Genuine test lesions wrongly rejected: {pct(m.ood.in_distribution_test_flagged_rate)}.</Text>
        </Card>
      ) : null}

      {m.audit ? (
        <Card title="Data">
          <Text className="text-sm text-slate-700">
            {m.audit.n_images} original images ({Object.entries(m.audit.class_counts).map(([k, v]) => `${v} ${k}`).join(', ')}). {m.audit.near_duplicate_pairs} near-duplicate
            pairs grouped into {m.audit.n_groups} groups so that no lesion appears in both training and test data.
          </Text>
          <Text className="mt-2 text-xs text-slate-500">{m.dataset}</Text>
        </Card>
      ) : null}

      {m.cpu_latency_ms != null ? (
        <Card title="Runtime">
          <Text className="text-sm text-slate-700">
            Median {Math.round(m.cpu_latency_ms)} ms per image for the full pipeline (quality, familiarity, 5 models, heatmap) on 2 CPU threads with ONNX Runtime. No GPU needed.
          </Text>
        </Card>
      ) : null}

      {m.notes.length ? <Banner tone="warning" title="Limitations" lines={m.notes} /> : null}
    </ScrollView>
  );
}

function RobustnessCard({ robustness }: { robustness: NonNullable<MetricsResponse['robustness']> }) {
  const entries = Object.entries(robustness).filter(([k]) => k !== 'clean') as [string, RobustnessRow[]][];
  return (
    <Card title="Robustness to poor photos">
      <Text className="mb-3 text-xs text-slate-500">AUC on the test set after synthetic degradation, and how often the app refuses to answer instead.</Text>
      {entries.map(([name, rows]) => (
        <View key={name} className="mb-3">
          <Text className="mb-1 text-sm font-medium capitalize text-slate-800">{name.replace(/_/g, ' ')}</Text>
          {rows.map((r) => (
            <View key={r.level} className="flex-row items-center gap-2 py-0.5">
              <Text className="w-14 text-xs text-slate-500">{r.level}</Text>
              <View className="h-2 flex-1 overflow-hidden rounded-full bg-slate-200">
                <View className="h-full rounded-full bg-teal-600" style={{ width: `${Math.round(r.auc * 100)}%` }} />
              </View>
              <Text className="w-28 text-right text-xs text-slate-700">
                AUC {r.auc.toFixed(2)} · abstain {pct(r.abstain_rate)}
              </Text>
            </View>
          ))}
        </View>
      ))}
    </Card>
  );
}
