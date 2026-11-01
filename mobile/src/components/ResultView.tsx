import { useState } from 'react';
import { Pressable, Text, View } from 'react-native';

import type { AnalyzeResponse, TriageLevel } from '@/lib/types';

import { Banner } from './Banner';
import { Card } from './Card';
import { RiskGauge } from './RiskGauge';
import { Tag } from './Tag';

export function triageStyle(level: TriageLevel) {
  return {
    refer: { box: 'bg-red-50 border-red-300', text: 'text-red-800', body: 'text-red-900', icon: '!' },
    routine: { box: 'bg-emerald-50 border-emerald-300', text: 'text-emerald-800', body: 'text-emerald-900', icon: '✓' },
    indeterminate: { box: 'bg-amber-50 border-amber-300', text: 'text-amber-800', body: 'text-amber-900', icon: '?' },
  }[level];
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

export function ResultView({ result }: { result: AnalyzeResponse }) {
  const t = triageStyle(result.triage.level);
  const { uncertainty: u, quality: q, ood } = result;
  const agreeCount = Math.round(u.member_agreement * u.member_probabilities.length);

  return (
    <View className="gap-4">
      {result.is_mock ? <Banner tone="info" title="Demo mode" lines={['The server has no trained model installed. Scores are placeholders.']} /> : null}

      <View className={`rounded-2xl border-2 p-4 ${t.box}`}>
        <View className="flex-row items-center gap-3">
          <View className="h-9 w-9 items-center justify-center rounded-full bg-white">
            <Text className={`text-lg font-bold ${t.text}`}>{t.icon}</Text>
          </View>
          <Text className={`flex-1 text-lg font-bold ${t.text}`}>{result.triage.title}</Text>
        </View>
        <Text className={`mt-3 text-sm leading-5 ${t.body}`}>{result.triage.message}</Text>
      </View>

      {result.abstained && result.abstain_reasons.length ? (
        <Card title="Why there is no result">
          {result.abstain_reasons.map((r) => (
            <Text key={r} className="mb-1 text-sm text-slate-700">
              • {r}
            </Text>
          ))}
        </Card>
      ) : null}

      <Card title="Malignancy-likeness score">
        <View className="mb-2 flex-row items-end gap-2">
          <Text className="text-3xl font-bold text-slate-900">{pct(result.positive_probability)}</Text>
          <Text className="mb-1 text-sm text-slate-500">calibrated ensemble score</Text>
        </View>
        <RiskGauge
          probability={result.positive_probability}
          threshold={result.operating_threshold}
          margin={result.abstain_margin}
          members={u.member_probabilities}
        />
        <View className="mt-3 flex-row flex-wrap gap-2">
          {u.member_probabilities.length ? (
            <Tag label={`${agreeCount}/${u.member_probabilities.length} models agree`} color={u.member_agreement >= 0.8 ? 'teal' : 'amber'} />
          ) : null}
          <Tag label={`Model range ${pct(u.score_range[0])}–${pct(u.score_range[1])}`} color="slate" />
          <Tag label={`Spread ±${pct(u.member_std)}`} color={u.member_std > 0.15 ? 'amber' : 'slate'} />
        </View>
      </Card>

      <View className="flex-row gap-3">
        <StatusCard
          title="Image quality"
          status={q.status === 'ok' ? 'Good' : q.status === 'warning' ? 'Warning' : 'Too poor'}
          color={q.status === 'ok' ? 'teal' : q.status === 'warning' ? 'amber' : 'red'}
          lines={q.reasons}
        />
        <StatusCard
          title="Familiarity"
          status={ood.in_distribution ? 'Looks familiar' : 'Unfamiliar image'}
          color={ood.in_distribution ? 'teal' : 'red'}
          lines={ood.in_distribution ? [] : ['Unlike the oral lesion photos used in training.']}
        />
      </View>

      <TechnicalDetails result={result} />
    </View>
  );
}

function StatusCard({ title, status, color, lines }: { title: string; status: string; color: 'teal' | 'amber' | 'red'; lines: string[] }) {
  return (
    <View className="flex-1 rounded-2xl border border-slate-200 bg-white p-3">
      <Text className="mb-2 text-xs text-slate-500">{title}</Text>
      <Tag label={status} color={color} />
      {lines.map((l) => (
        <Text key={l} className="mt-2 text-xs text-slate-600">
          {l}
        </Text>
      ))}
    </View>
  );
}

function TechnicalDetails({ result }: { result: AnalyzeResponse }) {
  const [open, setOpen] = useState(false);
  const rows: [string, string][] = [
    ['Model', result.model_version],
    ['Referral threshold', `${pct(result.operating_threshold)} (tuned for ≥90% sensitivity)`],
    ['Uncertain band', `±${pct(result.abstain_margin)} around threshold`],
    ['Familiarity distance', `${result.ood.score.toFixed(3)} (limit ${result.ood.threshold.toFixed(3)})`],
    ...Object.entries(result.class_scores).map(([k, v]) => [`P(${k})`, pct(v)] as [string, string]),
    ...Object.entries(result.quality.metrics).map(([k, v]) => [k.replace(/_/g, ' '), v.toFixed(k.includes('fraction') ? 3 : 1)] as [string, string]),
    ['Server time', `${Math.round(result.latency_ms)} ms (${Object.entries(result.timings_ms).map(([k, v]) => `${k} ${Math.round(v)}`).join(', ')})`],
  ];
  return (
    <View className="rounded-2xl border border-slate-200 bg-white">
      <Pressable onPress={() => setOpen((o) => !o)} className="flex-row items-center justify-between p-4">
        <Text className="font-semibold text-slate-900">Technical details</Text>
        <Text className="text-teal-700">{open ? 'Hide' : 'Show'}</Text>
      </Pressable>
      {open ? (
        <View className="border-t border-slate-100 px-4 pb-4 pt-2">
          {rows.map(([k, v]) => (
            <View key={k} className="flex-row justify-between gap-4 py-1">
              <Text className="text-xs capitalize text-slate-500">{k}</Text>
              <Text className="flex-1 text-right text-xs text-slate-800">{v}</Text>
            </View>
          ))}
        </View>
      ) : null}
    </View>
  );
}
