import { Text, View } from 'react-native';

import type { Metric } from '@/lib/types';

const pct = (v: number) => `${Math.round(v * 100)}%`;
const dec = (v: number) => v.toFixed(2);

export function MetricTile({ label, metric, format = 'pct', hint }: { label: string; metric: Metric; format?: 'pct' | 'dec'; hint?: string }) {
  const f = format === 'pct' ? pct : dec;
  return (
    <View className="min-w-[46%] flex-1 rounded-xl bg-slate-50 p-3">
      <Text className="text-xs text-slate-500">{label}</Text>
      <Text className="mt-0.5 text-xl font-bold text-slate-900">{f(metric.value)}</Text>
      <Text className="text-[11px] text-slate-500">
        95% CI {f(metric.ci95[0])}–{f(metric.ci95[1])}
      </Text>
      {hint ? <Text className="mt-1 text-[11px] text-slate-400">{hint}</Text> : null}
    </View>
  );
}
