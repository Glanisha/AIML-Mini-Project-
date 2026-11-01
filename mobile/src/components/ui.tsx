import type { ReactNode } from 'react';
import { ActivityIndicator, Pressable, Text, View } from 'react-native';

import type { Metric } from '@/lib/types';

type Tone = 'info' | 'warning' | 'danger' | 'success';

const toneStyles: Record<Tone, { box: string; title: string; body: string }> = {
  info: { box: 'bg-sky-50 border-sky-200', title: 'text-sky-900', body: 'text-sky-800' },
  warning: { box: 'bg-amber-50 border-amber-200', title: 'text-amber-900', body: 'text-amber-800' },
  danger: { box: 'bg-red-50 border-red-200', title: 'text-red-900', body: 'text-red-800' },
  success: { box: 'bg-emerald-50 border-emerald-200', title: 'text-emerald-900', body: 'text-emerald-800' },
};

interface BannerProps {
  tone: Tone;
  title: string;
  lines?: string[];
}

export function Banner({ tone, title, lines = [] }: BannerProps) {
  const s = toneStyles[tone];
  return (
    <View className={`rounded-xl border p-4 ${s.box}`}>
      <Text className={`font-semibold ${s.title}`}>{title}</Text>
      {lines.map((line) => (
        <Text key={line} className={`mt-1 text-sm ${s.body}`}>
          {line}
        </Text>
      ))}
    </View>
  );
}

type Variant = 'primary' | 'secondary' | 'ghost';

const containerStyles: Record<Variant, string> = {
  primary: 'bg-teal-700 active:bg-teal-800',
  secondary: 'bg-white border border-slate-300 active:bg-slate-100',
  ghost: 'active:bg-slate-100',
};

const textStyles: Record<Variant, string> = {
  primary: 'text-white',
  secondary: 'text-slate-800',
  ghost: 'text-teal-700',
};

interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: Variant;
  disabled?: boolean;
  loading?: boolean;
}

export function Button({ title, onPress, variant = 'primary', disabled, loading }: ButtonProps) {
  const inactive = disabled || loading;
  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      disabled={inactive}
      className={`flex-row items-center justify-center rounded-xl px-5 py-4 ${containerStyles[variant]} ${inactive ? 'opacity-50' : ''}`}
    >
      {loading ? (
        <ActivityIndicator color={variant === 'primary' ? '#fff' : '#0f766e'} />
      ) : (
        <Text className={`text-base font-semibold ${textStyles[variant]}`}>{title}</Text>
      )}
    </Pressable>
  );
}

export function Card({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <View className="rounded-2xl border border-slate-200 bg-white p-4">
      {title ? <Text className="mb-3 text-base font-semibold text-slate-900">{title}</Text> : null}
      {children}
    </View>
  );
}

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

const tagStyles = {
  teal: 'bg-teal-100 text-teal-800',
  red: 'bg-red-100 text-red-800',
  amber: 'bg-amber-100 text-amber-800',
  slate: 'bg-slate-200 text-slate-700',
  sky: 'bg-sky-100 text-sky-800',
} as const;

export type TagColor = keyof typeof tagStyles;

export function Tag({ label, color = 'slate' }: { label: string; color?: TagColor }) {
  const [bg, text] = tagStyles[color].split(' ');
  return (
    <View className={`self-start rounded-full px-2.5 py-0.5 ${bg}`}>
      <Text className={`text-xs font-medium ${text}`}>{label}</Text>
    </View>
  );
}
