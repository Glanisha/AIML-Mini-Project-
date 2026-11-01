import { Text, View } from 'react-native';

type Tone = 'info' | 'warning' | 'danger' | 'success';

const toneStyles: Record<Tone, { box: string; title: string; body: string }> = {
  info: { box: 'bg-sky-50 border-sky-200', title: 'text-sky-900', body: 'text-sky-800' },
  warning: { box: 'bg-amber-50 border-amber-200', title: 'text-amber-900', body: 'text-amber-800' },
  danger: { box: 'bg-red-50 border-red-200', title: 'text-red-900', body: 'text-red-800' },
  success: { box: 'bg-emerald-50 border-emerald-200', title: 'text-emerald-900', body: 'text-emerald-800' },
};

interface Props {
  tone: Tone;
  title: string;
  lines?: string[];
}

export function Banner({ tone, title, lines = [] }: Props) {
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
