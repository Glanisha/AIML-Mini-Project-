import { Text, View } from 'react-native';

interface Props {
  label: string;
  value: number; // 0..1
  highlight?: boolean;
}

export function ScoreBar({ label, value, highlight }: Props) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  const textStyle = highlight ? 'font-semibold text-slate-900' : 'text-slate-600';
  return (
    <View className="mb-3">
      <View className="mb-1 flex-row justify-between">
        <Text className={`capitalize ${textStyle}`}>{label.replace(/_/g, ' ')}</Text>
        <Text className={textStyle}>{pct}%</Text>
      </View>
      <View className="h-2.5 overflow-hidden rounded-full bg-slate-200">
        <View className={`h-full rounded-full ${highlight ? 'bg-teal-600' : 'bg-slate-400'}`} style={{ width: `${pct}%` }} />
      </View>
    </View>
  );
}
