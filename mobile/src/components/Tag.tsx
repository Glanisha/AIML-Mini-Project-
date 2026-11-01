import { Text, View } from 'react-native';

const styles = {
  teal: 'bg-teal-100 text-teal-800',
  red: 'bg-red-100 text-red-800',
  amber: 'bg-amber-100 text-amber-800',
  slate: 'bg-slate-200 text-slate-700',
  sky: 'bg-sky-100 text-sky-800',
} as const;

export type TagColor = keyof typeof styles;

export function Tag({ label, color = 'slate' }: { label: string; color?: TagColor }) {
  const [bg, text] = styles[color].split(' ');
  return (
    <View className={`self-start rounded-full px-2.5 py-0.5 ${bg}`}>
      <Text className={`text-xs font-medium ${text}`}>{label}</Text>
    </View>
  );
}
