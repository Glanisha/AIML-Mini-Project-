import type { ReactNode } from 'react';
import { Text, View } from 'react-native';

export function Card({ title, children }: { title?: string; children: ReactNode }) {
  return (
    <View className="rounded-2xl border border-slate-200 bg-white p-4">
      {title ? <Text className="mb-3 text-base font-semibold text-slate-900">{title}</Text> : null}
      {children}
    </View>
  );
}
