import { router } from 'expo-router';
import { Image, Pressable, Text, View } from 'react-native';

import type { HistoryEntry } from '@/lib/history';

import { triageStyle } from './ResultView';

export function HistoryRow({ entry }: { entry: HistoryEntry }) {
  const t = triageStyle(entry.result.triage.level);
  return (
    <Pressable
      onPress={() => router.push({ pathname: '/analyze', params: { historyId: entry.id } })}
      className="flex-row items-center gap-3 rounded-xl border border-slate-200 bg-white p-3 active:bg-slate-50"
    >
      {entry.imageUri ? (
        <Image source={{ uri: entry.imageUri }} className="h-12 w-12 rounded-lg bg-slate-200" />
      ) : (
        <View className="h-12 w-12 rounded-lg bg-slate-200" />
      )}
      <View className="flex-1">
        <Text className={`font-semibold ${t.text}`}>{entry.result.triage.title}</Text>
        <Text className="text-xs text-slate-500">
          {new Date(entry.createdAt).toLocaleString()} · score {Math.round(entry.result.positive_probability * 100)}%
          {entry.sampleTitle ? ' · sample case' : ''}
        </Text>
      </View>
    </Pressable>
  );
}
