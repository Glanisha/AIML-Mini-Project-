import { useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { Alert, Platform, ScrollView, Text, View } from 'react-native';

import { HistoryRow } from '@/components/HistoryRow';
import { Button } from '@/components/ui';
import { clearHistory, listHistory, type HistoryEntry } from '@/lib/history';

export default function History() {
  const [entries, setEntries] = useState<HistoryEntry[] | null>(null);

  useFocusEffect(
    useCallback(() => {
      listHistory().then(setEntries);
    }, []),
  );

  const confirmClear = () => {
    const doClear = async () => {
      await clearHistory();
      setEntries([]);
    };
    if (Platform.OS === 'web') {
      // Alert buttons are not supported on web
      if (globalThis.confirm?.('Delete all saved results from this device?')) doClear();
      return;
    }
    Alert.alert('Delete history?', 'All saved results and photos will be removed from this device.', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: doClear },
    ]);
  };

  return (
    <ScrollView className="flex-1" contentContainerClassName="gap-3 p-5 pb-10">
      <Text className="text-sm text-slate-600">Stored only on this device. Nothing is kept on the server.</Text>
      {entries?.length === 0 ? <Text className="py-10 text-center text-slate-400">No saved analyses yet.</Text> : null}
      {entries?.map((e) => (
        <HistoryRow key={e.id} entry={e} />
      ))}
      {entries?.length ? (
        <View className="mt-4">
          <Button title="Delete all history" variant="secondary" onPress={confirmClear} />
        </View>
      ) : null}
    </ScrollView>
  );
}
