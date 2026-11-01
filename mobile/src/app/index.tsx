import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { Pressable, ScrollView, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { HistoryRow } from '@/components/HistoryRow';
import { ServerStatus } from '@/components/ServerStatus';
import { Button } from '@/components/ui';
import { listHistory, type HistoryEntry } from '@/lib/history';
import { pickImage, type ImageSource } from '@/lib/pickImage';

export default function Home() {
  const [recent, setRecent] = useState<HistoryEntry[]>([]);

  useFocusEffect(
    useCallback(() => {
      listHistory().then((h) => setRecent(h.slice(0, 3)));
    }, []),
  );

  const start = async (source: ImageSource) => {
    const uri = await pickImage(source);
    if (uri) router.push({ pathname: '/analyze', params: { uri, source } });
  };

  return (
    <SafeAreaView className="flex-1 bg-slate-50">
      <ScrollView contentContainerClassName="gap-6 px-6 pb-10 pt-8">
        <View>
          <Text className="text-4xl font-bold text-slate-900">OralSight</Text>
          <Text className="mt-1 text-base text-slate-600">Oral lesion screening</Text>
        </View>

        <View className="gap-3">
          <Button title="Take photo" onPress={() => start('camera')} />
          <Button title="Choose from gallery" variant="secondary" onPress={() => start('library')} />
          <Button title="Try a sample case" variant="secondary" onPress={() => router.push('/samples')} />
        </View>

        <View className="rounded-2xl border border-slate-200 bg-white p-4">
          <Text className="mb-3 font-semibold text-slate-900">For a usable photo</Text>
          <Text className="text-sm leading-6 text-slate-700">• Bright, even light, with no flash glare{'\n'}• Hold steady; tap the lesion to focus{'\n'}• Crop close to the area of concern</Text>
        </View>

        {recent.length ? (
          <View>
            <View className="mb-3 flex-row items-center justify-between">
              <Text className="font-semibold text-slate-900">Recent</Text>
              <Pressable onPress={() => router.push('/history')}>
                <Text className="text-sm font-medium text-teal-700">See all</Text>
              </Pressable>
            </View>
            <View className="gap-2">
              {recent.map((e) => (
                <HistoryRow key={e.id} entry={e} />
              ))}
            </View>
          </View>
        ) : null}

        <Button title="Model evidence" variant="ghost" onPress={() => router.push('/evidence')} />

        <View className="gap-2">
          <ServerStatus />
          <Text className="text-center text-xs text-slate-400">Research prototype. Not a diagnosis.</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}
