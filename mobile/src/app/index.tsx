import { router, useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { Pressable, ScrollView, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { HistoryRow } from '@/components/HistoryRow';
import { ServerStatus } from '@/components/ServerStatus';
import { Button } from '@/components/ui';
import { listHistory, type HistoryEntry } from '@/lib/history';
import { pickImage, type ImageSource } from '@/lib/pickImage';

const PIPELINE = [
  ['Quality gate', 'Blur and exposure checked against the training images'],
  ['Familiarity check', 'Rejects photos unlike anything the model has seen'],
  ['5-model ensemble', 'Calibrated scores plus how much the models agree'],
  ['Explanation', 'Heatmap of the regions that drove the score'],
];

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
          <Text className="text-xs font-semibold uppercase tracking-widest text-teal-700">AI in Healthcare · Research prototype</Text>
          <Text className="mt-2 text-4xl font-bold text-slate-900">OralSight</Text>
          <Text className="mt-2 text-base leading-6 text-slate-600">
            Photograph an oral lesion and get a calibrated, explainable screening estimate that says when it is unsure.
          </Text>
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

        <View>
          <Text className="mb-3 font-semibold text-slate-900">What happens to your photo</Text>
          <View className="gap-2">
            {PIPELINE.map(([title, body], i) => (
              <View key={title} className="flex-row gap-3 rounded-xl bg-white p-3">
                <View className="h-7 w-7 items-center justify-center rounded-full bg-teal-700">
                  <Text className="text-xs font-bold text-white">{i + 1}</Text>
                </View>
                <View className="flex-1">
                  <Text className="font-medium text-slate-900">{title}</Text>
                  <Text className="text-xs text-slate-500">{body}</Text>
                </View>
              </View>
            ))}
          </View>
        </View>

        {recent.length ? (
          <View>
            <View className="mb-3 flex-row items-center justify-between">
              <Text className="font-semibold text-slate-900">Recent on this device</Text>
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

        <View className="flex-row gap-3">
          <View className="flex-1">
            <Button title="Model evidence" variant="ghost" onPress={() => router.push('/evidence')} />
          </View>
          <View className="flex-1">
            <Button title="About" variant="ghost" onPress={() => router.push('/about')} />
          </View>
        </View>

        <View className="gap-2">
          <ServerStatus />
          <Text className="text-center text-xs text-slate-400">Not a diagnosis. No cancer staging or treatment advice.</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}
