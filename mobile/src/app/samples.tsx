import { router } from 'expo-router';
import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, Image, Pressable, ScrollView, Text, View } from 'react-native';

import { Banner, Button, Tag, type TagColor } from '@/components/ui';
import { getSamples } from '@/lib/api';
import type { SamplesResponse } from '@/lib/types';

const CATEGORY: Record<string, { label: string; color: TagColor }> = {
  malignant: { label: 'Malignant', color: 'red' },
  benign: { label: 'Benign', color: 'teal' },
  uncertain: { label: 'Borderline', color: 'amber' },
  misclassified: { label: 'Known error', color: 'slate' },
  degraded: { label: 'Poor photo', color: 'sky' },
};

export default function Samples() {
  const [data, setData] = useState<SamplesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchSamples = useCallback(() => {
    getSamples()
      .then(setData)
      .catch((e: Error) => setError(e.message));
  }, []);
  useEffect(fetchSamples, [fetchSamples]);

  if (error) {
    return (
      <View className="flex-1 gap-3 p-5">
        <Banner tone="danger" title="Could not load samples" lines={[error]} />
        <Button
          title="Retry"
          onPress={() => {
            setError(null);
            fetchSamples();
          }}
        />
      </View>
    );
  }
  if (!data) {
    return (
      <View className="flex-1 items-center justify-center">
        <ActivityIndicator color="#0f766e" />
      </View>
    );
  }

  return (
    <ScrollView className="flex-1" contentContainerClassName="gap-3 p-5 pb-10">
      <Text className="text-sm leading-5 text-slate-600">
        Real held-out test images that the model never saw during training or tuning. They include a known error, so you can see where the model fails as well as where it works.
      </Text>
      {data.samples.map((s) => {
        const c = CATEGORY[s.category] ?? { label: s.category, color: 'slate' as TagColor };
        return (
          <Pressable
            key={s.id}
            onPress={() =>
              router.push({
                pathname: '/analyze',
                params: { sampleId: s.id, sampleUrl: s.image_url, sampleTitle: s.title, sampleLabel: s.true_label },
              })
            }
            className="flex-row gap-3 rounded-2xl border border-slate-200 bg-white p-3 active:bg-slate-50"
          >
            <Image source={{ uri: s.image_url }} className="h-20 w-20 rounded-xl bg-slate-200" />
            <View className="flex-1 justify-center gap-1.5">
              <Tag label={c.label} color={c.color} />
              <Text className="font-medium text-slate-900">{s.title}</Text>
              <Text className="text-xs text-slate-500">Ground truth: {s.true_label}</Text>
            </View>
          </Pressable>
        );
      })}
      <Text className="mt-2 text-[11px] leading-4 text-slate-400">{data.attribution}</Text>
    </ScrollView>
  );
}
