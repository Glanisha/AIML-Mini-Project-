import { useState } from 'react';
import { Pressable, ScrollView, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Button } from './ui';

const POINTS = [
  ['Research prototype', 'Not a medical device and not clinically validated.'],
  ['Not a diagnosis', 'It estimates whether a photo resembles benign or malignant lesions from a small training dataset. Only a clinician, usually with a biopsy, can diagnose oral cancer.'],
  ['No staging or treatment advice', 'The app never stages disease or recommends treatment.'],
  ['Your privacy', 'Photos are analysed in memory and never stored on the server. History is kept only on this device and can be deleted at any time.'],
  ['When in doubt, get checked', 'Any mouth ulcer, white or red patch, or lump lasting more than 2 weeks should be examined by a dentist or doctor, whatever this app shows.'],
];

export function ConsentScreen({ onAccept }: { onAccept: () => void }) {
  const [checked, setChecked] = useState(false);

  return (
    <SafeAreaView className="flex-1 bg-slate-50">
      <ScrollView contentContainerClassName="gap-4 px-6 py-8">
        <Text className="text-3xl font-bold text-teal-700">Before you start</Text>
        {POINTS.map(([title, body]) => (
          <View key={title} className="rounded-2xl border border-slate-200 bg-white p-4">
            <Text className="font-semibold text-slate-900">{title}</Text>
            <Text className="mt-1 text-sm leading-5 text-slate-600">{body}</Text>
          </View>
        ))}
        <Pressable onPress={() => setChecked((c) => !c)} className="flex-row items-center gap-3 py-2" accessibilityRole="checkbox" accessibilityState={{ checked }}>
          <View className={`h-6 w-6 items-center justify-center rounded-md border-2 ${checked ? 'border-teal-700 bg-teal-700' : 'border-slate-400 bg-white'}`}>
            {checked ? <Text className="font-bold text-white">✓</Text> : null}
          </View>
          <Text className="flex-1 text-sm text-slate-700">I understand this is a research prototype and not a substitute for professional care.</Text>
        </Pressable>
        <Button title="Continue" onPress={onAccept} disabled={!checked} />
      </ScrollView>
    </SafeAreaView>
  );
}
