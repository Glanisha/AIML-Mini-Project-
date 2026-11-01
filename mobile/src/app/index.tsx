import { router } from 'expo-router';
import { Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Button } from '@/components/Button';
import { ServerStatus } from '@/components/ServerStatus';
import { pickImage, type ImageSource } from '@/lib/pickImage';

export default function Home() {
  const start = async (source: ImageSource) => {
    const uri = await pickImage(source);
    if (uri) router.push({ pathname: '/analyze', params: { uri } });
  };

  return (
    <SafeAreaView className="flex-1 bg-slate-50">
      <View className="flex-1 justify-between px-6 py-8">
        <View className="mt-10">
          <Text className="text-4xl font-bold text-teal-700">OralSight</Text>
          <Text className="mt-2 text-lg text-slate-600">Oral lesion screening and review prototype</Text>

          <View className="mt-8 gap-3">
            <Text className="text-sm text-slate-500">For a usable result:</Text>
            <Text className="text-sm text-slate-700">• Use good, even lighting. Avoid flash glare.</Text>
            <Text className="text-sm text-slate-700">• Hold steady and keep the lesion in focus.</Text>
            <Text className="text-sm text-slate-700">• Crop close to the area of interest.</Text>
          </View>
        </View>

        <View className="gap-3">
          <Button title="Take photo" onPress={() => start('camera')} />
          <Button title="Choose from gallery" variant="secondary" onPress={() => start('library')} />
          <Button title="View model evidence" variant="ghost" onPress={() => router.push('/evidence')} />

          <View className="mt-2">
            <ServerStatus />
          </View>
          <Text className="mt-2 text-center text-xs text-slate-400">
            Academic prototype. Not a diagnosis. No cancer staging or treatment advice.
          </Text>
        </View>
      </View>
    </SafeAreaView>
  );
}
