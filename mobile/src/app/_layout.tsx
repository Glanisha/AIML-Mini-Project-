import '../../global.css';

import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { useEffect, useState } from 'react';
import { ActivityIndicator, View } from 'react-native';

import { ConsentScreen } from '@/components/ConsentScreen';
import { giveConsent, hasConsented } from '@/lib/history';

export default function RootLayout() {
  const [consented, setConsented] = useState<boolean | null>(null);

  useEffect(() => {
    hasConsented().then(setConsented);
  }, []);

  if (consented === null) {
    return (
      <View className="flex-1 items-center justify-center bg-slate-50">
        <ActivityIndicator color="#0f766e" />
      </View>
    );
  }

  if (!consented) {
    return (
      <ConsentScreen
        onAccept={async () => {
          await giveConsent();
          setConsented(true);
        }}
      />
    );
  }

  return (
    <>
      <StatusBar style="dark" />
      <Stack
        screenOptions={{
          headerTintColor: '#0f766e',
          headerTitleStyle: { color: '#0f172a' },
          headerShadowVisible: false,
          headerStyle: { backgroundColor: '#f8fafc' },
          contentStyle: { backgroundColor: '#f8fafc' },
        }}
      >
        <Stack.Screen name="index" options={{ headerShown: false }} />
        <Stack.Screen name="analyze" options={{ title: 'Analysis' }} />
        <Stack.Screen name="samples" options={{ title: 'Sample cases' }} />
        <Stack.Screen name="history" options={{ title: 'History' }} />
        <Stack.Screen name="evidence" options={{ title: 'Model evidence' }} />
        <Stack.Screen name="about" options={{ title: 'About' }} />
      </Stack>
    </>
  );
}
