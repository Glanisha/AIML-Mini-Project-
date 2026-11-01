import '../../global.css';

import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';

export default function RootLayout() {
  return (
    <>
      <StatusBar style="dark" />
      <Stack
        screenOptions={{
          headerTintColor: '#0f766e',
          headerTitleStyle: { color: '#0f172a' },
          contentStyle: { backgroundColor: '#f8fafc' },
        }}
      >
        <Stack.Screen name="index" options={{ headerShown: false }} />
        <Stack.Screen name="analyze" options={{ title: 'Analysis' }} />
        <Stack.Screen name="evidence" options={{ title: 'Model evidence' }} />
      </Stack>
    </>
  );
}
