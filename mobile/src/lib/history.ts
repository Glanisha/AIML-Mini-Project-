// On-device history only: results never leave the phone and the server keeps nothing.
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Directory, File, Paths } from 'expo-file-system';
import { Platform } from 'react-native';

import type { AnalyzeResponse } from './types';

const KEY = 'oralsight.history.v1';
const MAX_ENTRIES = 20;

export interface HistoryEntry {
  id: string;
  createdAt: string;
  imageUri: string | null;
  source: 'camera' | 'library' | 'sample';
  sampleTitle?: string;
  result: AnalyzeResponse;
}

export async function listHistory(): Promise<HistoryEntry[]> {
  try {
    const raw = await AsyncStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

export async function getHistoryEntry(id: string) {
  return (await listHistory()).find((e) => e.id === id) ?? null;
}

// Copy the photo into app storage so the entry survives the picker's temporary cache
function persistImage(uri: string, id: string): string | null {
  if (Platform.OS === 'web') return uri.startsWith('http') ? uri : null;
  if (uri.startsWith('http')) return uri;
  try {
    const dir = new Directory(Paths.document, 'history');
    if (!dir.exists) dir.create();
    const dest = new File(dir, `${id}.jpg`);
    new File(uri).copy(dest);
    return dest.uri;
  } catch {
    return null;
  }
}

export async function saveToHistory(entry: Omit<HistoryEntry, 'id' | 'createdAt' | 'imageUri'> & { imageUri: string }) {
  const id = `${Date.now()}`;
  const full: HistoryEntry = { ...entry, id, createdAt: new Date().toISOString(), imageUri: persistImage(entry.imageUri, id) };
  const entries = [full, ...(await listHistory())];
  for (const old of entries.slice(MAX_ENTRIES)) deleteImage(old);
  await AsyncStorage.setItem(KEY, JSON.stringify(entries.slice(0, MAX_ENTRIES)));
  return full;
}

function deleteImage(entry: HistoryEntry) {
  if (Platform.OS === 'web' || !entry.imageUri?.startsWith('file:')) return;
  try {
    new File(entry.imageUri).delete();
  } catch {
    // already gone
  }
}

export async function clearHistory() {
  (await listHistory()).forEach(deleteImage);
  await AsyncStorage.removeItem(KEY);
}

const CONSENT_KEY = 'oralsight.consent.v1';
export const hasConsented = async () => (await AsyncStorage.getItem(CONSENT_KEY)) === 'yes';
export const giveConsent = () => AsyncStorage.setItem(CONSENT_KEY, 'yes');
