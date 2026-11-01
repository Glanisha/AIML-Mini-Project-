import Constants from 'expo-constants';
import { File } from 'expo-file-system';
import { fetch } from 'expo/fetch';
import { Platform } from 'react-native';

import type { AnalyzeResponse, HealthResponse, MetricsResponse, ModelCard, SamplesResponse } from './types';

// Free hosts can cold-start (Render sleeps after 15 min idle), so allow a generous first request
const REQUEST_TIMEOUT_MS = 60_000;

// Explicit URL wins (required for APK builds). Otherwise, during development,
// reach the backend on the same machine that is running the Expo dev server.
function resolveBaseUrl(): string {
  const explicit = process.env.EXPO_PUBLIC_API_URL;
  if (explicit) return explicit.replace(/\/+$/, '');

  // 127.0.0.1, not localhost: localhost may resolve to IPv6 (::1) and reach a different
  // service bound to [::]:8000 (e.g. a Docker container) instead of this backend
  const host = Platform.OS === 'web' ? '127.0.0.1' : Constants.expoConfig?.hostUri?.split(':')[0];
  return `http://${host ?? '127.0.0.1'}:8000`;
}

export const API_BASE_URL = resolveBaseUrl();

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: Parameters<typeof fetch>[1]): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let res: Response;
  try {
    res = (await fetch(`${API_BASE_URL}${path}`, { ...init, signal: controller.signal })) as unknown as Response;
  } catch {
    throw new ApiError(
      controller.signal.aborted
        ? 'The server took too long to respond. If it was asleep, try again in a moment.'
        : `Cannot reach the server at ${API_BASE_URL}. Check your connection.`,
    );
  } finally {
    clearTimeout(timer);
  }

  if (!res.ok) {
    let detail = `Request failed (${res.status}).`;
    try {
      const body = await res.json();
      if (typeof body?.detail === 'string') detail = body.detail;
    } catch {
      // non-JSON error body; keep the generic message
    }
    throw new ApiError(detail, res.status);
  }
  return (await res.json()) as T;
}

export const getHealth = () => request<HealthResponse>('/health');
export const getMetrics = () => request<MetricsResponse>('/metrics');
export const getModelCard = () => request<ModelCard>('/model-card');
export const getSamples = () => request<SamplesResponse>('/samples');

export const analyzeSample = (id: string) =>
  request<AnalyzeResponse>(`/samples/${encodeURIComponent(id)}/analyze`, { method: 'POST' });

export async function analyzeImage(uri: string) {
  const form = new FormData();
  if (Platform.OS === 'web') {
    // On web the picker returns a blob:/data: URI; expo-file-system is native-only
    const blob = await (await globalThis.fetch(uri)).blob();
    form.append('file', blob, 'upload.jpg');
  } else {
    form.append('file', new File(uri));
  }
  return request<AnalyzeResponse>('/analyze', { method: 'POST', body: form });
}
