import { router, useLocalSearchParams } from 'expo-router';
import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Alert, Image, Pressable, ScrollView, Text, View } from 'react-native';

import { ResultView } from '@/components/ResultView';
import { Banner, Button, Tag } from '@/components/ui';
import { analyzeImage, analyzeSample } from '@/lib/api';
import { getHistoryEntry, saveToHistory } from '@/lib/history';
import { pickImage, type ImageSource } from '@/lib/pickImage';
import { exportReport } from '@/lib/report';
import type { AnalyzeResponse } from '@/lib/types';

type Params = {
  uri?: string;
  source?: ImageSource;
  sampleId?: string;
  sampleUrl?: string;
  sampleTitle?: string;
  sampleLabel?: string;
  historyId?: string;
};

type State =
  | { kind: 'preview' }
  | { kind: 'loading' }
  | { kind: 'error'; message: string }
  | { kind: 'done'; result: AnalyzeResponse; savedAt?: string };

const STEPS = ['Checking image quality', 'Checking familiarity', 'Running 5-model ensemble', 'Generating heatmap'];

export default function Analyze() {
  const params = useLocalSearchParams<Params>();
  const isSample = Boolean(params.sampleId);
  const [uri, setUri] = useState(params.uri ?? params.sampleUrl ?? '');
  const [state, setState] = useState<State>(params.historyId ? { kind: 'loading' } : { kind: 'preview' });
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [sampleMeta, setSampleMeta] = useState({ title: params.sampleTitle, label: params.sampleLabel });

  useEffect(() => {
    if (!params.historyId) return;
    getHistoryEntry(params.historyId).then((entry) => {
      if (!entry) return setState({ kind: 'error', message: 'This history entry no longer exists.' });
      setUri(entry.imageUri ?? '');
      setSampleMeta({ title: entry.sampleTitle, label: undefined });
      setState({ kind: 'done', result: entry.result, savedAt: entry.createdAt });
    });
  }, [params.historyId]);

  const analyze = async () => {
    setState({ kind: 'loading' });
    setShowHeatmap(false);
    try {
      const result = isSample ? await analyzeSample(params.sampleId!) : await analyzeImage(uri);
      setState({ kind: 'done', result });
      if (!result.is_mock) {
        saveToHistory({ imageUri: uri, source: isSample ? 'sample' : (params.source ?? 'library'), sampleTitle: params.sampleTitle, result }).catch(() => {});
      }
    } catch (e) {
      setState({ kind: 'error', message: e instanceof Error ? e.message : 'Something went wrong.' });
    }
  };

  const retake = async () => {
    const next = await pickImage(params.source === 'camera' ? 'camera' : 'library');
    if (next) {
      setUri(next);
      setState({ kind: 'preview' });
    }
  };

  const onExport = async (result: AnalyzeResponse) => {
    setExporting(true);
    try {
      await exportReport(result, uri);
    } catch {
      Alert.alert('Export failed', 'Could not create the PDF summary. Please try again.');
    } finally {
      setExporting(false);
    }
  };

  const result = state.kind === 'done' ? state.result : null;
  const heatmapUri = result?.heatmap_jpeg_base64 ? `data:image/jpeg;base64,${result.heatmap_jpeg_base64}` : null;
  const shownUri = showHeatmap && heatmapUri ? heatmapUri : uri;

  return (
    <ScrollView className="flex-1" contentContainerClassName="gap-4 p-5 pb-12">
      {sampleMeta.title ? (
        <View className="flex-row flex-wrap items-center gap-2">
          <Tag label="Sample case" color="sky" />
          {sampleMeta.label ? <Tag label={`Ground truth: ${sampleMeta.label}`} color="slate" /> : null}
        </View>
      ) : null}
      {state.kind === 'done' && state.savedAt ? <Text className="text-xs text-slate-500">Saved {new Date(state.savedAt).toLocaleString()}</Text> : null}

      <View className="overflow-hidden rounded-2xl bg-slate-200">
        {shownUri ? (
          <Image source={{ uri: shownUri }} className="aspect-square w-full" resizeMode="cover" accessibilityLabel={showHeatmap ? 'Grad-CAM heatmap overlay' : 'Oral image'} />
        ) : (
          <View className="aspect-square w-full items-center justify-center">
            <Text className="text-slate-500">Image not stored</Text>
          </View>
        )}
        {state.kind === 'loading' ? (
          <View className="absolute inset-0 items-center justify-center bg-slate-900/40">
            <LoadingSteps />
          </View>
        ) : null}
      </View>

      {result?.heatmap_available ? (
        <View className="gap-2">
          <View className="flex-row rounded-xl bg-slate-200 p-1">
            {(['Photo', 'Heatmap'] as const).map((tab) => {
              const active = (tab === 'Heatmap') === showHeatmap;
              return (
                <Pressable key={tab} onPress={() => setShowHeatmap(tab === 'Heatmap')} className={`flex-1 items-center rounded-lg py-2 ${active ? 'bg-white' : ''}`}>
                  <Text className={active ? 'font-semibold text-slate-900' : 'text-slate-600'}>{tab}</Text>
                </Pressable>
              );
            })}
          </View>
          {showHeatmap ? (
            <Text className="text-xs leading-4 text-slate-500">
              Red areas influenced the model most toward &ldquo;
              {result.predicted_label ?? (result.positive_probability >= result.operating_threshold ? 'malignant' : 'benign')}&rdquo;. Shows model attention, not a lesion
              boundary.
            </Text>
          ) : null}
        </View>
      ) : null}

      {state.kind === 'preview' ? (
        <View className="gap-3">
          <Button title="Analyse image" onPress={analyze} />
          {isSample ? (
            <Button title="Back to samples" variant="secondary" onPress={() => router.back()} />
          ) : (
            <Button title={params.source === 'camera' ? 'Retake photo' : 'Choose a different image'} variant="secondary" onPress={retake} />
          )}
        </View>
      ) : null}

      {state.kind === 'error' ? (
        <View className="gap-3">
          <Banner tone="danger" title="Analysis failed" lines={[state.message]} />
          {!params.historyId ? <Button title="Try again" onPress={analyze} /> : null}
          {!isSample && !params.historyId ? <Button title="Choose a different image" variant="secondary" onPress={retake} /> : null}
        </View>
      ) : null}

      {result ? <ResultView result={result} /> : null}

      {result ? (
        <View className="gap-3">
          <Button title="Export PDF summary" onPress={() => onExport(result)} loading={exporting} />
          <Button title="Analyse another image" variant="secondary" onPress={() => router.dismissTo('/')} />
        </View>
      ) : null}
    </ScrollView>
  );
}

function LoadingSteps() {
  const [step, setStep] = useState(0);
  const timer = useRef<ReturnType<typeof setInterval>>(undefined);
  useEffect(() => {
    timer.current = setInterval(() => setStep((s) => Math.min(s + 1, STEPS.length - 1)), 450);
    return () => clearInterval(timer.current);
  }, []);
  return (
    <View className="items-center gap-3 rounded-2xl bg-white/95 px-6 py-5">
      <ActivityIndicator color="#0f766e" />
      <Text className="font-medium text-slate-800">{STEPS[step]}…</Text>
    </View>
  );
}
