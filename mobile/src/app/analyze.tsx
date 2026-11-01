import { router, useLocalSearchParams } from 'expo-router';
import { useState } from 'react';
import { Alert, Image, Pressable, ScrollView, Text, View } from 'react-native';

import { Banner } from '@/components/Banner';
import { Button } from '@/components/Button';
import { Card } from '@/components/Card';
import { ScoreBar } from '@/components/ScoreBar';
import { analyzeImage } from '@/lib/api';
import { pickImage } from '@/lib/pickImage';
import { exportReport } from '@/lib/report';
import type { AnalyzeResponse } from '@/lib/types';

type State =
  | { kind: 'preview' }
  | { kind: 'loading' }
  | { kind: 'error'; message: string }
  | { kind: 'done'; result: AnalyzeResponse };

const formatLabel = (label: string) => label.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase());

export default function Analyze() {
  const params = useLocalSearchParams<{ uri: string }>();
  const [uri, setUri] = useState(params.uri);
  const [state, setState] = useState<State>({ kind: 'preview' });
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [exporting, setExporting] = useState(false);

  const analyze = async () => {
    setState({ kind: 'loading' });
    setShowHeatmap(false);
    try {
      setState({ kind: 'done', result: await analyzeImage(uri) });
    } catch (e) {
      setState({ kind: 'error', message: e instanceof Error ? e.message : 'Something went wrong.' });
    }
  };

  const retake = async () => {
    const next = await pickImage('library');
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
  const heatmapUri = result?.heatmap_png_base64 ? `data:image/png;base64,${result.heatmap_png_base64}` : null;

  return (
    <ScrollView className="flex-1" contentContainerClassName="gap-4 p-5 pb-10">
      <View className="overflow-hidden rounded-2xl bg-slate-200">
        <Image
          source={{ uri: showHeatmap && heatmapUri ? heatmapUri : uri }}
          className="aspect-square w-full"
          resizeMode="cover"
          accessibilityLabel={showHeatmap ? 'Grad-CAM heatmap overlay' : 'Selected oral image'}
        />
      </View>

      {result?.heatmap_available ? (
        <View className="flex-row rounded-xl bg-slate-200 p-1">
          {(['Original', 'Heatmap'] as const).map((tab) => {
            const active = (tab === 'Heatmap') === showHeatmap;
            return (
              <Pressable
                key={tab}
                onPress={() => setShowHeatmap(tab === 'Heatmap')}
                className={`flex-1 items-center rounded-lg py-2 ${active ? 'bg-white' : ''}`}
              >
                <Text className={active ? 'font-semibold text-slate-900' : 'text-slate-600'}>{tab}</Text>
              </Pressable>
            );
          })}
        </View>
      ) : null}
      {showHeatmap ? (
        <Text className="text-xs text-slate-500">
          The heatmap shows where the model focused. It is not a verified lesion boundary.
        </Text>
      ) : null}

      {state.kind === 'preview' || state.kind === 'loading' ? (
        <View className="gap-3">
          <Button title="Analyse image" onPress={analyze} loading={state.kind === 'loading'} />
          <Button title="Choose a different image" variant="secondary" onPress={retake} disabled={state.kind === 'loading'} />
        </View>
      ) : null}

      {state.kind === 'error' ? (
        <View className="gap-3">
          <Banner tone="danger" title="Analysis failed" lines={[state.message]} />
          <Button title="Try again" onPress={analyze} />
          <Button title="Choose a different image" variant="secondary" onPress={retake} />
        </View>
      ) : null}

      {result ? <ResultView result={result} /> : null}

      {result ? (
        <View className="gap-3">
          <Button title="Export PDF summary" onPress={() => onExport(result)} loading={exporting} />
          <Button title="Analyse another image" variant="secondary" onPress={() => router.back()} />
        </View>
      ) : null}
    </ScrollView>
  );
}

function ResultView({ result }: { result: AnalyzeResponse }) {
  const scores = Object.entries(result.class_scores).sort(([, a], [, b]) => b - a);
  const { quality } = result;

  return (
    <View className="gap-4">
      {result.is_mock ? (
        <Banner tone="info" title="Demo response" lines={['The real model is not connected yet. These scores are placeholders.']} />
      ) : null}

      {quality.status === 'reject' ? (
        <Banner tone="danger" title="Image quality too low" lines={[...quality.reasons, 'Retake the photo for a reliable result.']} />
      ) : quality.status === 'warning' ? (
        <Banner tone="warning" title="Image quality warning" lines={quality.reasons} />
      ) : null}

      {result.abstained ? (
        <Banner
          tone="warning"
          title="Uncertain: no prediction made"
          lines={['The model is not confident enough to classify this image. Try a clearer photo, or refer to a clinician.']}
        />
      ) : result.predicted_label ? (
        <Card>
          <Text className="text-sm text-slate-500">Model prediction</Text>
          <Text className="mt-1 text-2xl font-bold text-slate-900">{formatLabel(result.predicted_label)}</Text>
        </Card>
      ) : null}

      <Card title="Class scores">
        {scores.map(([label, score]) => (
          <ScoreBar key={label} label={label} value={score} highlight={!result.abstained && label === result.predicted_label} />
        ))}
      </Card>

      <Text className="text-center text-xs text-slate-400">
        Model {result.model_version} · {Math.round(result.latency_ms)} ms · Not a diagnosis
      </Text>
    </View>
  );
}
