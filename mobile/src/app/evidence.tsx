import { useCallback, useEffect, useState } from 'react';
import { ActivityIndicator, ScrollView, Text, View } from 'react-native';

import { Banner } from '@/components/Banner';
import { Button } from '@/components/Button';
import { Card } from '@/components/Card';
import { ScoreBar } from '@/components/ScoreBar';
import { getMetrics } from '@/lib/api';
import type { MetricsResponse, ModelMetrics } from '@/lib/types';

const METRIC_LABELS: [keyof Omit<ModelMetrics, 'name'>, string][] = [
  ['malignant_recall', 'Malignant recall'],
  ['specificity', 'Specificity'],
  ['f1', 'F1 score'],
  ['roc_auc', 'ROC-AUC'],
  ['accuracy', 'Accuracy'],
];

export default function Evidence() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = useCallback(() => {
    getMetrics()
      .then(setMetrics)
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(fetchMetrics, [fetchMetrics]);

  const retry = () => {
    setError(null);
    fetchMetrics();
  };

  if (error) {
    return (
      <View className="flex-1 gap-3 p-5">
        <Banner tone="danger" title="Could not load metrics" lines={[error]} />
        <Button title="Retry" onPress={retry} />
      </View>
    );
  }
  if (!metrics) {
    return (
      <View className="flex-1 items-center justify-center">
        <ActivityIndicator color="#0f766e" />
      </View>
    );
  }

  const { labels, matrix } = metrics.confusion_matrix;

  return (
    <ScrollView className="flex-1" contentContainerClassName="gap-4 p-5 pb-10">
      {metrics.is_mock ? (
        <Banner tone="info" title="Placeholder metrics" lines={['These are not real results. They will be replaced by the saved evaluation output.']} />
      ) : null}

      <Card title="Evaluation setup">
        <Text className="text-slate-700">Dataset: {metrics.dataset}</Text>
        <Text className="text-slate-700">Held-out test images: {metrics.test_size}</Text>
        <Text className="text-slate-700">Model version: {metrics.model_version}</Text>
        {metrics.cpu_latency_ms != null ? (
          <Text className="text-slate-700">CPU latency: {Math.round(metrics.cpu_latency_ms)} ms</Text>
        ) : null}
      </Card>

      {metrics.models.map((m) => (
        <Card key={m.name} title={m.name}>
          {METRIC_LABELS.map(([key, label]) => (
            <ScoreBar key={key} label={label} value={m[key]} highlight={key === 'malignant_recall'} />
          ))}
        </Card>
      ))}

      <Card title="Confusion matrix (test set)">
        <Text className="mb-2 text-xs text-slate-500">Rows: true class · Columns: predicted class</Text>
        <View className="flex-row">
          <View className="flex-1" />
          {labels.map((l) => (
            <Text key={l} className="flex-1 text-center text-xs font-semibold capitalize text-slate-600">
              {l}
            </Text>
          ))}
        </View>
        {matrix.map((row, i) => (
          <View key={labels[i]} className="mt-1 flex-row items-center">
            <Text className="flex-1 text-xs font-semibold capitalize text-slate-600">{labels[i]}</Text>
            {row.map((count, j) => (
              <View key={j} className={`mx-0.5 flex-1 items-center rounded-md py-3 ${i === j ? 'bg-teal-100' : 'bg-red-50'}`}>
                <Text className="font-semibold text-slate-900">{count}</Text>
              </View>
            ))}
          </View>
        ))}
      </Card>

      {metrics.notes.length ? <Banner tone="warning" title="Limitations" lines={metrics.notes} /> : null}
    </ScrollView>
  );
}
