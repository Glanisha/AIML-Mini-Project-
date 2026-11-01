import { useEffect, useState } from 'react';
import { Text, View } from 'react-native';

import { getHealth } from '@/lib/api';
import type { HealthResponse } from '@/lib/types';

type State = { kind: 'checking' } | { kind: 'up'; health: HealthResponse } | { kind: 'down' };

export function ServerStatus() {
  const [state, setState] = useState<State>({ kind: 'checking' });

  useEffect(() => {
    getHealth()
      .then((health) => setState({ kind: 'up', health }))
      .catch(() => setState({ kind: 'down' }));
  }, []);

  const [dot, text] =
    state.kind === 'up'
      ? ['bg-emerald-500', `Server online · model ${state.health.model_version}`]
      : state.kind === 'down'
        ? ['bg-red-500', 'Server unreachable']
        : ['bg-slate-400', 'Checking server…'];

  return (
    <View className="flex-row items-center justify-center gap-2">
      <View className={`h-2 w-2 rounded-full ${dot}`} />
      <Text className="text-xs text-slate-500">{text}</Text>
    </View>
  );
}
