import { ActivityIndicator, Pressable, Text } from 'react-native';

type Variant = 'primary' | 'secondary' | 'ghost';

const containerStyles: Record<Variant, string> = {
  primary: 'bg-teal-700 active:bg-teal-800',
  secondary: 'bg-white border border-slate-300 active:bg-slate-100',
  ghost: 'active:bg-slate-100',
};

const textStyles: Record<Variant, string> = {
  primary: 'text-white',
  secondary: 'text-slate-800',
  ghost: 'text-teal-700',
};

interface Props {
  title: string;
  onPress: () => void;
  variant?: Variant;
  disabled?: boolean;
  loading?: boolean;
}

export function Button({ title, onPress, variant = 'primary', disabled, loading }: Props) {
  const inactive = disabled || loading;
  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      disabled={inactive}
      className={`flex-row items-center justify-center rounded-xl px-5 py-4 ${containerStyles[variant]} ${inactive ? 'opacity-50' : ''}`}
    >
      {loading ? (
        <ActivityIndicator color={variant === 'primary' ? '#fff' : '#0f766e'} />
      ) : (
        <Text className={`text-base font-semibold ${textStyles[variant]}`}>{title}</Text>
      )}
    </Pressable>
  );
}
