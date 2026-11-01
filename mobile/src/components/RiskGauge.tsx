import { useState } from 'react';
import { Text, View } from 'react-native';
import Svg, { Circle, Defs, Line, LinearGradient, Rect, Stop, Text as SvgText } from 'react-native-svg';

interface Props {
  probability: number; // ensemble mean P(malignant)
  threshold: number; // operating (referral) threshold
  margin: number; // abstention band half-width around the threshold
  members: number[]; // individual ensemble members
}

// Visualises the score, the referral threshold, the "uncertain" band and how much the ensemble agrees
export function RiskGauge({ probability, threshold, margin, members }: Props) {
  const [width, setWidth] = useState(0);
  const pad = 12;
  const w = Math.max(width - pad * 2, 1);
  const x = (p: number) => pad + Math.max(0, Math.min(1, p)) * w;
  const barY = 30;
  const barH = 14;

  return (
    <View onLayout={(e) => setWidth(e.nativeEvent.layout.width)}>
      {width > 0 ? (
        <Svg width={width} height={86}>
          <Defs>
            <LinearGradient id="risk" x1="0" y1="0" x2="1" y2="0">
              <Stop offset="0" stopColor="#10b981" />
              <Stop offset={String(threshold)} stopColor="#fbbf24" />
              <Stop offset="1" stopColor="#dc2626" />
            </LinearGradient>
          </Defs>
          <Rect x={pad} y={barY} width={w} height={barH} rx={7} fill="url(#risk)" opacity={0.85} />
          {margin > 0 ? (
            <Rect x={x(threshold - margin)} y={barY - 4} width={x(threshold + margin) - x(threshold - margin)} height={barH + 8} fill="#64748b" opacity={0.18} rx={3} />
          ) : null}
          <Line x1={x(threshold)} x2={x(threshold)} y1={barY - 8} y2={barY + barH + 8} stroke="#0f172a" strokeWidth={2} strokeDasharray="3 2" />
          <SvgText x={x(threshold)} y={barY - 12} fontSize={10} fill="#0f172a" textAnchor="middle">
            referral threshold {Math.round(threshold * 100)}%
          </SvgText>
          {members.map((m, i) => (
            <Circle key={i} cx={x(m)} cy={barY + barH + 16} r={4} fill="#334155" opacity={0.55} />
          ))}
          <Circle cx={x(probability)} cy={barY + barH / 2} r={9} fill="#fff" stroke="#0f172a" strokeWidth={3} />
          <SvgText x={pad} y={82} fontSize={9} fill="#64748b">
            0% benign-like
          </SvgText>
          <SvgText x={pad + w} y={82} fontSize={9} fill="#64748b" textAnchor="end">
            100% malignant-like
          </SvgText>
        </Svg>
      ) : (
        <View style={{ height: 86 }} />
      )}
      <Text className="mt-1 text-xs text-slate-500">
        ○ ensemble score · ● individual models · shaded band = too close to call
      </Text>
    </View>
  );
}
