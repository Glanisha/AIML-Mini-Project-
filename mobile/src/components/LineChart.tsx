import { useState } from 'react';
import { Text, View } from 'react-native';
import Svg, { Circle, G, Line, Polyline, Rect, Text as SvgText } from 'react-native-svg';

export interface Series {
  label: string;
  color: string;
  points: [number, number][];
  dashed?: boolean;
  dots?: boolean;
}

interface Props {
  series: Series[];
  xLabel: string;
  yLabel: string;
  xDomain?: [number, number];
  yDomain?: [number, number];
  diagonal?: boolean; // y = x reference (ROC chance line, perfect calibration)
  markers?: { x: number; y: number; color: string; label: string }[];
  verticalLine?: { x: number; color: string; label: string };
  height?: number;
}

const PAD = { left: 38, right: 10, top: 10, bottom: 34 };

export function LineChart({ series, xLabel, yLabel, xDomain = [0, 1], yDomain = [0, 1], diagonal, markers = [], verticalLine, height = 220 }: Props) {
  const [width, setWidth] = useState(0);
  const w = Math.max(width - PAD.left - PAD.right, 1);
  const h = height - PAD.top - PAD.bottom;
  const sx = (x: number) => PAD.left + ((x - xDomain[0]) / (xDomain[1] - xDomain[0])) * w;
  const sy = (y: number) => PAD.top + h - ((y - yDomain[0]) / (yDomain[1] - yDomain[0])) * h;
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <View onLayout={(e) => setWidth(e.nativeEvent.layout.width)}>
      {width > 0 ? (
        <Svg width={width} height={height}>
          <Rect x={PAD.left} y={PAD.top} width={w} height={h} fill="#f8fafc" />
          {ticks.map((t) => {
            const xv = xDomain[0] + t * (xDomain[1] - xDomain[0]);
            const yv = yDomain[0] + t * (yDomain[1] - yDomain[0]);
            return (
              <G key={t}>
                <Line x1={PAD.left} x2={PAD.left + w} y1={sy(yv)} y2={sy(yv)} stroke="#e2e8f0" strokeWidth={1} />
                <SvgText x={PAD.left - 5} y={sy(yv) + 3} fontSize={9} fill="#64748b" textAnchor="end">
                  {fmt(yv)}
                </SvgText>
                <SvgText x={sx(xv)} y={PAD.top + h + 12} fontSize={9} fill="#64748b" textAnchor="middle">
                  {fmt(xv)}
                </SvgText>
              </G>
            );
          })}
          {diagonal ? <Line x1={sx(xDomain[0])} y1={sy(yDomain[0])} x2={sx(xDomain[1])} y2={sy(yDomain[1])} stroke="#cbd5e1" strokeDasharray="4 4" /> : null}
          {verticalLine ? (
            <Line x1={sx(verticalLine.x)} x2={sx(verticalLine.x)} y1={PAD.top} y2={PAD.top + h} stroke={verticalLine.color} strokeDasharray="5 3" strokeWidth={1.5} />
          ) : null}
          {series.map((s) => (
            <G key={s.label}>
              <Polyline
                points={s.points.map(([x, y]) => `${sx(x)},${sy(Math.max(yDomain[0], Math.min(yDomain[1], y)))}`).join(' ')}
                fill="none"
                stroke={s.color}
                strokeWidth={2}
                strokeDasharray={s.dashed ? '5 4' : undefined}
              />
              {s.dots ? s.points.map(([x, y], i) => <Circle key={i} cx={sx(x)} cy={sy(y)} r={3} fill={s.color} />) : null}
            </G>
          ))}
          {markers.map((m) => (
            <Circle key={m.label} cx={sx(m.x)} cy={sy(m.y)} r={5} fill={m.color} stroke="#fff" strokeWidth={1.5} />
          ))}
          <SvgText x={PAD.left + w / 2} y={height - 4} fontSize={10} fill="#334155" textAnchor="middle">
            {xLabel}
          </SvgText>
          <SvgText x={10} y={PAD.top + h / 2} fontSize={10} fill="#334155" textAnchor="middle" rotation={-90} originX={10} originY={PAD.top + h / 2}>
            {yLabel}
          </SvgText>
        </Svg>
      ) : (
        <View style={{ height }} />
      )}
      <View className="mt-1 flex-row flex-wrap gap-x-4 gap-y-1">
        {[...series, ...markers.map((m) => ({ label: m.label, color: m.color, dashed: false })), ...(verticalLine ? [{ ...verticalLine, dashed: true }] : [])].map((s) => (
          <View key={s.label} className="flex-row items-center gap-1.5">
            <View style={{ width: 14, height: 3, backgroundColor: s.color, opacity: s.dashed ? 0.6 : 1 }} />
            <Text className="text-xs text-slate-600">{s.label}</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

const fmt = (v: number) => (Math.abs(v) < 1 && v !== 0 ? v.toFixed(2).replace(/^0/, '').replace(/^-0/, '-') : String(Math.round(v * 100) / 100));
