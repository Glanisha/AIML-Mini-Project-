import { File } from 'expo-file-system';
import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';
import { Platform } from 'react-native';

import type { AnalyzeResponse } from './types';

const HTML_ESCAPES: Record<string, string> = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
const escape = (s: string) => s.replace(/[&<>"']/g, (c) => HTML_ESCAPES[c]);
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

const TRIAGE_COLOURS = { refer: '#b91c1c', routine: '#047857', indeterminate: '#b45309' };

function buildHtml(r: AnalyzeResponse, imageBase64: string | null): string {
  const scores = Object.entries(r.class_scores)
    .map(([label, score]) => `<tr><td>P(${escape(label)})</td><td>${pct(score)}</td></tr>`)
    .join('');
  const u = r.uncertainty;
  const list = (items: string[]) => (items.length ? `<ul>${items.map((i) => `<li>${escape(i)}</li>`).join('')}</ul>` : '<p>None</p>');

  // iOS printing cannot load local file URIs, so images are inlined as base64
  const photo = imageBase64 ? `<img src="data:image/jpeg;base64,${imageBase64}" />` : '';
  const heat = r.heatmap_jpeg_base64 ? `<img src="data:image/jpeg;base64,${r.heatmap_jpeg_base64}" />` : '';

  return `<!doctype html><html><head><meta charset="utf-8" />
<style>
  body { font-family: -apple-system, Roboto, sans-serif; color: #0f172a; padding: 28px; font-size: 13px; }
  h1 { color: #0f766e; margin: 0; } h2 { font-size: 14px; margin: 22px 0 6px; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }
  .images { display: flex; gap: 12px; margin-top: 16px; } .images img { width: 48%; border-radius: 8px; }
  .triage { border: 2px solid ${TRIAGE_COLOURS[r.triage.level]}; color: ${TRIAGE_COLOURS[r.triage.level]}; border-radius: 10px; padding: 12px 14px; margin-top: 16px; }
  .triage p { color: #0f172a; margin: 6px 0 0; }
  table { border-collapse: collapse; } td { padding: 3px 18px 3px 0; }
  .muted { color: #64748b; font-size: 11px; }
  .warn { background: #fef3c7; padding: 10px 12px; border-radius: 6px; font-size: 11px; margin-top: 22px; }
</style></head><body>
  <h1>OralSight screening summary</h1>
  <p class="muted">${escape(new Date().toLocaleString())} · model ${escape(r.model_version)}${r.is_mock ? ' · DEMO / MOCK RESPONSE' : ''}</p>
  <div class="images">${photo}${heat}</div>
  ${heat ? '<p class="muted">Right: model attention heatmap (not a verified lesion boundary).</p>' : ''}
  <div class="triage"><strong>${escape(r.triage.title)}</strong><p>${escape(r.triage.message)}</p></div>
  <h2>Scores</h2>
  <table>
    ${scores}
    <tr><td>Referral threshold</td><td>${pct(r.operating_threshold)}</td></tr>
    <tr><td>Ensemble agreement</td><td>${Math.round(u.member_agreement * 100)}% of ${u.member_probabilities.length} models</td></tr>
    <tr><td>Model score range</td><td>${pct(u.score_range[0])} – ${pct(u.score_range[1])}</td></tr>
  </table>
  <h2>Reasons for abstaining</h2>${list(r.abstain_reasons)}
  <h2>Image quality: ${escape(r.quality.status)}</h2>${list(r.quality.reasons)}
  <h2>Familiarity check</h2><p>${r.ood.in_distribution ? 'Image resembles the training data.' : 'Image is unlike the training data; the result is not reliable.'} (distance ${r.ood.score.toFixed(3)}, limit ${r.ood.threshold.toFixed(3)})</p>
  <p class="warn">Research prototype. Not a medical device and not a diagnosis.
  It does not stage cancer or give treatment advice. Any oral lesion lasting more than 2 weeks should be examined by a clinician.</p>
</body></html>`;
}

async function imageAsBase64(uri: string): Promise<string | null> {
  try {
    if (uri.startsWith('http')) {
      const blob = await (await globalThis.fetch(uri)).blob();
      return await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(String(reader.result).split(',')[1] ?? null);
        reader.readAsDataURL(blob);
      });
    }
    return Platform.OS === 'web' ? null : await new File(uri).base64();
  } catch {
    return null; // the summary is still useful without the photo
  }
}

export async function exportReport(result: AnalyzeResponse, imageUri: string): Promise<void> {
  const html = buildHtml(result, imageUri ? await imageAsBase64(imageUri) : null);

  if (Platform.OS === 'web') {
    // Browser print dialog offers "Save as PDF"; local files cannot be shared on web
    await Print.printAsync({ html });
    return;
  }

  const { uri } = await Print.printToFileAsync({ html });
  if (await Sharing.isAvailableAsync()) {
    await Sharing.shareAsync(uri, { mimeType: 'application/pdf', UTI: '.pdf', dialogTitle: 'Share screening summary' });
  } else {
    await Print.printAsync({ uri });
  }
}
