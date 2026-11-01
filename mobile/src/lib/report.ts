import { File } from 'expo-file-system';
import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';
import { Platform } from 'react-native';

import type { AnalyzeResponse } from './types';

const HTML_ESCAPES: Record<string, string> = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
const escape = (s: string) => s.replace(/[&<>"']/g, (c) => HTML_ESCAPES[c]);

function buildHtml(result: AnalyzeResponse, imageBase64: string | null): string {
  const scores = Object.entries(result.class_scores)
    .sort(([, a], [, b]) => b - a)
    .map(([label, score]) => `<tr><td>${escape(label)}</td><td>${(score * 100).toFixed(1)}%</td></tr>`)
    .join('');

  const outcome = result.abstained
    ? 'Uncertain: the model abstained from making a prediction.'
    : escape(result.predicted_label ?? 'n/a');

  const quality = result.quality.reasons.length
    ? `${escape(result.quality.status)}: ${result.quality.reasons.map(escape).join('; ')}`
    : escape(result.quality.status);

  // iOS printing cannot load local file URIs, so images are inlined as base64
  const image = imageBase64 ? `<img src="data:image/jpeg;base64,${imageBase64}" />` : '';
  const heatmap = result.heatmap_png_base64
    ? `<h2>Model attention (Grad-CAM)</h2><img src="data:image/png;base64,${result.heatmap_png_base64}" />
       <p class="muted">Heatmaps show where the model focused, not verified lesion boundaries.</p>`
    : '';

  return `<!doctype html><html><head><meta charset="utf-8" />
<style>
  body { font-family: -apple-system, Roboto, sans-serif; color: #0f172a; padding: 24px; }
  h1 { color: #0f766e; margin-bottom: 4px; } h2 { font-size: 16px; margin-top: 24px; }
  img { max-width: 320px; border-radius: 8px; }
  table { border-collapse: collapse; } td { padding: 4px 16px 4px 0; text-transform: capitalize; }
  .muted { color: #64748b; font-size: 12px; }
  .warn { background: #fef3c7; padding: 8px 12px; border-radius: 6px; font-size: 12px; }
</style></head><body>
  <h1>OralSight screening summary</h1>
  <p class="muted">Generated ${escape(new Date().toLocaleString())} · model ${escape(result.model_version)}${result.is_mock ? ' (DEMO / MOCK RESPONSE)' : ''}</p>
  ${image}
  <h2>Result</h2><p><strong>${outcome}</strong></p>
  <table>${scores}</table>
  <h2>Image quality</h2><p>${quality}</p>
  ${heatmap}
  <p class="warn">Academic prototype for research demonstration only. Not a medical device and not a diagnosis.
  It does not stage cancer or give treatment advice. Consult a qualified clinician.</p>
</body></html>`;
}

export async function exportReport(result: AnalyzeResponse, imageUri: string): Promise<void> {
  if (Platform.OS === 'web') {
    // Browser print dialog offers "Save as PDF"; local files cannot be shared on web
    await Print.printAsync({ html: buildHtml(result, null) });
    return;
  }

  let imageBase64: string | null = null;
  try {
    imageBase64 = await new File(imageUri).base64();
  } catch {
    // the summary is still useful without the photo
  }

  const { uri } = await Print.printToFileAsync({ html: buildHtml(result, imageBase64) });

  if (await Sharing.isAvailableAsync()) {
    await Sharing.shareAsync(uri, { mimeType: 'application/pdf', UTI: '.pdf', dialogTitle: 'Share screening summary' });
  } else {
    await Print.printAsync({ uri });
  }
}
