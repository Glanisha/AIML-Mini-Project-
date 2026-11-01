import { useEffect, useState } from 'react';
import { ScrollView, Text, View } from 'react-native';

import { Card } from '@/components/ui';
import { getModelCard } from '@/lib/api';
import type { ModelCard } from '@/lib/types';

const SECTIONS: [string, string[]][] = [
  [
    'Intended use',
    [
      'Screening support for photographs of visible oral lesions.',
      'Helping a user or health worker decide whether a lesion warrants prompt professional examination.',
    ],
  ],
  [
    'Not intended for',
    [
      'Diagnosis, cancer staging, treatment decisions or replacing a clinical examination and biopsy.',
      'Healthy mouths, teeth, or non-oral images: the model only learned benign vs malignant lesions.',
      'Pigmented (brown or black) lesions such as oral melanoma: the training data contains essentially none, so the model does not recognise them.',
      'Use on populations or cameras very different from the training data without re-validation.',
    ],
  ],
  [
    'How it works',
    [
      'Quality gate: sharpness and exposure are compared with the distribution of training photos.',
      'Familiarity check: an image is rejected if even its 5th-closest training photo is far away in a general-purpose (ImageNet) feature space (deep k-nearest-neighbour detection).',
      'Prediction: five EfficientNet-B0 models (one per cross-validation fold) are averaged, with temperature scaling for calibrated scores.',
      'Abstention: no answer when the score is too close to the referral threshold or the models disagree.',
      'Explanation: class activation maps (exact Grad-CAM for this architecture), averaged across the ensemble.',
    ],
  ],
  [
    'Safety & ethics',
    [
      'The referral threshold favours sensitivity: missing a malignancy is worse than an unnecessary check-up.',
      'Uploaded photos are processed in memory and never stored on the server. History stays on the device.',
      'No accounts, tracking or third-party analytics.',
    ],
  ],
  [
    'Known limitations',
    [
      'Small, single-region dataset (323 images from Karnataka, India) with no patient IDs or demographics.',
      'Balanced classes (~50% malignant) overstate positive predictive value compared with real screening.',
      'No external validation yet; performance on other populations and devices is unknown.',
    ],
  ],
];

export default function About() {
  const [card, setCard] = useState<ModelCard | null>(null);
  useEffect(() => {
    getModelCard()
      .then(setCard)
      .catch(() => setCard(null));
  }, []);

  return (
    <ScrollView className="flex-1" contentContainerClassName="gap-4 p-5 pb-12">
      {card ? (
        <Card title="Deployed model">
          {(
            [
              ['Version', card.version],
              ['Architecture', `${card.architecture} × ${card.ensemble_size} (deep ensemble)`],
              ['Calibration', `${card.calibration.method}, T = ${card.calibration.temperature.toFixed(2)}`],
              ['Referral threshold', `${Math.round(card.decision.operating_threshold * 100)}% (target sensitivity ${Math.round(card.decision.target_sensitivity * 100)}%)`],
              ['Built', new Date(card.created).toLocaleDateString()],
            ] as [string, string][]
          ).map(([k, v]) => (
            <View key={k} className="flex-row justify-between gap-4 py-1">
              <Text className="text-xs text-slate-500">{k}</Text>
              <Text className="flex-1 text-right text-xs text-slate-800">{v}</Text>
            </View>
          ))}
        </Card>
      ) : null}

      {SECTIONS.map(([title, items]) => (
        <Card key={title} title={title}>
          {items.map((i) => (
            <Text key={i} className="mb-1.5 text-sm leading-5 text-slate-700">
              • {i}
            </Text>
          ))}
        </Card>
      ))}

      <Card title="Data credit">
        <Text className="text-xs leading-5 text-slate-600">
          {card?.training_data ??
            'Chandrashekar H S, Geetha Kiran A, Murali S, Dinesh M S, Nanditha B R (2021). Oral Images Dataset, Mendeley Data, V2. doi:10.17632/mhjyrn35p4.2. CC BY 4.0.'}
        </Text>
      </Card>
    </ScrollView>
  );
}
