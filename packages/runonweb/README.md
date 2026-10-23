# runonweb

Run AI locally in the browser with one import. No API keys, backend, or file uploads.

## Install

```bash
npm install runonweb
```

## Quick start

```ts
import { removeBackground } from 'runonweb/remove-bg'

const png = await removeBackground(imageFile)
const url = URL.createObjectURL(png)
```

`imageFile` can be a `File` from an `<input type="file">`. The first call downloads the model; the browser caches it for later use.

For repeated operations, reuse an instance and call `dispose()` when finished:

```ts
import { RemoveBackground } from 'runonweb/remove-bg'

const remover = new RemoveBackground({
  onProgress: ({ status, progress }) => console.log(status, progress),
})

const first = await remover.remove(firstFile)
const second = await remover.remove(secondFile)

remover.dispose()
```

## More examples

### Speech to text

```ts
import { transcribe } from 'runonweb/stt'

const { text } = await transcribe(audioFile, {
  onPartial: (partial) => console.log(partial),
})
```

### Image captioning

```ts
import { caption } from 'runonweb/caption'

const { text } = await caption(imageFile, {
  language: 'es',
  detail: 'short', // 'short' | 'detailed' | 'more'
})
```

> **License:** Image captioning uses the LFM Open License v1.0. It is free for individuals and companies under USD 10M annual revenue; larger companies need a [commercial license](https://huggingface.co/LiquidAI/LFM2.5-VL-450M/blob/main/LICENSE).

### Depth estimation

```ts
import { estimateDepth } from 'runonweb/depth'

const { depth } = await estimateDepth(imageFile)
const url = URL.createObjectURL(depth) // Grayscale PNG: brighter means closer.
```

### Object detection

```ts
import { detect } from 'runonweb/detect'

const objects = await detect(imageFile, { threshold: 0.5 })
// [{ label: 'cat', score: 0.98, box: { xmin, ymin, xmax, ymax } }]
```

### OCR

```ts
import { read } from 'runonweb/ocr'

const { text, lines } = await read(imageFile, {
  size: 'small', // 'tiny' | 'small' | 'medium'
})
```

### Text embeddings

```ts
import { embed, cosineSimilarity } from 'runonweb/embed'

const { embeddings } = await embed(['Hello world', 'Hola mundo'])
const similarity = cosineSimilarity(embeddings[0], embeddings[1])
```

### Text classification

```ts
import { classify } from 'runonweb/classify'

const { answers } = await classify({
  state: 'Shoes arrived two weeks late and in the wrong size.',
  questions: {
    department: {
      type: 'choice',
      instructions: 'Which team should handle this?',
      criteria: { returns: null, shipping: null, billing: null },
    },
    escalate: {
      type: 'noul',
      instructions: 'Does this need urgent human attention?',
    },
  },
})

console.log(answers.department.choice)
console.log(answers.escalate.noul) // Probability of "yes".
```

### Translation

```ts
import { translate } from 'runonweb/translate'

const { text } = await translate('Hello world', {
  from: 'en',
  to: 'es',
})
```

### Transcript cleanup

```ts
import { clean } from 'runonweb/clean'

const { text } = await clean(
  'so um i need to like send the report by uh friday',
  { styling: 'semi-formal', structure: 'prose', context: 'general' },
)
```

### Text to emoji

```ts
import { emojify } from 'runonweb/emoji'

const { text, emojis } = await emojify('I love pizza and my dog')
// text: '🍕❤️🐶'
```

### Text to speech

```ts
import { TextToSpeech } from 'runonweb/tts'

const tts = new TextToSpeech({ voice: 'af_heart' })
const wav = await tts.speakToBlob('Hello from the browser')
const url = URL.createObjectURL(wav)

tts.dispose()
```

For English, Korean, Spanish, Portuguese, or French, use `new TextToSpeech({ size: 'multi', voice: 'st_f1', language: 'es' })`.

### Text to image

Requires WebGPU.

```ts
import { generateImage } from 'runonweb/image'

const { image } = await generateImage('A bonsai tree in a ceramic studio')
const url = URL.createObjectURL(image) // PNG Blob.
```

[API documentation and live demos](https://runonweb.ai/docs) · [Model details and licenses](https://runonweb.ai/models)
