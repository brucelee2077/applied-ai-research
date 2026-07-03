# D3 Visualization Patterns for Frontier Lab Learning

## Attention heatmap
Data shape:
```js
const tokens = ["The", "cat", "sat"];
const weights = [
  [0.7, 0.2, 0.1],
  [0.1, 0.8, 0.1],
  [0.2, 0.3, 0.5]
];
```
Visual: square matrix, row token attends to column token.
Interaction: hover cell to show explanation.

## KV cache timeline
Data shape:
```js
const steps = [
  {step: "prefill", tokens: 8, cached: 8},
  {step: "decode 1", tokens: 1, cached: 9},
  {step: "decode 2", tokens: 1, cached: 10}
];
```
Visual: growing blocks across decode steps.
Interaction: slider for sequence length.

## Scaling law curve
Data shape:
```js
const points = [
  {compute: 1e18, loss: 3.2},
  {compute: 1e19, loss: 2.6},
  {compute: 1e20, loss: 2.1}
];
```
Visual: log-log line/scatter.
Interaction: toggle undertrained vs compute-optimal.

## MoE routing
Data shape:
```js
const tokens = [{id:0, text:"cat", expert:1}, ...];
const experts = [{id:0, load:3}, {id:1, load:5}];
```
Visual: tokens flow to expert boxes.
Interaction: capacity slider.
