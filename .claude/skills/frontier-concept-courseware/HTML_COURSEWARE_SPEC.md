# HTML Courseware Output Spec

Create self-contained educational HTML pages that can be opened locally in a browser.

## Required sections
1. Hero: topic title, one-sentence intuition, estimated time
2. Mental model: beginner-friendly analogy
3. Mechanism: step-by-step technical explanation
4. Interactive visualization: D3.js or plain JS if D3 is unavailable
5. Frontier relevance: why labs care
6. Mini exercise: 3 tasks
7. Quiz: 5 questions
8. Research log prompt

## Visual style
- Clean single-page HTML
- Responsive layout
- Use cards/sections
- Avoid distracting animation
- Use accessible contrast
- Keep labels large and readable

## D3.js usage
- Prefer local D3 if available: `./vendor/d3.v7.min.js`
- If not, use CDN: `https://cdn.jsdelivr.net/npm/d3@7`
- If neither works, provide a fallback static SVG or simple DOM visualization.

## Output contract
Save to:

```text
courseware/<topic-slug>/index.html
courseware/<topic-slug>/README.md
```

The README should explain:
- what the page teaches
- how to open it
- what interaction to try
- what file to modify next
