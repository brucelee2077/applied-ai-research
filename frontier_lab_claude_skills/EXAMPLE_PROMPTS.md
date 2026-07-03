# Example Prompts

## Start a day
```text
/frontier-session-coach topic="KV cache" duration="3 hours" output="sessions/day-001-kv-cache.md"
```

## Build concept courseware
```text
/frontier-concept-courseware topic="scaled dot-product attention" output="courseware/attention/index.html"
```

## Turn a paper into courseware
```text
/frontier-paper-course source="Attention Is All You Need" focus="attention mechanism" output="notes/papers/attention-is-all-you-need.md"
```

## Create a D3 visualization
```text
/frontier-d3-visual-lab topic="MoE expert routing" interaction="capacity slider and token routing animation"
```

## Build an experiment
```text
/frontier-experiment-lab topic="toy scaling law" language="python" goal="fit a power law to synthetic compute/loss data"
```

## Review a session
```text
/frontier-review-quiz input="sessions/day-001-kv-cache.md and courseware/kv-cache/index.html"
```

## Package the week
```text
/frontier-portfolio-packager week="1" theme="attention and KV cache" output="portfolio/week-01.md"
```
