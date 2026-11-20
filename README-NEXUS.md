# Agent-Five-Nexus-1.0.0

Nexus is the v9 continuation of the Agent Five embodied-agent prototype. The implementation keeps the existing fixed-step 2D world and actor/critic learning pipeline, adds a 24-feature observation space and five-channel policy, and introduces a pure-JavaScript language subsystem plus deterministic environment tools.

## Files

- `agent-five-nexus.js` — host-agnostic runtime, v5-v8 migration, 2D simulation, chat/tool loop, memory, PPO orchestration, and frozen public API.
- `agent-learning.js` — dependency-free action sampling, path planning, and PPO updates.
- `nexus-language.js` — serializable tokenizer and compact Float32Array GRU decoder.
- `learning-worker.js` — optional PPO worker; the runtime retains its main-thread fallback.
- `nexus-headless-demo.js` — headless migration/simulation/chat/reward demonstration.
- `PROMPT.txt` — original implementation specification.

## Run on GitHub Pages

The repository includes a complete static browser application at `index.html`. A visitor opens the published GitHub Pages URL and the app starts automatically: no terminal, command line, package installation, build step, account, API key, or user configuration is required. The page loads its JavaScript files directly from the same repository and runs the simulation and learning loop in the browser.

Agent Five's canonical on-screen form is a **solid red circle** moving around a 2D plane. The goal, obstacle, movement trail, chat, and learning telemetry are environment/UI elements; they do not change the agent's canonical shape.

A Pages deployment workflow is included at `.github/workflows/deploy-pages.yml`. The repository owner must enable GitHub Pages once in **Settings → Pages** and choose **GitHub Actions** as the source; that is a one-time repository publishing setting, not a setup step for visitors. Pushes to `agent-five-nexus-1.0.0` then trigger deployment. The published URL appears in the workflow's `github-pages` deployment environment.

The app uses browser `localStorage` for persistence. Model data stays in that browser unless explicitly exported. The core simulation does not require a backend or API key.

## Browser integration

Load the scripts in this order:

```html
<script src="agent-learning.js"></script>
<script src="nexus-language.js"></script>
<script src="agent-five-nexus.js"></script>
```

Create an instance with:

```js
const agent = AgentFiveNexusCore.create({
  world: { width: 500, height: 500 },
  ui: {
    onChatMessage(message) { console.log(message); },
    onTokenStream(token) { console.log(token); }
  },
  createWorker(url) { return new Worker(url); }
});

agent.start();
agent.sendMessage("scan obstacle distance");
```

The factory accepts injected storage, clock, frame scheduling, worker construction, world dimensions, learning services, and UI callbacks. No DOM access is required by the runtime.

## Public API

The frozen API includes `modelName`, `version`, `start`, `stop`, `sendMessage`, `getSnapshot`, `setPopulationSize`, `setEvolutionMode`, `setAgentRadius`, `reward`, `correct`, `clearMemory`, `reset`, `exportData`, `importData`, and the environment tools `inspect_environment`, `set_waypoint`, `estimate_path_risk`, `calculate`, and `query_memory`.

## Headless demonstration

With Node.js installed, run:

```sh
node nexus-headless-demo.js
```

The demo creates a synthetic v8 actor/critic save in isolated storage, verifies migration to the 24-input/5-output v9 actor, advances the fixed-step simulation, sends an environment-inspection request, observes chat/token callbacks, applies reward feedback, and prints a compact report.

## Migration and persistence

Nexus writes active data under:

- `agent5-nexus-1.0.0-brain`
- `agent5-nexus-1.0.0-memory`
- `agent5-nexus-1.0.0-episodes`
- `agent5-nexus-1.0.0-tokenizer`

The brain loader checks legacy v8, v7, v6, and v5 actor/critic saves. The v8 migration copies the existing spatial sensor and policy pathways, zero-initializes the eight new input rows and two new actor heads, and preserves the deep hidden layers. Legacy memory and episode inputs are normalized to 24 features and older three-channel action targets are padded to five channels.

## Current implementation note

The GRU decoder is initialized from scratch and learns its output projection from interaction sequences; it is not a pretrained general-purpose language model. The first chat responses are deliberately deterministic templates grounded in current simulation telemetry, while the decoder is exposed through `generateText` for experiments and continued training. Validate behavior and migration against real v5-v8 exports before treating this branch as a production release.
