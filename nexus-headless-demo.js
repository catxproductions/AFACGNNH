/* Run with: node nexus-headless-demo.js
 * Demonstrates v8 -> Nexus v9 migration, fixed-step simulation, tool use,
 * token streaming, and reinforcement feedback without a browser or API key.
 */
"use strict";

require("./agent-learning.js");
require("./nexus-language.js");
require("./agent-five-nexus.js");

const MODEL_NAME = "Agent-Five-Nexus-1.0.0";
const hidden = [
    15, 16, 15, 14, 15, 13, 14, 12, 13, 11, 12, 10, 11,
    12, 10, 9, 10, 8, 9, 7, 8, 6, 7, 5, 6, 4
];

function makeLegacyV8Model(outputSize, modelType) {
    const sizes = [16, ...hidden, outputSize];
    const weights = [];
    const biases = [];
    for (let layer = 0; layer < sizes.length - 1; layer++) {
        weights.push(Array.from({ length: sizes[layer] }, () => Array(sizes[layer + 1]).fill(0)));
        biases.push(Array(sizes[layer + 1]).fill(0));
    }
    return {
        modelName: "AgentFiveCore-v8",
        version: 8,
        modelType,
        architecture: sizes,
        weights,
        biases
    };
}

const storageData = new Map();
storageData.set("agent5-brain-v8", JSON.stringify({
    ...makeLegacyV8Model(3, "policy"),
    critic: makeLegacyV8Model(1, "critic")
}));
storageData.set("agent5-memory-v2", JSON.stringify([{
    inputs: Array(16).fill(0),
    target: [0, 0, 0],
    outcome: "goal-reward"
}]));
const storage = {
    getItem(key) { return storageData.has(key) ? storageData.get(key) : null; },
    setItem(key, value) { storageData.set(key, String(value)); },
    removeItem(key) { storageData.delete(key); }
};

let clock = 0;
const scheduled = [];
const messages = [];
const streamedTokens = [];
const agent = globalThis.AgentFiveNexusCore.create({
    world: { width: 500, height: 500 },
    storage,
    now: () => clock,
    requestFrame(callback) { scheduled.push(callback); return scheduled.length; },
    cancelFrame() {},
    ui: {
        onChatMessage(message) { messages.push(message); },
        onTokenStream(token) { streamedTokens.push(token); }
    }
});

const migrated = JSON.parse(storage.getItem("agent5-nexus-1.0.0-brain"));
if (!migrated || migrated.version !== 9 || migrated.architecture[0] !== 24 ||
    migrated.architecture[migrated.architecture.length - 1] !== 5) {
    throw new Error("The legacy v8 model did not migrate to the Nexus v9 actor.");
}

agent.start();
for (let frame = 0; frame < 60; frame++) {
    const callback = scheduled.shift();
    if (!callback) break;
    clock += 1000 / 60;
    callback(clock);
}

const response = agent.sendMessage("scan obstacle distance and report my energy");
agent.reward();
const snapshot = agent.getSnapshot();
const exported = agent.exportData();
if (snapshot.lastInputs.length !== 24 || snapshot.memories.length !== 1 ||
    !exported.decoder || !exported.tokenizer ||
    !response.toolCalls.some(call => call.name === "inspect_environment") ||
    agent.calculate("2 * (3 + 4)") !== 14) {
    throw new Error("Nexus multimodal, tool, persistence, or arithmetic smoke check failed.");
}
agent.stop();

console.log(JSON.stringify({
    model: MODEL_NAME,
    migratedFrom: "AgentFiveCore-v8",
    actorArchitecture: migrated.architecture,
    simulation: {
        position: snapshot.agent,
        energy: snapshot.energy,
        lastObservationSize: snapshot.lastInputs && snapshot.lastInputs.length
    },
    toolCalls: response.toolCalls.map(call => call.name),
    chatMessages: messages.length,
    streamedTokens: streamedTokens.length,
    reinforcementFeedback: "reward() applied"
}, null, 2));
