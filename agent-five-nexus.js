((root) => {
    "use strict";


    // Agent Five's runtime is independent of any particular UI. Host
    // applications inject world dimensions, persistence, scheduling,
    // learning services, workers, and optional rendering callbacks.
    function createAgentFive(options = {}) {
        const configuredWorld = options.world || options;
        const world = Object.freeze({
            width: Math.max(1, Number(configuredWorld.width) || 500),
            height: Math.max(1, Number(configuredWorld.height) || 500)
        });
        const ui = options.ui || Object.create(null);
        const learning = options.learning || root.AgentLearning;
        if (!learning || typeof learning.sampleAction !== "function" ||
            typeof learning.trainPPO !== "function") {
            throw new Error("Agent Five requires an AgentLearning implementation.");
        }
        const fallbackStore = new Map();
        const storage = options.storage || {
            getItem: key => fallbackStore.has(key) ? fallbackStore.get(key) : null,
            setItem: (key, value) => fallbackStore.set(key, String(value)),
            removeItem: key => fallbackStore.delete(key)
        };
        const createWorker = options.createWorker;
        const getNow = options.now || (() => Date.now());
        const requestFrame = options.requestFrame ||
            (callback => setTimeout(() => callback(getNow()), 16));
        const cancelFrame = options.cancelFrame || clearTimeout;


    const STORAGE_KEY = "agent5-brain-v8";
    const LEGACY_STORAGE_KEY_V7 = "agent5-brain-v7";
    const LEGACY_STORAGE_KEY_V6 = "agent5-brain-v6";
    const LEGACY_STORAGE_KEY_V5 = "agent5-brain-v5";
    const MEMORY_STORAGE_KEY = "agent5-memory-v2";
    const LEGACY_MEMORY_STORAGE_KEY_V1 = "agent5-memory-v1";
    const EPISODE_STORAGE_KEY = "agent5-episodes-v2";
    const LEGACY_EPISODE_STORAGE_KEY_V1 = "agent5-episodes-v1";
    const MEMORY_CAPACITY = 192;
    const EPISODE_MEMORY_CAPACITY = 48;
    const EPISODE_SEQUENCE_LENGTH = 12;
    const MEMORY_RECALL_THRESHOLD = 0.72;
    const MEMORY_RECALL_STRENGTH = 0.18;
    const MEMORY_FEATURE_WEIGHTS = [
        0.7, 0.7, 1.8, 1.8, 1.1, 1.1, 1.2, 0.12, 0.12, 0.75,
        0.9, 0.9, 0.9, 0.9, 0.8, 0.5
    ];
    const NETWORK_VERSION = 8;
    const INPUTS = 16;
    const LEGACY_HIDDEN_LAYERS_V7 = [9, 8, 7, 6, 5, 4];
    const HIDDEN_LAYERS = [
        15, 16, 15, 14, 15, 13, 14, 12, 13, 11, 12, 10, 11,
        12, 10, 9, 10, 8, 9, 7, 8, 6, 7, 5, 6, 4
    ];
    const OUTPUTS = 3;
    const DEFAULT_AGENT_RADIUS = 10;
    const GOAL_RADIUS = 12;
    const OBSTACLE_RADIUS = 30;
    const SPEED = 2.4;
    const ACCELERATION = 0.16;
    const BRAKING_STRENGTH = 0.18;
    const AVOIDANCE_RANGE = 115;
    const STILL_PUNISHMENT_MS = 10_000;
    const MOVEMENT_THRESHOLD = 0.05;
    const WALL_ESCAPE_TARGET = 0.82;
    const MAX_ENERGY = 100;
    const ENERGY_GOAL_REWARD = 35;
    const ENERGY_MOVEMENT_COST = 0.018;
    const ENERGY_THRUST_COST = 0.0125;
    const ENERGY_BRAKING_COST = 0.008;
    const ENERGY_WALL_IMPACT_COST = 0.75;
    const ENERGY_OBSTACLE_IMPACT_COST = 1.25;
    const DEFAULT_EVOLUTION_POPULATION_SIZE = 15;
    const MIN_EVOLUTION_POPULATION_SIZE = 15;
    const MAX_EVOLUTION_POPULATION_SIZE = 300;
    const EVOLUTION_GOALS_TO_WIN = 5;
    const EVOLUTION_MUTATION_RATE = 0.08;
    const EVOLUTION_MUTATION_STRENGTH = 0.10;
    const PPO_ROLLOUT_SIZE = 64;
    const PPO_POLICY_RATE = 0.00045;
    const PPO_VALUE_RATE = 0.0012;
    const PPO_POLICY_STD = 0.20;
    const PPO_EPOCHS = 2;
    const PPO_MINIBATCH_SIZE = 32;
    const SIMULATION_STEP_MS = 1000 / 60;
    const MAX_FRAME_DELTA_MS = 100;
    const MAX_SIMULATION_STEPS_PER_FRAME = 8;
    const EVOLUTION_TRANSITION_MS = 1200;
    const EVOLUTION_GENERATION_TIMEOUT_MS = 120000;


    class NeuralNetwork {
        constructor(saved, outputSize = OUTPUTS) {
            this.outputSize = outputSize;
            this.modelType = outputSize === 1 ? "critic" : "policy";
            this.sizes = [INPUTS, ...HIDDEN_LAYERS, outputSize];
            this.weights = [];
            this.biases = [];
            if (saved && this.isValid(saved)) {
                this.weights = saved.weights.map(matrix => matrix.map(row => row.slice()));
                this.biases = saved.biases.map(vector => vector.slice());
            } else {
                for (let layer = 0; layer < this.sizes.length - 1; layer++) {
                    const fanIn = this.sizes[layer];
                    const fanOut = this.sizes[layer + 1];
                    const limit = Math.sqrt(6 / (fanIn + fanOut));
                    this.weights.push(Array.from({ length: fanIn }, () =>
                        Array.from({ length: fanOut }, () => (Math.random() * 2 - 1) * limit)));
                    this.biases.push(Array(fanOut).fill(0));
                }
            }
        }


        isValid(data) {
            return data && data.version === NETWORK_VERSION &&
                data.modelType === this.modelType &&
                Array.isArray(data.architecture) &&
                data.architecture.length === this.sizes.length &&
                data.architecture.every((size, index) => size === this.sizes[index]) &&
                Array.isArray(data.weights) && Array.isArray(data.biases) &&
                data.weights.length === this.sizes.length - 1 &&
                data.biases.length === this.sizes.length - 1 &&
                data.weights.every((matrix, layer) => Array.isArray(matrix) &&
                    matrix.length === this.sizes[layer] &&
                    matrix.every(row => Array.isArray(row) &&
                        row.length === this.sizes[layer + 1] && row.every(Number.isFinite))) &&
                data.biases.every((vector, layer) => Array.isArray(vector) &&
                    vector.length === this.sizes[layer + 1] && vector.every(Number.isFinite));
        }


        forward(input) {
            const activations = [input];
            for (let layer = 0; layer < this.weights.length; layer++) {
                const prior = activations[layer];
                const next = this.biases[layer].map((bias, j) => {
                    let sum = bias;
                    for (let i = 0; i < prior.length; i++) sum += prior[i] * this.weights[layer][i][j];
                    return Math.tanh(sum);
                });
                activations.push(next);
            }
            return { output: activations[activations.length - 1], activations };
        }


        export() {
            return {
                version: NETWORK_VERSION,
                modelType: this.modelType,
                architecture: this.sizes.slice(),
                weights: this.weights,
                biases: this.biases
            };
        }


        applyTrainedModel(model) {
            if (!this.isValid(model)) return false;
            this.weights = model.weights.map(matrix => matrix.map(row => row.slice()));
            this.biases = model.biases.map(vector => vector.slice());
            return true;
        }
    }


    function cloneNetwork(source, mutationRate = 0, mutationStrength = EVOLUTION_MUTATION_STRENGTH) {
        const saved = source.export();
        const weights = saved.weights.map(matrix => matrix.map(row => row.map(value => {
            if (mutationRate > 0 && Math.random() < mutationRate) {
                value += (Math.random() * 2 - 1) * mutationStrength;
            }
            return clamp(value, -4, 4);
        })));
        const biases = saved.biases.map(vector => vector.map(value => {
            if (mutationRate > 0 && Math.random() < mutationRate) {
                value += (Math.random() * 2 - 1) * mutationStrength;
            }
            return clamp(value, -4, 4);
        }));
        return new NeuralNetwork({
            ...saved,
            architecture: saved.architecture.slice(),
            weights,
            biases
        }, source.outputSize);
    }


    function legacyBrainShape(data) {
        if (!data || ![5, 6, 7].includes(data.version) ||
            !Array.isArray(data.architecture) || !Array.isArray(data.weights) ||
            !Array.isArray(data.biases)) return null;


        const oldInputs = data.version === 5 ? 9 : data.version === 6 ? 10 : INPUTS;
        const outputSize = data.modelType === "critic" ||
            data.architecture[data.architecture.length - 1] === 1 ? 1 : OUTPUTS;
        const oldSizes = [oldInputs, ...LEGACY_HIDDEN_LAYERS_V7, outputSize];
        const validArchitecture = data.architecture.length === oldSizes.length &&
            data.architecture.every((size, index) => size === oldSizes[index]);
        const validWeights = data.weights.length === oldSizes.length - 1 &&
            data.weights.every((matrix, layer) => Array.isArray(matrix) &&
                matrix.length === oldSizes[layer] &&
                matrix.every(row => Array.isArray(row) &&
                    row.length === oldSizes[layer + 1] && row.every(Number.isFinite)));
        const validBiases = data.biases.length === oldSizes.length - 1 &&
            data.biases.every((vector, layer) => Array.isArray(vector) &&
                vector.length === oldSizes[layer + 1] && vector.every(Number.isFinite));


        return validArchitecture && validWeights && validBiases ?
            { oldInputs, outputSize, oldSizes } : null;
    }


    function isValidLegacyBrain(data) {
        return Boolean(legacyBrainShape(data));
    }


    // Move v5-v7 models into the wider v8 network while retaining the old
    // learned pathway. New input connections start neutral; the added hidden
    // layers carry the old four-unit representation forward until PPO adapts it.
    function migrateLegacyBrain(data) {
        const shape = legacyBrainShape(data);
        if (!shape) return null;


        const migrated = new NeuralNetwork(null, shape.outputSize).export();
        migrated.weights = migrated.weights.map(matrix =>
            matrix.map(row => row.map(() => 0)));
        migrated.biases = migrated.biases.map(vector => vector.map(() => 0));


        for (let layer = 0; layer < LEGACY_HIDDEN_LAYERS_V7.length; layer++) {
            const oldMatrix = data.weights[layer];
            for (let oldSource = 0; oldSource < oldMatrix.length; oldSource++) {
                let newSource = oldSource;
                if (layer === 0 && data.version === 7) {
                    if (oldSource === 14) continue;
                    if (oldSource === 15) newSource = 14;
                }
                if (newSource >= INPUTS) continue;


                for (let target = 0; target < oldMatrix[oldSource].length; target++) {
                    migrated.weights[layer][newSource][target] = oldMatrix[oldSource][target];
                }
            }


            for (let target = 0; target < data.biases[layer].length; target++) {
                migrated.biases[layer][target] = data.biases[layer][target];
            }
        }


        // The v7 actor and critic ended in four hidden units. Keep those four
        // channels connected through the twenty added layers as a neutral bridge.
        const bridgeWidth = LEGACY_HIDDEN_LAYERS_V7[LEGACY_HIDDEN_LAYERS_V7.length - 1];
        for (let layer = LEGACY_HIDDEN_LAYERS_V7.length;
            layer < HIDDEN_LAYERS.length; layer++) {
            const matrix = migrated.weights[layer];
            for (let index = 0; index < Math.min(bridgeWidth, matrix.length, matrix[0].length); index++) {
                matrix[index][index] = 1;
            }
        }


        const oldOutputLayer = data.weights[LEGACY_HIDDEN_LAYERS_V7.length];
        const finalLayer = HIDDEN_LAYERS.length;
        for (let sourceIndex = 0; sourceIndex < oldOutputLayer.length; sourceIndex++) {
            for (let outputIndex = 0; outputIndex < shape.outputSize; outputIndex++) {
                migrated.weights[finalLayer][sourceIndex][outputIndex] =
                    oldOutputLayer[sourceIndex][outputIndex] * 2.5;
            }
        }
        migrated.biases[finalLayer] = data.biases[data.biases.length - 1].slice();


        return {
            ...migrated,
            version: NETWORK_VERSION,
            modelType: shape.outputSize === 1 ? "critic" : "policy",
            architecture: [INPUTS, ...HIDDEN_LAYERS, shape.outputSize]
        };
    }


    let loadedCriticModel = null;


    function loadNetwork() {
        try {
            const network = new NeuralNetwork(null);
            const currentRaw = storage.getItem(STORAGE_KEY);
            if (currentRaw) {
                const currentModel = JSON.parse(currentRaw);
                if (network.isValid(currentModel)) {
                    const criticValidator = new NeuralNetwork(null, 1);
                    if (criticValidator.isValid(currentModel.critic)) {
                        loadedCriticModel = currentModel.critic;
                    }
                    return new NeuralNetwork(currentModel);
                }
            }


            const legacySources = [
                storage.getItem(LEGACY_STORAGE_KEY_V7),
                storage.getItem(LEGACY_STORAGE_KEY_V6),
                storage.getItem(LEGACY_STORAGE_KEY_V5)
            ];
            for (const legacyRaw of legacySources) {
                if (!legacyRaw) continue;
                const legacyData = JSON.parse(legacyRaw);
                const migrated = migrateLegacyBrain(legacyData);
                if (!migrated || migrated.modelType !== "policy") continue;


                const migratedCritic = migrateLegacyBrain(legacyData.critic);
                if (migratedCritic && migratedCritic.modelType === "critic") {
                    loadedCriticModel = migratedCritic;
                }
                const upgradedBrain = {
                    ...migrated,
                    ...(loadedCriticModel ? { critic: loadedCriticModel } : {})
                };
                storage.setItem(STORAGE_KEY, JSON.stringify(upgradedBrain));
                return new NeuralNetwork(migrated);
            }
            return network;
        } catch (error) {
            console.warn("Could not load or migrate the saved Agent Five model; starting fresh.", error);
            return new NeuralNetwork(null);
        }
    }


    function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
    function distance(x, y) { return Math.hypot(x, y); }


    function defaultMemoryPriority(outcome) {
        if (outcome === "goal-reward") return 3;
        if (outcome === "energy-depletion") return 2.4;
        if (outcome === "wall-contact") return 2.1;
        return 1.4;
    }


    function normalizeObservationInputs(values, remapV7Ordering = false) {
        if (!Array.isArray(values) || !values.every(Number.isFinite)) return null;


        // v7 stored obstacle-closing speed at index 14 and goal distance at
        // index 15. v8 replaces those slots with goal distance and energy use.
        const inputs = values.slice();
        if (remapV7Ordering && inputs.length === INPUTS) {
            inputs[14] = values[15];
            inputs[15] = 0;
        }


        // Older v5/v6 memories have 9 or 10 inputs. Preserve their known
        // values and neutral-pad the perception features introduced later.
        if (inputs.length === 9) inputs.push(0);
        while (inputs.length < INPUTS && inputs.length >= 10) inputs.push(0);
        return inputs.length === INPUTS ? inputs : null;
    }


    function normalizeMemoryItem(item, remapV7Ordering = false) {
        if (!item || !Array.isArray(item.inputs) ||
            !Array.isArray(item.target) || item.target.length !== OUTPUTS ||
            !item.target.every(Number.isFinite) || typeof item.outcome !== "string") return null;


        const inputs = normalizeObservationInputs(item.inputs, remapV7Ordering);
        if (!inputs) return null;


        const now = Date.now();
        return {
            inputs,
            target: item.target.map(value => clamp(value, -1, 1)),
            outcome: item.outcome,
            priority: clamp(Number.isFinite(item.priority) ? item.priority : defaultMemoryPriority(item.outcome), 0.1, 4),
            createdAt: Number.isFinite(item.createdAt) && item.createdAt > 0 ? item.createdAt : now,
            lastUsedAt: Number.isFinite(item.lastUsedAt) && item.lastUsedAt > 0 ? item.lastUsedAt : 0,
            replayCount: Number.isFinite(item.replayCount) ? Math.max(0, Math.floor(item.replayCount)) : 0,
            occurrences: Number.isFinite(item.occurrences) ? Math.max(1, Math.floor(item.occurrences)) : 1
        };
    }


    function memorySimilarity(left, right) {
        let weightedDistance = 0;
        let totalWeight = 0;
        for (let index = 0; index < INPUTS; index++) {
            const weight = MEMORY_FEATURE_WEIGHTS[index];
            const difference = left[index] - right[index];
            weightedDistance += difference * difference * weight;
            totalWeight += weight;
        }
        return Math.exp(-2.4 * Math.sqrt(weightedDistance / totalWeight));
    }


    function memoryAgeFactor(item, now = Date.now()) {
        const ageDays = Math.max(0, now - item.createdAt) / 86400000;
        return Math.pow(0.5, ageDays / 45);
    }


    function memoryRetentionScore(item, now = Date.now()) {
        const repetitionBonus = 1 + Math.min(Math.log2(item.occurrences), 4) * 0.06;
        const replayBonus = 1 + Math.min(item.replayCount, 16) * 0.025;
        const recentUseBonus = item.lastUsedAt > 0 &&
            now - item.lastUsedAt < 7 * 86400000 ? 1.08 : 1;
        return item.priority * memoryAgeFactor(item, now) *
            repetitionBonus * replayBonus * recentUseBonus;
    }


    function normalizeMemoryCollection(items, remapV7Ordering = false) {
        if (!Array.isArray(items)) return [];
        const normalized = items.map(item => normalizeMemoryItem(item, remapV7Ordering)).filter(Boolean);
        if (normalized.length <= MEMORY_CAPACITY) return normalized;


        const now = Date.now();
        return normalized
            .sort((a, b) => memoryRetentionScore(b, now) - memoryRetentionScore(a, now))
            .slice(0, MEMORY_CAPACITY)
            .sort((a, b) => a.createdAt - b.createdAt);
    }


    function cloneMemories(items) {
        return normalizeMemoryCollection(items).map(item => ({
            ...item, inputs: item.inputs.slice(), target: item.target.slice()
        }));
    }


    function loadMemory() {
        try {
            const currentRaw = storage.getItem(MEMORY_STORAGE_KEY);
            if (currentRaw) return normalizeMemoryCollection(JSON.parse(currentRaw));


            const legacyRaw = storage.getItem(LEGACY_MEMORY_STORAGE_KEY_V1);
            if (!legacyRaw) return [];
            const migrated = normalizeMemoryCollection(JSON.parse(legacyRaw), true);
            storage.setItem(MEMORY_STORAGE_KEY, JSON.stringify(migrated));
            return migrated;
        } catch (error) {
            console.warn("Could not load or migrate Agent Five's experience memory.", error);
            return [];
        }
    }


    function normalizeEpisode(item, remapV7Ordering = false) {
        if (!item || !Array.isArray(item.steps) || item.steps.length < 3 ||
            typeof item.outcome !== "string") return null;
        const steps = item.steps.slice(-EPISODE_SEQUENCE_LENGTH).map(step => {
            if (!step || !Array.isArray(step.inputs) ||
                !Array.isArray(step.target) || step.target.length !== OUTPUTS ||
                !step.target.every(Number.isFinite) || !Number.isFinite(step.reward)) return null;
            const inputs = normalizeObservationInputs(step.inputs, remapV7Ordering);
            if (!inputs) return null;
            return {
                inputs,
                target: step.target.map(value => clamp(value, -1, 1)),
                reward: clamp(step.reward, -8, 8)
            };
        });
        if (steps.some(step => !step)) return null;
        return {
            steps,
            outcome: item.outcome,
            priority: clamp(Number.isFinite(item.priority) ? item.priority : 1, 0.1, 5),
            createdAt: Number.isFinite(item.createdAt) && item.createdAt > 0 ? item.createdAt : Date.now(),
            uses: Number.isFinite(item.uses) ? Math.max(0, Math.floor(item.uses)) : 0
        };
    }


    function loadEpisodes() {
        try {
            const currentRaw = storage.getItem(EPISODE_STORAGE_KEY);
            if (currentRaw) {
                const current = JSON.parse(currentRaw);
                return Array.isArray(current) ?
                    current.map(item => normalizeEpisode(item)).filter(Boolean).slice(-EPISODE_MEMORY_CAPACITY) : [];
            }


            const legacyRaw = storage.getItem(LEGACY_EPISODE_STORAGE_KEY_V1);
            if (!legacyRaw) return [];
            const legacy = JSON.parse(legacyRaw);
            const migrated = Array.isArray(legacy) ?
                legacy.map(item => normalizeEpisode(item, true)).filter(Boolean).slice(-EPISODE_MEMORY_CAPACITY) : [];
            storage.setItem(EPISODE_STORAGE_KEY, JSON.stringify(migrated));
            return migrated;
        } catch (error) {
            console.warn("Could not load or migrate Agent Five's episodic memory.", error);
            return [];
        }
    }


    function saveEpisodes(episodeBank = episodes) {
        try {
            storage.setItem(EPISODE_STORAGE_KEY, JSON.stringify(episodeBank));
        } catch (error) {
            console.warn("Could not save Agent Five's episodic memory.", error);
        }
        updateMemoryOverview(memories, lastMemoryRecallCount, episodeBank);
    }


    function cloneEpisodes(items) {
        return (Array.isArray(items) ? items : []).map(item => normalizeEpisode(item))
            .filter(Boolean).map(item => ({
                ...item,
                steps: item.steps.map(step => ({
                    inputs: step.inputs.slice(),
                    target: step.target.slice(),
                    reward: step.reward
                }))
            })).slice(-EPISODE_MEMORY_CAPACITY);
    }


    function rememberEpisode(episodeBank, trajectory, outcome) {
        if (!Array.isArray(trajectory) || trajectory.length < 3) return false;
        const steps = trajectory.slice(-EPISODE_SEQUENCE_LENGTH).map(item => ({
            inputs: item.inputs.slice(),
            target: [
                clamp(item.action[0], -1, 1),
                clamp(item.action[1], -1, 1),
                clamp(item.action[2] * 2 - 1, -1, 1)
            ],
            reward: clamp(item.reward, -8, 8)
        }));
        const episode = normalizeEpisode({
            steps,
            outcome,
            priority: outcome === "goal" ? 3.2 : 2.4,
            createdAt: Date.now()
        });
        if (!episode) return false;
        episodeBank.push(episode);
        if (episodeBank.length > EPISODE_MEMORY_CAPACITY) {
            episodeBank.sort((a, b) => {
                const scoreA = a.priority * Math.pow(0.5, Math.max(0, Date.now() - a.createdAt) / 86400000 / 45) +
                    Math.min(a.uses, 20) * 0.015;
                const scoreB = b.priority * Math.pow(0.5, Math.max(0, Date.now() - b.createdAt) / 86400000 / 45) +
                    Math.min(b.uses, 20) * 0.015;
                return scoreB - scoreA;
            });
            episodeBank.length = EPISODE_MEMORY_CAPACITY;
        }
        return true;
    }


    function recallEpisodeAction(inputs, episodeBank) {
        let best = null;
        const now = Date.now();
        for (const episode of episodeBank) {
            for (let index = 0; index < episode.steps.length; index++) {
                const step = episode.steps[index];
                const similarity = memorySimilarity(inputs, step.inputs);
                if (similarity < 0.78) continue;
                const ageFactor = Math.pow(0.5,
                    Math.max(0, now - episode.createdAt) / 86400000 / 60);
                const score = similarity * ageFactor * episode.priority;
                if (!best || score > best.score) {
                    best = { score, similarity, target: step.target, episode };
                }
            }
        }
        if (!best) return null;
        best.episode.uses++;
        const closeness = clamp((best.similarity - 0.78) / 0.22, 0, 1);
        return {
            target: best.target.slice(),
            strength: 0.12 * closeness,
            count: 1
        };
    }


    function memoryOutcomeLabel(outcome) {
        const labels = {
            "goal-reward": "Goal reward",
            "behavior-correction": "Behavior correction",
            "energy-depletion": "Energy depletion",
            "wall-contact": "Wall contact"
        };
        return labels[outcome] || "Other experience";
    }


    function createSnapshot() {
        const selected = evolutionaryMode ?
            (evolutionSuccessor || getEvolutionDisplayAgent()) : null;
        return {
            dimensions: world,
            config: {
                maxEnergy: MAX_ENERGY, goalRadius: GOAL_RADIUS,
                obstacleRadius: OBSTACLE_RADIUS, avoidanceRange: AVOIDANCE_RANGE,
                goalsToWin: EVOLUTION_GOALS_TO_WIN, memoryCapacity: MEMORY_CAPACITY,
                episodeCapacity: EPISODE_MEMORY_CAPACITY,
                minPopulationSize: MIN_EVOLUTION_POPULATION_SIZE,
                maxPopulationSize: MAX_EVOLUTION_POPULATION_SIZE,
                policyRate: PPO_POLICY_RATE, valueRate: PPO_VALUE_RATE, policyStd: PPO_POLICY_STD
            },
            agent: { ...agent }, velocity: { ...velocity }, goal: { ...goal }, obstacle: { ...obstacle },
            energy, energyDepleted, energyUsageRate,
            lastInputs: lastInputs ? lastInputs.slice() : null,
            lastOutput: lastOutput ? lastOutput.slice() : null,
            lastMemoryRecall, lastMemoryRecallCount, memories, episodes,
            counters: {
                trainingExamples, trainingRewards, trainingCorrections, wallPenaltyFrames,
                ppoUpdateCount, ppoRewardAverage, evolutionTrainingExamples,
                evolutionTrainingRewards, evolutionTrainingCorrections, evolutionWallPenaltyFrames
            },
            evolutionaryMode, evolutionPopulation, evolutionPopulationSize, evolutionGeneration,
            evolutionMutationRate, evolutionDiversity, evolutionSuccessor,
            evolutionLastSuccessorId, selectedEvolutionAgent: selected
        };
    }


    function updateMemoryOverview(memoryBank = memories,
        recallCount = lastMemoryRecallCount, episodeBank = episodes) {
        if (typeof ui.updateMemoryOverview === "function") {
            ui.updateMemoryOverview({
                memoryBank: Array.isArray(memoryBank) ? memoryBank : [],
                recallCount: Number(recallCount) || 0,
                episodeBank: Array.isArray(episodeBank) ? episodeBank : [],
                memoryCapacity: MEMORY_CAPACITY,
                episodeCapacity: EPISODE_MEMORY_CAPACITY
            });
        }
    }


    function saveMemory() {
        try {
            storage.setItem(MEMORY_STORAGE_KEY, JSON.stringify(memories));
        } catch (error) {
            console.warn("Could not save Agent Five's experience memory.", error);
        }
        updateMemoryOverview(memories, lastMemoryRecallCount);
    }


    // Similar experiences are merged instead of filling the bank with duplicates.
    // When full, the least useful/oldest item is evicted rather than always
    // deleting the oldest experience.
    function addMemory(memoryBank, inputs, target, outcome) {
        if (!Array.isArray(inputs) || inputs.length !== INPUTS ||
            !inputs.every(Number.isFinite) ||
            !Array.isArray(target) || target.length !== OUTPUTS ||
            !target.every(Number.isFinite)) return null;


        const now = Date.now();
        const existing = memoryBank.find(item =>
            item.outcome === outcome && memorySimilarity(item.inputs, inputs) >= 0.985
        );


        if (existing) {
            existing.target = existing.target.map((value, index) =>
                clamp(value * 0.75 + target[index] * 0.25, -1, 1));
            existing.priority = clamp(existing.priority + 0.2, 0.1, 4);
            existing.createdAt = now;
            existing.occurrences++;
            return existing;
        }


        const item = normalizeMemoryItem({
            inputs: inputs.slice(),
            target: target.slice(),
            outcome,
            priority: defaultMemoryPriority(outcome),
            createdAt: now
        });
        if (!item) return null;
        memoryBank.push(item);


        if (memoryBank.length > MEMORY_CAPACITY) {
            let weakestIndex = 0;
            let weakestScore = Infinity;
            for (let index = 0; index < memoryBank.length; index++) {
                const score = memoryRetentionScore(memoryBank[index], now);
                if (score < weakestScore) {
                    weakestScore = score;
                    weakestIndex = index;
                }
            }
            memoryBank.splice(weakestIndex, 1);
        }
        return item;
    }


    function replayWeight(item, now = Date.now()) {
        const outcomeWeight = item.outcome === "goal-reward" ? 1.6 :
            item.outcome === "energy-depletion" ? 1.25 : 1;
        const repetitionBonus = 1 + Math.min(Math.log2(item.occurrences), 4) * 0.05;
        return item.priority * outcomeWeight * repetitionBonus * memoryAgeFactor(item, now);
    }


    // Experiences are stored and retrieved as policy hints. Policy updates
    // now come from PPO rollouts rather than replaying isolated actions as labels.
    function storeExperience(memoryBank, inputs, target, outcome) {
        addMemory(memoryBank, inputs, target, outcome);
        return memoryBank.length;
    }


    function recallMemoryAction(networkOutput, inputs, memoryBank, episodeBank = episodes) {
        let output = networkOutput.slice();
        let totalStrength = 0;
        let recallCount = 0;


        if (memoryBank.length) {
            const now = Date.now();
            const selected = [];
            for (const item of memoryBank) {
                const similarity = memorySimilarity(inputs, item.inputs);
                if (similarity < MEMORY_RECALL_THRESHOLD) continue;
                const match = { item, similarity, weight: 0 };
                let insertAt = selected.findIndex(existing => similarity > existing.similarity);
                if (insertAt < 0) insertAt = selected.length;
                selected.splice(insertAt, 0, match);
                if (selected.length > 4) selected.pop();
            }


            if (selected.length) {
                let totalWeight = 0;
                const recalledAction = Array(OUTPUTS).fill(0);
                for (const match of selected) {
                    match.weight = Math.pow(match.similarity, 6) * replayWeight(match.item, now);
                    totalWeight += match.weight;
                    for (let index = 0; index < OUTPUTS; index++) {
                        recalledAction[index] += match.item.target[index] * match.weight;
                    }
                }
                if (totalWeight > 0) {
                    for (let index = 0; index < OUTPUTS; index++) recalledAction[index] /= totalWeight;
                    const closeness = clamp(
                        (selected[0].similarity - MEMORY_RECALL_THRESHOLD) /
                            (1 - MEMORY_RECALL_THRESHOLD), 0, 1
                    );
                    const strength = MEMORY_RECALL_STRENGTH * closeness;
                    output = output.map((value, index) =>
                        clamp(value * (1 - strength) + recalledAction[index] * strength, -1, 1));
                    totalStrength += strength;
                    recallCount += selected.length;
                    for (const match of selected) match.item.lastUsedAt = now;
                }
            }
        }


        const episodicRecall = recallEpisodeAction(inputs, episodeBank);
        if (episodicRecall && episodicRecall.strength > 0) {
            const strength = episodicRecall.strength;
            output = output.map((value, index) =>
                clamp(value * (1 - strength) + episodicRecall.target[index] * strength, -1, 1));
            totalStrength += strength;
            recallCount += episodicRecall.count;
        }


        return {
            output,
            strength: clamp(totalStrength, 0, 0.35),
            count: recallCount
        };
    }


    let brain = loadNetwork();
    let critic = new NeuralNetwork(loadedCriticModel, 1);
    let memories = loadMemory();
    let episodes = loadEpisodes();
    const singlePPOState = {
        rolloutBuffer: [], trajectory: [], lastTransition: null,
        ppoPending: false, ppoToken: 0
    };
    let ppoUpdateCount = 0;
    let ppoRewardAverage = 0;
    let ppoReadyJobs = [];
    let ppoFallbackQueue = [];
    let ppoPendingJobs = new Map();
    let ppoWorkerQueue = [];
    let ppoWorkerBusy = false;
    let ppoRequestId = 0;
    let learningWorker = null;
    let lastTransition = null;
    let lastPlannedMean = [0, 0, 0];
    let simulationAccumulator = 0;
    let simulationClock = 0;
    let evolutionMutationRate = EVOLUTION_MUTATION_RATE;
    let evolutionMutationStrength = EVOLUTION_MUTATION_STRENGTH;
    let evolutionDiversity = 0;
    let previousGenerationDuration = null;
    let trainingExamples = 0;
    let trainingRewards = 0;
    let trainingCorrections = 0;
    let wallPenaltyFrames = 0;
    let analyticsHistory = [];
    let analyticsLastSampleAt = 0;
    let evolutionTrainingRewards = 0;
    let evolutionTrainingCorrections = 0;
    let evolutionWallPenaltyFrames = 0;
    let agent = { x: 90, y: 410, radius: DEFAULT_AGENT_RADIUS };
    let velocity = { x: 0, y: 0 };
    let goal = { x: 395, y: 100 };
    let obstacle = { x: 245, y: 250 };
    let lastInputs = null;
    let lastOutput = [0, 0, 0];
    let lastMemoryRecall = 0;
    let lastMemoryRecallCount = 0;
    let saveCountdown = 0;
    let stillSince = null;
    // This flag gates the one-time impact energy cost; learning correction is
    // applied on every frame that the agent remains in contact with a wall.
    let touchingWall = false;
    let energy = MAX_ENERGY;
    let energyUsageRate = 0;
    let energyDepleted = false;
    let lastFrameTimestamp = null;
    let running = false;
    let frameHandle = null;
    let evolutionaryMode = false;
    let evolutionStopPending = false;
    let evolutionPopulation = [];
    let evolutionPopulationSize = DEFAULT_EVOLUTION_POPULATION_SIZE;
    let evolutionGeneration = 0;
    let evolutionTrainingExamples = 0;
    let evolutionSuccessor = null;
    let evolutionLastSuccessorId = null;
    let evolutionGenerationStartedAt = 0;
    let generationTransitionUntil = 0;


    function updatePPOStatistics(stats) {
        if (!stats || !Number.isFinite(stats.updates)) return;
        ppoUpdateCount += stats.updates;
        const reward = Number.isFinite(stats.meanReward) ? stats.meanReward : 0;
        ppoRewardAverage = ppoUpdateCount <= stats.updates ? reward :
            ppoRewardAverage * 0.9 + reward * 0.1;
    }


    function applyPPOTrainingResult(record, result) {
        const target = record.target;
        if (!target || !result || result.error) {
            if (target && target.ppoToken === record.token) target.ppoPending = false;
            if (result && result.error) console.warn("Agent Five PPO training failed:", result.error);
            return;
        }


        if (record.kind === "single") {
            if (record.token !== singlePPOState.ppoToken) return;
            brain.applyTrainedModel(result.policy);
            critic.applyTrainedModel(result.critic);
            singlePPOState.ppoPending = false;
        } else {
            if (record.generation !== evolutionGeneration ||
                !evolutionPopulation.includes(target) ||
                target.ppoToken !== record.token) return;
            target.network.applyTrainedModel(result.policy);
            target.critic.applyTrainedModel(result.critic);
            target.ppoPending = false;
        }
        updatePPOStatistics(result.stats);
    }


    function pumpPPOWorkerQueue() {
        if (!learningWorker || ppoWorkerBusy || !ppoWorkerQueue.length) return;
        const records = ppoWorkerQueue.splice(0, 16);
        const requestId = ++ppoRequestId;
        ppoPendingJobs.set(requestId, records);
        ppoWorkerBusy = true;
        try {
            learningWorker.postMessage({
                requestId,
                jobs: records.map(record => record.job)
            });
        } catch (error) {
            console.warn("Agent Five could not send a PPO batch to its worker.", error);
            ppoPendingJobs.delete(requestId);
            ppoWorkerBusy = false;
            try { learningWorker.terminate(); } catch (terminateError) {}
            learningWorker = null;
            for (const record of records) ppoFallbackQueue.push(record);
            for (const record of ppoWorkerQueue) ppoFallbackQueue.push(record);
            ppoWorkerQueue = [];
        }
    }


    function handlePPOWorkerMessage(event) {
        const message = event.data || {};
        const records = ppoPendingJobs.get(message.requestId);
        if (!records) return;
        ppoPendingJobs.delete(message.requestId);
        ppoWorkerBusy = false;
        const results = Array.isArray(message.results) ? message.results : [];
        records.forEach((record, index) => {
            const result = results[index];
            if (result && result.token === record.token) {
                applyPPOTrainingResult(record, result);
            } else if (record.target && record.target.ppoToken === record.token) {
                record.target.ppoPending = false;
            }
        });
        pumpPPOWorkerQueue();
    }


    function initializeLearningWorker() {
        if (typeof createWorker !== "function") return;
        try {
            learningWorker = createWorker("learning-worker.js?v=20261106-neural-v8");
            learningWorker.addEventListener("message", handlePPOWorkerMessage);
            learningWorker.addEventListener("error", error => {
                console.warn("Agent Five learning worker stopped; using the safe main-thread fallback.", error);
                if (learningWorker) learningWorker.terminate();
                learningWorker = null;
                ppoWorkerBusy = false;
                for (const records of ppoPendingJobs.values()) {
                    for (const record of records) ppoFallbackQueue.push(record);
                }
                for (const record of ppoWorkerQueue) ppoFallbackQueue.push(record);
                ppoPendingJobs.clear();
                ppoWorkerQueue = [];
            });
        } catch (error) {
            console.warn("Agent Five could not start its learning worker.", error);
            learningWorker = null;
        }
    }


    function dispatchPPOJobs(records) {
        if (!records || !records.length) return;
        if (learningWorker) {
            ppoWorkerQueue.push(...records);
            pumpPPOWorkerQueue();
            return;
        }
        for (const record of records) ppoFallbackQueue.push(record);
    }


    function drainFallbackPPOQueue(maximumJobs = 2) {
        for (let count = 0; count < maximumJobs && ppoFallbackQueue.length; count++) {
            const record = ppoFallbackQueue.shift();
            try {
                const trained = learning.trainPPO(
                    record.job.policy,
                    record.job.critic,
                    record.job.rollout,
                    record.job.options
                );
                applyPPOTrainingResult(record, {
                    token: record.token,
                    policy: trained.policy,
                    critic: trained.critic,
                    stats: trained.stats
                });
            } catch (error) {
                console.warn("Agent Five main-thread PPO fallback failed.", error);
                if (record.target && record.target.ppoToken === record.token) {
                    record.target.ppoPending = false;
                }
            }
        }
    }


    function schedulePPOTraining(target, kind, generation = 0, force = false) {
        if (!target || target.ppoPending || !Array.isArray(target.rolloutBuffer) ||
            target.rolloutBuffer.length < (force ? 1 : PPO_ROLLOUT_SIZE)) return null;


        const network = kind === "single" ? brain : target.network;
        const valueNetwork = kind === "single" ? critic : target.critic;
        const rollout = target.rolloutBuffer.splice(
            0, force ? target.rolloutBuffer.length : PPO_ROLLOUT_SIZE
        );
        const token = target.ppoToken;
        const lastTransitionInBatch = rollout[rollout.length - 1];
        const nextInputs = kind === "single" ?
            (lastInputs ? observe(agent, goal, obstacle) : null) : observeEvolution(target);


        if (lastTransitionInBatch && !lastTransitionInBatch.done && nextInputs) {
            lastTransitionInBatch.nextValue = valueNetwork.forward(nextInputs).output[0];
        }


        target.ppoPending = true;
        return {
            target,
            kind,
            token,
            generation,
            job: {
                id: kind === "single" ? 0 : target.id,
                generation,
                token,
                policy: network.export(),
                critic: valueNetwork.export(),
                rollout,
                options: {
                    policyRate: PPO_POLICY_RATE,
                    valueRate: PPO_VALUE_RATE,
                    policyStandardDeviation: PPO_POLICY_STD,
                    epochs: PPO_EPOCHS,
                    minibatchSize: PPO_MINIBATCH_SIZE
                }
            }
        };
    }


    function createPPOTransition(inputs, decision, target) {
        const initialReward = Number.isFinite(target.pendingReward) ? target.pendingReward : 0;
        target.pendingReward = 0;
        const transition = {
            inputs: inputs.slice(),
            latent: decision.latent.slice(),
            meanOffset: decision.meanOffset.slice(),
            logProbability: decision.logProbability,
            value: decision.value,
            reward: -0.002 + initialReward,
            done: false,
            action: decision.action.slice()
        };
        target.rolloutBuffer.push(transition);
        if (target.rolloutBuffer.length > PPO_ROLLOUT_SIZE * 4) {
            target.rolloutBuffer.shift();
        }
        target.lastTransition = transition;
        target.lastTrajectoryEntry = {
            inputs: inputs.slice(),
            action: decision.action.slice(),
            reward: transition.reward
        };
        target.trajectory.push(target.lastTrajectoryEntry);
        if (target.trajectory.length > 80) target.trajectory.shift();
        return transition;
    }


    function addTransitionReward(target, amount) {
        if (!target || !Number.isFinite(amount)) return;
        const transition = target.lastTransition;
        if (transition && target.rolloutBuffer.includes(transition)) {
            transition.reward = clamp(transition.reward + amount, -8, 8);
        } else {
            target.pendingReward = clamp((target.pendingReward || 0) + amount, -8, 8);
        }
        const recent = target.lastTrajectoryEntry;
        if (recent) recent.reward = clamp(recent.reward + amount, -8, 8);
    }


    function planAndSampleAction(network, valueNetwork, inputs, position, currentVelocity,
        environment, currentEnergy, memoryBank, episodeBank) {
        const networkResult = network.forward(inputs);
        const recalled = recallMemoryAction(networkResult.output, inputs, memoryBank, episodeBank);
        const plannedMean = learning.planAction(recalled.output, {
            x: position.x,
            y: position.y,
            velocity: currentVelocity,
            goal: environment.goal,
            obstacle: environment.obstacle
        }, {
            width: world.width,
            height: world.height,
            speed: SPEED,
            acceleration: ACCELERATION,
            brakingStrength: BRAKING_STRENGTH,
            agentRadius: position.radius || DEFAULT_AGENT_RADIUS,
            obstacleRadius: OBSTACLE_RADIUS,
            avoidanceRange: AVOIDANCE_RANGE,
            horizon: 12,
            blend: 0.24
        });
        const decision = learning.sampleAction(
            network, valueNetwork, inputs, plannedMean, PPO_POLICY_STD, networkResult
        );
        decision.activations = networkResult.activations;
        decision.memoryRecall = recalled;
        return decision;
    }


    function setAgentRadius(value) {
        const nextRadius = clamp(Number(value) || DEFAULT_AGENT_RADIUS, 8, 30);
        agent.radius = nextRadius;
        agent.x = clamp(agent.x, nextRadius, world.width - nextRadius);
        agent.y = clamp(agent.y, nextRadius, world.height - nextRadius);
        if (typeof ui.setAgentRadius === "function") ui.setAgentRadius(nextRadius);
        return nextRadius;
    }


    function createObservation(position, currentVelocity, target, block, currentEnergy, currentEnergyUsageRate = 0) {
        const radius = position.radius ?? DEFAULT_AGENT_RADIUS;
        const goalDx = (target.x - position.x) / world.width;
        const goalDy = (target.y - position.y) / world.height;
        const obstacleDeltaX = block.x - position.x;
        const obstacleDeltaY = block.y - position.y;
        const obstacleDx = obstacleDeltaX / world.width;
        const obstacleDy = obstacleDeltaY / world.height;
        const obstacleDistance = distance(obstacleDeltaX, obstacleDeltaY);
        const gap = obstacleDistance - OBSTACLE_RADIUS - radius;
        const clearance = clamp(gap / Math.hypot(world.width, world.height) * 4, -1, 1);
        const currentLocationX = clamp(position.x / world.width, 0, 1);
        const currentLocationY = clamp(position.y / world.height, 0, 1);
        const currentSpeedX = clamp(currentVelocity.x / SPEED, -1, 1);
        const currentSpeedY = clamp(currentVelocity.y / SPEED, -1, 1);
        const energyLeft = clamp(currentEnergy / MAX_ENERGY, 0, 1);
        const halfWidth = Math.max(1, world.width / 2);
        const halfHeight = Math.max(1, world.height / 2);
        const leftClearance = clamp((position.x - radius) / halfWidth, 0, 1);
        const rightClearance = clamp((world.width - radius - position.x) / halfWidth, 0, 1);
        const topClearance = clamp((position.y - radius) / halfHeight, 0, 1);
        const bottomClearance = clamp((world.height - radius - position.y) / halfHeight, 0, 1);
        const goalDistance = clamp(
            distance(target.x - position.x, target.y - position.y) /
                Math.hypot(world.width, world.height), 0, 1
        );
        const normalizedEnergyUsageRate = clamp(currentEnergyUsageRate, 0, 1);


        return [
            currentSpeedX, currentSpeedY, goalDx, goalDy, obstacleDx, obstacleDy,
            clearance, currentLocationX, currentLocationY, energyLeft,
            leftClearance, rightClearance, topClearance, bottomClearance,
            goalDistance, normalizedEnergyUsageRate
        ];
    }


    function observe(position, target, block) {
        return createObservation(position, velocity, target, block, energy, energyUsageRate);
    }


    function saveBrain() {
        try {
            storage.setItem(STORAGE_KEY, JSON.stringify({
                ...brain.export(),
                critic: critic.export()
            }));
            if (typeof ui.setModelStatus === "function") ui.setModelStatus("Actor + critic saved");
        } catch (error) {
            console.warn("Could not save Agent Five's actor-critic model.", error);
            if (typeof ui.setModelStatus === "function") ui.setModelStatus("Session only");
        }
    }


    function showFeedback(message) {
        if (typeof ui.showFeedback === "function") ui.showFeedback(String(message));
    }


    // Feedback adjusts the network's own action at the observed state; there is
    // no hand-authored steering policy controlling or replacing its output.
    function reinforce(isReward) {
        if (!lastInputs) return;
        const target = lastOutput.map((value, index) => {
            if (isReward) return index < 2 ? clamp(value * 1.1, -1, 1) : value;
            return index < 2 ? 0 : Math.min(value, 0);
        });
        storeExperience(memories, lastInputs, target,
            isReward ? "goal-reward" : "behavior-correction");
        trainingExamples++;
        if (isReward) trainingRewards++;
        else trainingCorrections++;
        saveBrain();
        saveMemory();
        updateDiagnostics(lastInputs, brain.forward(lastInputs).output);
        showFeedback(isReward ? "Reward applied to the network's action." : "Correction applied to the network's action.");
    }


    function wallEscapeTarget(position, radius) {
        let targetX = 0;
        let targetY = 0;
        if (position.x <= radius + 0.001) targetX = WALL_ESCAPE_TARGET;
        else if (position.x >= world.width - radius - 0.001) targetX = -WALL_ESCAPE_TARGET;
        if (position.y <= radius + 0.001) targetY = WALL_ESCAPE_TARGET;
        else if (position.y >= world.height - radius - 0.001) targetY = -WALL_ESCAPE_TARGET;
        return [targetX, targetY, 0];
    }


    // Apply a corrective learning step on every wall-contact frame. The target
    // points back into the play area, so the network learns how to leave the
    // boundary instead of merely learning to stop moving.
    function applyWallContactPunishment(network, position, radius, inEvolution = false,
        memoryBank = null, recordMemory = false, learningTarget = null) {
        const target = wallEscapeTarget(position, radius);
        if (target[0] === 0 && target[1] === 0) return false;


        const inputs = inEvolution ? observeEvolution(position) : observe(position, goal, obstacle);
        if (learningTarget) addTransitionReward(learningTarget, -0.035);
        if (recordMemory && memoryBank) addMemory(memoryBank, inputs, target, "wall-contact");


        if (inEvolution) {
            position.trainingExamples++;
            evolutionTrainingExamples++;
            evolutionTrainingCorrections++;
            evolutionWallPenaltyFrames++;
        } else {
            trainingExamples++;
            trainingCorrections++;
            wallPenaltyFrames++;
        }
        return true;
    }


    function reinforce(isReward) {
        if (!lastInputs) return;
        const target = [
            clamp(lastOutput[0] * (isReward ? 1.08 : 0), -1, 1),
            clamp(lastOutput[1] * (isReward ? 1.08 : 0), -1, 1),
            clamp(lastOutput[2] * 2 - 1, -1, 1)
        ];
        addMemory(memories, lastInputs, target,
            isReward ? "goal-reward" : "behavior-correction");
        addTransitionReward(singlePPOState, isReward ? 1.5 : -1.0);
        trainingExamples++;
        if (isReward) trainingRewards++;
        else trainingCorrections++;
        saveBrain();
        saveMemory();
        updateDiagnostics(lastInputs, brain.forward(lastInputs).output);
        showFeedback(isReward ?
            "Positive reward added to the current PPO learning trajectory." :
            "Negative correction added to the current PPO learning trajectory.");
    }


    function reinforceEvolutionAgent(candidate, isReward) {
        if (!candidate.lastInputs) return;
        const target = [
            clamp(candidate.lastOutput[0] * (isReward ? 1.08 : 0), -1, 1),
            clamp(candidate.lastOutput[1] * (isReward ? 1.08 : 0), -1, 1),
            clamp(candidate.lastOutput[2] * 2 - 1, -1, 1)
        ];
        addMemory(candidate.memories, candidate.lastInputs, target,
            isReward ? "goal-reward" : "behavior-correction");
        addTransitionReward(candidate, isReward ? 1.5 : -0.9);
        candidate.trainingExamples++;
        evolutionTrainingExamples++;
        if (isReward) evolutionTrainingRewards++;
        else evolutionTrainingCorrections++;
    }


    function updateDiagnostics(inputs, output, display = {}) {
        if (typeof ui.updateDiagnostics === "function") {
            ui.updateDiagnostics({
                snapshot: createSnapshot(),
                inputs: Array.isArray(inputs) ? inputs.slice() : [],
                output: Array.isArray(output) ? output.slice() : [],
                display: { ...display }
            });
        }
    }


    // Each evolutionary candidate reads only the world state owned by that
    // candidate. The shared single-agent goal and obstacle are never consulted.
    function observeEvolution(candidate) {
        return createObservation(
            candidate,
            candidate.velocity,
            candidate.environment.goal,
            candidate.environment.obstacle,
            candidate.energy,
            candidate.energyUsageRate
        );
    }


    function reinforceEvolutionAgent(candidate, isReward) {
        if (!candidate.lastInputs) return;
        const target = candidate.lastOutput.map((value, index) =>
            isReward ? (index < 2 ? clamp(value * 1.1, -1, 1) : value) :
                (index < 2 ? 0 : Math.min(value, 0)));
        storeExperience(candidate.memories, candidate.lastInputs, target,
            isReward ? "goal-reward" : "behavior-correction");
        candidate.trainingExamples++;
        evolutionTrainingExamples++;
        if (isReward) evolutionTrainingRewards++;
        else evolutionTrainingCorrections++;
    }


    function createEvolutionEnvironment() {
        // Every candidate receives fresh, separately allocated world objects.
        // Coordinates are sampled independently of every other candidate.
        const obstacleMargin = OBSTACLE_RADIUS + DEFAULT_AGENT_RADIUS + 20;
        const environment = {
            obstacle: {
                x: obstacleMargin + Math.random() * Math.max(1, world.width - obstacleMargin * 2),
                y: obstacleMargin + Math.random() * Math.max(1, world.height - obstacleMargin * 2)
            },
            goal: null
        };
        environment.goal = findGoalPosition(environment.obstacle);
        return environment;
    }


    function randomEvolutionSpawn(environment) {
        const block = environment.obstacle;
        const target = environment.goal;
        for (let attempt = 0; attempt < 240; attempt++) {
            const position = {
                x: DEFAULT_AGENT_RADIUS + 18 +
                    Math.random() * (world.width - (DEFAULT_AGENT_RADIUS + 18) * 2),
                y: DEFAULT_AGENT_RADIUS + 18 +
                    Math.random() * (world.height - (DEFAULT_AGENT_RADIUS + 18) * 2)
            };
            const clearObstacle = distance(position.x - block.x, position.y - block.y) >
                OBSTACLE_RADIUS + DEFAULT_AGENT_RADIUS + 18;
            const clearGoal = distance(position.x - target.x, position.y - target.y) >
                GOAL_RADIUS + DEFAULT_AGENT_RADIUS + 20;
            if (clearObstacle && clearGoal) return position;
        }


        // Deterministic fallback is based only on this candidate's own world.
        const step = 12;
        for (let y = DEFAULT_AGENT_RADIUS + 8; y <= world.height - DEFAULT_AGENT_RADIUS - 8; y += step) {
            for (let x = DEFAULT_AGENT_RADIUS + 8; x <= world.width - DEFAULT_AGENT_RADIUS - 8; x += step) {
                const clearObstacle = distance(x - block.x, y - block.y) >
                    OBSTACLE_RADIUS + DEFAULT_AGENT_RADIUS + 18;
                const clearGoal = distance(x - target.x, y - target.y) >
                    GOAL_RADIUS + DEFAULT_AGENT_RADIUS + 20;
                if (clearObstacle && clearGoal) return { x, y };
            }
        }
        return {
            x: DEFAULT_AGENT_RADIUS + 8,
            y: world.height - DEFAULT_AGENT_RADIUS - 8
        };
    }


    function createEvolutionGeneration(parentNetwork, parentCritic,
        parentMemories = memories, parentEpisodes = episodes, keepElite = false) {
        const population = [];
        for (let index = 0; index < evolutionPopulationSize; index++) {
            const environment = createEvolutionEnvironment();
            const position = randomEvolutionSpawn(environment);
            const mutationRate = index === 0 ? 0 : evolutionMutationRate;
            const mutationStrength = index === 0 ? 0 : evolutionMutationStrength;
            population.push({
                id: index + 1,
                x: position.x,
                y: position.y,
                radius: DEFAULT_AGENT_RADIUS,
                environment,
                velocity: { x: 0, y: 0 },
                energy: MAX_ENERGY,
                energyUsageRate: 0,
                energyDepleted: false,
                goalsCollected: 0,
                network: cloneNetwork(parentNetwork, mutationRate, mutationStrength),
                critic: cloneNetwork(parentCritic, mutationRate, mutationStrength),
                memories: cloneMemories(parentMemories),
                episodes: cloneEpisodes(parentEpisodes),
                rolloutBuffer: [],
                trajectory: [],
                lastTransition: null,
                lastTrajectoryEntry: null,
                pendingReward: 0,
                ppoPending: false,
                ppoToken: 0,
                progressScore: 0,
                lastInputs: null,
                lastOutput: [0, 0, 0],
                lastActivations: null,
                lastMemoryRecall: 0,
                lastMemoryRecallCount: 0,
                stillSince: null,
                touchingWall: false,
                trainingExamples: 0,
                isSuccessor: keepElite && index === 0
            });
        }
        return population;
    }


    function updateEvolutionStatus() {
        if (typeof ui.updateEvolutionStatus === "function") ui.updateEvolutionStatus(createSnapshot());
    }


    function getEvolutionDisplayAgent() {
        if (!evolutionPopulation.length) return null;
        return evolutionPopulation.reduce((best, candidate) => {
            if (!best || candidate.goalsCollected > best.goalsCollected) return candidate;
            if (candidate.goalsCollected < best.goalsCollected) return best;
            if (candidate.progressScore > best.progressScore) return candidate;
            if (candidate.progressScore < best.progressScore) return best;
            if (candidate.energy > best.energy) return candidate;
            if (candidate.energy < best.energy) return best;
            return candidate.id < best.id ? candidate : best;
        }, null);
    }


    function updateEvolutionDiagnostics() {
        const selected = getEvolutionDisplayAgent();
        if (!selected || !selected.lastInputs) return;
        updateDiagnostics(selected.lastInputs, selected.lastOutput, {
            energy: selected.energy, energyDepleted: selected.energyDepleted,
            trainingExamples: evolutionTrainingExamples, rewards: evolutionTrainingRewards,
            corrections: evolutionTrainingCorrections, wallFrames: evolutionWallPenaltyFrames,
            memoryCount: selected.memories.length, memoryRecall: selected.lastMemoryRecall || 0,
            memoryRecallCount: selected.lastMemoryRecallCount || 0,
            memoryBank: selected.memories, episodeBank: selected.episodes
        });
        if (selected.lastActivations && typeof ui.updateNeuralNetwork === "function") {
            ui.updateNeuralNetwork(selected.lastActivations, selected.network.weights);
        }
    }


    function drawEvolution() {
        if (typeof ui.drawEvolution === "function") ui.drawEvolution(createSnapshot());
    }


    function measurePopulationDiversity(population) {
        if (!Array.isArray(population) || population.length < 2) return 0;
        const flattened = population.map(candidate => {
            const values = [];
            for (const matrix of candidate.network.weights) {
                for (const row of matrix) values.push(...row);
            }
            for (const vector of candidate.network.biases) values.push(...vector);
            return values;
        });
        const parameterCount = Math.min(...flattened.map(values => values.length));
        if (!parameterCount) return 0;


        const sampleCount = Math.min(48, parameterCount);
        let totalDeviation = 0;
        for (let sample = 0; sample < sampleCount; sample++) {
            const parameterIndex = Math.floor(sample * parameterCount / sampleCount);
            const mean = flattened.reduce((sum, values) => sum + values[parameterIndex], 0) /
                flattened.length;
            const variance = flattened.reduce((sum, values) =>
                sum + Math.pow(values[parameterIndex] - mean, 2), 0
            ) / flattened.length;
            totalDeviation += Math.sqrt(variance);
        }
        return clamp((totalDeviation / sampleCount) / 0.45, 0, 1);
    }


    function adjustEvolutionMutation(duration, diversity) {
        if (previousGenerationDuration === null) {
            if (diversity < 0.06) {
                evolutionMutationRate *= 1.12;
                evolutionMutationStrength *= 1.08;
            }
        } else if (diversity < 0.055) {
            evolutionMutationRate *= 1.22;
            evolutionMutationStrength *= 1.15;
        } else if (duration < previousGenerationDuration * 0.90) {
            evolutionMutationRate *= 0.92;
            evolutionMutationStrength *= 0.95;
        } else if (duration > previousGenerationDuration * 1.08) {
            evolutionMutationRate *= 1.08;
            evolutionMutationStrength *= 1.06;
        }
        evolutionMutationRate = clamp(evolutionMutationRate, 0.035, 0.20);
        evolutionMutationStrength = clamp(evolutionMutationStrength, 0.045, 0.22);
        previousGenerationDuration = duration;
        evolutionDiversity = diversity;
    }


    function selectEvolutionSuccessor(candidate, now, reason) {
        if (!candidate || evolutionSuccessor) return;


        const generationDuration = Math.max(1, now - evolutionGenerationStartedAt);
        adjustEvolutionMutation(generationDuration, measurePopulationDiversity(evolutionPopulation));
        evolutionPopulation.forEach(agent => { agent.isSuccessor = false; });
        candidate.isSuccessor = true;
        evolutionSuccessor = {
            id: candidate.id,
            goalsCollected: candidate.goalsCollected,
            energy: candidate.energy,
            progressScore: candidate.progressScore,
            network: candidate.network,
            critic: candidate.critic,
            memories: candidate.memories,
            episodes: candidate.episodes,
            candidateRef: candidate
        };
        generationTransitionUntil = now + EVOLUTION_TRANSITION_MS;
        updateEvolutionStatus();


        if (reason === "timeout") {
            showFeedback("Two-minute limit reached. Agent #" + candidate.id +
                " selected with " + candidate.goalsCollected +
                " goal(s). The entire population will be reseeded from this winner.");
        } else {
            showFeedback("Agent #" + candidate.id +
                " collected " + EVOLUTION_GOALS_TO_WIN + " goals! The entire population will be reseeded from its brain and memories.");
        }
    }


    function selectTimeoutWinner(now) {
        if (evolutionSuccessor || !evolutionPopulation.length) return;
        // Resolve timeout ties by measurable progress and remaining energy,
        // not by random selection. This reduces accidental one-off winners.
        const winner = evolutionPopulation.slice().sort((left, right) =>
            right.goalsCollected - left.goalsCollected ||
            right.progressScore - left.progressScore ||
            right.energy - left.energy ||
            left.id - right.id
        )[0];
        selectEvolutionSuccessor(winner, now, "timeout");
    }


    function beginNextEvolutionGeneration(now) {
        if (!evolutionSuccessor) return;


        // Keep an unchanged elite policy and seed every other slot with a
        // lightly mutated descendant of this same winner. Each gets a new
        // independently allocated environment.
        const parent = evolutionSuccessor.network;
        const parentCritic = evolutionSuccessor.critic;
        const parentMemories = evolutionSuccessor.memories;
        const parentEpisodes = evolutionSuccessor.episodes;
        const winningAgentId = evolutionSuccessor.id;


        evolutionGeneration++;
        evolutionPopulation = createEvolutionGeneration(
            parent, parentCritic, parentMemories, parentEpisodes, true
        );
        evolutionLastSuccessorId = 1;
        evolutionSuccessor = null;
        generationTransitionUntil = 0;
        evolutionGenerationStartedAt = now;


        updateEvolutionStatus();
        showFeedback("Generation " + evolutionGeneration +
            " started: Agent #1 is the unchanged elite; the other " +
            (evolutionPopulationSize - 1) + " agents are mutated descendants of winner #" +
            winningAgentId + ". Every agent has a fresh independent world.");
    }


    function updateEvolution(now, frameScale) {
        if (evolutionStopPending) {
            drawEvolution();
            updateEvolutionDiagnostics();
            const retained = evolutionSuccessor && evolutionSuccessor.candidateRef ?
                evolutionSuccessor.candidateRef : getEvolutionDisplayAgent();
            if (retained && retained.ppoPending) return;
            if (retained && retained.rolloutBuffer.length) {
                const finalTraining = schedulePPOTraining(
                    retained, "evolution", evolutionGeneration, true
                );
                if (finalTraining) {
                    dispatchPPOJobs([finalTraining]);
                    return;
                }
                if (retained.ppoPending) return;
            }
            evolutionStopPending = false;
            stopEvolutionMode(true);
            return;
        }


        if (evolutionSuccessor) {
            drawEvolution();
            updateEvolutionDiagnostics();


            // Let any in-flight PPO update finish while the winning agent is
            // frozen. Then train its remaining terminal rollout before cloning
            // it into the next generation.
            const winner = evolutionSuccessor.candidateRef;
            if (winner && !winner.ppoPending && winner.rolloutBuffer.length) {
                const finalTraining = schedulePPOTraining(
                    winner, "evolution", evolutionGeneration, true
                );
                if (finalTraining) {
                    dispatchPPOJobs([finalTraining]);
                    return;
                }
            }
            if (winner && winner.ppoPending) return;
            if (now >= generationTransitionUntil) beginNextEvolutionGeneration(now);
            return;
        }


        for (const candidate of evolutionPopulation) {
            if (candidate.energyDepleted) {
                if (!candidate.ppoPending && candidate.rolloutBuffer.length) {
                    const finalTraining = schedulePPOTraining(
                        candidate, "evolution", evolutionGeneration, true
                    );
                    if (finalTraining) ppoReadyJobs.push(finalTraining);
                }
                continue;
            }


            const environment = candidate.environment;
            const target = environment.goal;
            const block = environment.obstacle;
            const goalDistanceBefore = distance(target.x - candidate.x, target.y - candidate.y);
            const previousX = candidate.x;
            const previousY = candidate.y;
            let additionalEnergyCost = 0;
            let reachedGoal = false;


            candidate.lastInputs = observeEvolution(candidate);
            const decision = planAndSampleAction(
                candidate.network, candidate.critic, candidate.lastInputs,
                candidate, candidate.velocity, environment, candidate.energy,
                candidate.memories, candidate.episodes
            );
            candidate.lastOutput = decision.action;
            candidate.lastMemoryRecall = decision.memoryRecall.strength;
            candidate.lastMemoryRecallCount = decision.memoryRecall.count;
            candidate.lastActivations = decision.activations;
            const transition = createPPOTransition(candidate.lastInputs, decision, candidate);


            const brake = clamp(candidate.lastOutput[2], 0, 1);
            const damping = 1 - brake * BRAKING_STRENGTH;
            candidate.velocity.x = clamp(
                (candidate.velocity.x + candidate.lastOutput[0] * ACCELERATION) * damping,
                -SPEED, SPEED
            );
            candidate.velocity.y = clamp(
                (candidate.velocity.y + candidate.lastOutput[1] * ACCELERATION) * damping,
                -SPEED, SPEED
            );
            candidate.x += candidate.velocity.x;
            candidate.y += candidate.velocity.y;


            if (candidate.x < candidate.radius || candidate.x > world.width - candidate.radius) {
                candidate.x = clamp(candidate.x, candidate.radius, world.width - candidate.radius);
                candidate.velocity.x = 0;
            }
            if (candidate.y < candidate.radius || candidate.y > world.height - candidate.radius) {
                candidate.y = clamp(candidate.y, candidate.radius, world.height - candidate.radius);
                candidate.velocity.y = 0;
            }


            const wallContact = candidate.x <= candidate.radius + 0.001 ||
                candidate.x >= world.width - candidate.radius - 0.001 ||
                candidate.y <= candidate.radius + 0.001 ||
                candidate.y >= world.height - candidate.radius - 0.001;
            if (wallContact) {
                applyWallContactPunishment(
                    candidate.network, candidate, candidate.radius, true,
                    candidate.memories, !candidate.touchingWall, candidate
                );
            }
            if (wallContact && !candidate.touchingWall) {
                candidate.stillSince = now;
                additionalEnergyCost += ENERGY_WALL_IMPACT_COST;
            }
            candidate.touchingWall = wallContact;


            const moved = distance(candidate.x - previousX, candidate.y - previousY) > MOVEMENT_THRESHOLD;
            if (moved) candidate.stillSince = null;
            else if (candidate.stillSince === null) candidate.stillSince = now;
            else if (now - candidate.stillSince >= STILL_PUNISHMENT_MS) {
                if (!wallContact) reinforceEvolutionAgent(candidate, false);
                candidate.stillSince = now;
            }


            if (distance(target.x - candidate.x, target.y - candidate.y) <
                candidate.radius + GOAL_RADIUS) {
                reinforceEvolutionAgent(candidate, true);
                addTransitionReward(candidate, 3.5);
                candidate.energy = MAX_ENERGY;
                candidate.goalsCollected++;
                candidate.stillSince = null;
                reachedGoal = true;


                if (candidate.goalsCollected < EVOLUTION_GOALS_TO_WIN) {
                    environment.goal = findGoalPosition(block, candidate);
                }
            }


            if (distance(block.x - candidate.x, block.y - candidate.y) <
                candidate.radius + OBSTACLE_RADIUS) {
                const dx = candidate.x - block.x || 1;
                const dy = candidate.y - block.y || 1;
                const magnitude = Math.hypot(dx, dy) || 1;
                const nx = dx / magnitude;
                const ny = dy / magnitude;
                candidate.x = clamp(candidate.x + nx * 26, candidate.radius, world.width - candidate.radius);
                candidate.y = clamp(candidate.y + ny * 26, candidate.radius, world.height - candidate.radius);
                candidate.velocity.x += nx * 0.5;
                candidate.velocity.y += ny * 0.5;
                reinforceEvolutionAgent(candidate, false);
                candidate.stillSince = now;
                additionalEnergyCost += ENERGY_OBSTACLE_IMPACT_COST;
            }


            const distanceMoved = distance(candidate.x - previousX, candidate.y - previousY);
            const thrustCost = (Math.abs(candidate.lastOutput[0]) + Math.abs(candidate.lastOutput[1])) *
                ENERGY_THRUST_COST * frameScale;
            const brakingCost = clamp(candidate.lastOutput[2], 0, 1) * ENERGY_BRAKING_COST * frameScale;
            const movementCost = distanceMoved * ENERGY_MOVEMENT_COST;
            const energyBefore = candidate.energy;
            const energySpent = movementCost + thrustCost + brakingCost + additionalEnergyCost;
            const instantUsageRate = clamp(energySpent / 0.18, 0, 1);
            candidate.energyUsageRate = clamp(
                candidate.energyUsageRate * 0.65 + instantUsageRate * 0.35, 0, 1
            );
            candidate.energy = Math.max(0, candidate.energy - energySpent);


            // Use the goal object captured at the beginning of this step so
            // a newly spawned goal cannot distort the terminal progress reward.
            const goalDistanceAfter = distance(target.x - candidate.x, target.y - candidate.y);
            const progressReward = clamp((goalDistanceBefore - goalDistanceAfter) / 90, -0.10, 0.10);
            transition.reward = clamp(
                transition.reward + progressReward - (energyBefore - candidate.energy) * 0.003,
                -8, 8
            );
            candidate.progressScore += progressReward;
            if (candidate.lastTrajectoryEntry) {
                candidate.lastTrajectoryEntry.reward = transition.reward;
            }


            if (reachedGoal) {
                rememberEpisode(candidate.episodes, candidate.trajectory, "goal");
                candidate.trajectory = [];
            }


            if (candidate.energy <= 0 && candidate.goalsCollected === 0 && !candidate.energyDepleted) {
                candidate.energyDepleted = true;
                candidate.velocity = { x: 0, y: 0 };
                transition.reward = clamp(transition.reward - 4.0, -8, 8);
                transition.done = true;
                if (candidate.lastTrajectoryEntry) {
                    candidate.lastTrajectoryEntry.reward = transition.reward;
                }
                const depletedInputs = candidate.lastInputs.slice();
                depletedInputs[9] = 0;
                const reducedAction = [
                    clamp(candidate.lastOutput[0] * 0.1, -1, 1),
                    clamp(candidate.lastOutput[1] * 0.1, -1, 1),
                    clamp(candidate.lastOutput[2] * 2 - 1, -1, 1)
                ];
                addMemory(candidate.memories, depletedInputs, reducedAction, "energy-depletion");
                rememberEpisode(candidate.episodes, candidate.trajectory, "energy-depletion");
                candidate.trainingExamples++;
                evolutionTrainingExamples++;
                candidate.lastInputs = depletedInputs;
            }


            const forceTraining = candidate.energyDepleted ||
                (reachedGoal && candidate.goalsCollected >= EVOLUTION_GOALS_TO_WIN);
            const training = schedulePPOTraining(
                candidate, "evolution", evolutionGeneration, forceTraining
            );
            if (training) ppoReadyJobs.push(training);


            if (reachedGoal && candidate.goalsCollected >= EVOLUTION_GOALS_TO_WIN) {
                selectEvolutionSuccessor(candidate, now, "goals");
                break;
            }
        }


        if (ppoReadyJobs.length) {
            dispatchPPOJobs(ppoReadyJobs.splice(0));
        }


        if (!evolutionSuccessor &&
            now - evolutionGenerationStartedAt >= EVOLUTION_GENERATION_TIMEOUT_MS) {
            selectTimeoutWinner(now);
        }


        updateEvolutionStatus();
        updateEvolutionDiagnostics();
        drawEvolution();
    }


    function startEvolutionMode() {
        if (evolutionStopPending) return;
        evolutionStopPending = false;
        evolutionaryMode = true;
        // The single-agent world remains untouched while evolution is running.
        evolutionGeneration = 1;
        evolutionTrainingExamples = 0;
        evolutionTrainingRewards = 0;
        evolutionTrainingCorrections = 0;
        evolutionWallPenaltyFrames = 0;
        evolutionSuccessor = null;
        evolutionLastSuccessorId = null;
        evolutionGenerationStartedAt = getNow();
        evolutionMutationRate = EVOLUTION_MUTATION_RATE;
        evolutionMutationStrength = EVOLUTION_MUTATION_STRENGTH;
        evolutionPopulation = createEvolutionGeneration(brain, critic, memories, episodes);
        if (typeof ui.setEvolutionViewEnabled === "function") ui.setEvolutionViewEnabled(true);
        if (typeof ui.setEvolutionModeState === "function") ui.setEvolutionModeState(true);
        drawEvolution();
        updateEvolutionStatus();
        showFeedback("Evolutionary Training started: " + evolutionPopulationSize +
            " agents are running in independent environments.");
    }


    function stopEvolutionMode(finalized = false) {
        const retainedAgent = evolutionSuccessor || getEvolutionDisplayAgent();
        const retainedCandidate = retainedAgent && retainedAgent.candidateRef ?
            retainedAgent.candidateRef : retainedAgent;


        if (!finalized && retainedCandidate &&
            (retainedCandidate.ppoPending || retainedCandidate.rolloutBuffer.length > 0)) {
            evolutionStopPending = true;
            if (typeof ui.setEvolutionStopPending === "function") ui.setEvolutionStopPending(true);
            showFeedback("Finishing the leading agent's pending learning update before stopping evolution.");
            return;
        }


        if (retainedCandidate && retainedCandidate.network &&
            Array.isArray(retainedCandidate.memories)) {
            if (!retainedCandidate.ppoPending && retainedCandidate.rolloutBuffer.length) {
                try {
                    const trained = learning.trainPPO(
                        retainedCandidate.network.export(),
                        retainedCandidate.critic.export(),
                        retainedCandidate.rolloutBuffer.splice(0),
                        {
                            policyRate: PPO_POLICY_RATE,
                            valueRate: PPO_VALUE_RATE,
                            policyStandardDeviation: PPO_POLICY_STD,
                            epochs: PPO_EPOCHS,
                            minibatchSize: PPO_MINIBATCH_SIZE
                        }
                    );
                    retainedCandidate.network.applyTrainedModel(trained.policy);
                    retainedCandidate.critic.applyTrainedModel(trained.critic);
                    updatePPOStatistics(trained.stats);
                } catch (error) {
                    console.warn("Final evolutionary PPO update could not be applied.", error);
                }
            }
            brain = cloneNetwork(retainedCandidate.network);
            critic = cloneNetwork(retainedCandidate.critic);
            memories = cloneMemories(retainedCandidate.memories);
            episodes = cloneEpisodes(retainedCandidate.episodes);
            saveBrain();
            saveMemory();
            saveEpisodes();
        }


        // Drop queued updates for discarded descendants. Keep only any
        // non-evolution work and the retained candidate's own records.
        const keepTrainingRecord = record =>
            record.kind !== "evolution" || record.target === retainedCandidate;
        ppoFallbackQueue = ppoFallbackQueue.filter(record => {
            const keep = keepTrainingRecord(record);
            if (!keep && record.target && record.target.ppoToken === record.token) {
                record.target.ppoPending = false;
            }
            return keep;
        });
        ppoWorkerQueue = ppoWorkerQueue.filter(record => {
            const keep = keepTrainingRecord(record);
            if (!keep && record.target && record.target.ppoToken === record.token) {
                record.target.ppoPending = false;
            }
            return keep;
        });


        evolutionStopPending = false;
        evolutionaryMode = false;
        if (typeof ui.setEvolutionViewEnabled === "function") ui.setEvolutionViewEnabled(false);
        if (typeof ui.setEvolutionModeState === "function") ui.setEvolutionModeState(false);
        if (typeof ui.setEvolutionStopPending === "function") ui.setEvolutionStopPending(false);
        evolutionPopulation = [];
        evolutionSuccessor = null;
        showFeedback(retainedCandidate ?
            "Evolution stopped. The leading agent's actor, critic, and memories were kept." :
            "Evolutionary Training stopped. Individual training resumed.");
    }




    function draw() {
        if (typeof ui.drawWorld === "function") ui.drawWorld(createSnapshot());
    }


    function findGoalPosition(block, avoidPosition = null) {
        const margin = GOAL_RADIUS + 4;
        const minX = margin;
        const maxX = world.width - margin;
        const minY = margin;
        const maxY = world.height - margin;
        const minObstacleDistance = OBSTACLE_RADIUS + GOAL_RADIUS + 8;
        const minAgentDistance = avoidPosition ?
            (avoidPosition.radius || DEFAULT_AGENT_RADIUS) + GOAL_RADIUS + 10 : 0;
        const isSafe = point =>
            distance(point.x - block.x, point.y - block.y) > minObstacleDistance &&
            (!avoidPosition || distance(point.x - avoidPosition.x, point.y - avoidPosition.y) > minAgentDistance);


        // Random placement keeps normal gameplay varied while rejecting every
        // position that would put any part of the goal inside the obstacle.
        for (let attempt = 0; attempt < 300; attempt++) {
            const candidate = {
                x: minX + Math.random() * Math.max(0, maxX - minX),
                y: minY + Math.random() * Math.max(0, maxY - minY)
            };
            if (isSafe(candidate)) return candidate;
        }


        // Deterministic fallback: scan the playable area so placement does
        // not silently fall back to an overlapping/random invalid position.
        const step = Math.max(8, Math.min(16, Math.floor(Math.min(world.width, world.height) / 30)));
        for (let y = minY; y <= maxY; y += step) {
            for (let x = minX; x <= maxX; x += step) {
                const candidate = { x, y };
                if (isSafe(candidate)) return candidate;
            }
        }


        // On the normal 500×500 playfield this branch is unreachable. Keep a
        // safe corner fallback for unusually small/resized canvases.
        const corners = [
            { x: minX, y: minY }, { x: maxX, y: minY },
            { x: minX, y: maxY }, { x: maxX, y: maxY }
        ];
        const safeCorner = corners.find(isSafe);
        if (safeCorner) return safeCorner;


        // Select the corner with the most clearance if unusual world sizes
        // make all strict candidates unavailable.
        return corners.reduce((best, point) => {
            const obstacleClearance = distance(point.x - block.x, point.y - block.y);
            const agentClearance = avoidPosition ?
                distance(point.x - avoidPosition.x, point.y - avoidPosition.y) : Infinity;
            const score = Math.min(obstacleClearance, agentClearance);
            const bestScore = Math.min(
                distance(best.x - block.x, best.y - block.y),
                avoidPosition ? distance(best.x - avoidPosition.x, best.y - avoidPosition.y) : Infinity
            );
            return score > bestScore ? point : best;
        }, corners[0]);
    }


    function moveGoal() {
        goal = findGoalPosition(obstacle, agent);
        return goal;
    }


    // Manual punishment also relocates the agent. Automatic punishments never
    // call this function, so wall/obstacle/stillness corrections leave it in place.
    function teleportAfterManualPunishment() {
        const oldX = agent.x;
        const oldY = agent.y;
        const minTravel = Math.max(60, Math.min(world.width, world.height) * 0.2);
        let nextPosition = null;


        for (let attempt = 0; attempt < 60; attempt++) {
            const candidate = {
                x: agent.radius + Math.random() * (world.width - agent.radius * 2),
                y: agent.radius + Math.random() * (world.height - agent.radius * 2)
            };
            const farEnough = distance(candidate.x - oldX, candidate.y - oldY) >= minTravel;
            const clearOfObstacle = distance(candidate.x - obstacle.x, candidate.y - obstacle.y) >
                agent.radius + OBSTACLE_RADIUS + 12;
            const clearOfGoal = distance(candidate.x - goal.x, candidate.y - goal.y) >
                agent.radius + GOAL_RADIUS + 12;


            if (farEnough && clearOfObstacle && clearOfGoal) {
                nextPosition = candidate;
                break;
            }
        }


        // A fallback still guarantees a different valid world position.
        if (!nextPosition) {
            nextPosition = {
                x: clamp(world.width - oldX, agent.radius, world.width - agent.radius),
                y: clamp(world.height - oldY, agent.radius, world.height - agent.radius)
            };
            if (distance(nextPosition.x - oldX, nextPosition.y - oldY) < 1) {
                nextPosition.x = clamp(oldX + minTravel, agent.radius, world.width - agent.radius);
            }
        }


        agent.x = nextPosition.x;
        agent.y = nextPosition.y;
        velocity = { x: 0, y: 0 };
        stillSince = null;
        touchingWall = false;
    }


    function applyEnergyDepletionPunishment() {
        if (energyDepleted) return;
        energy = 0;
        energyDepleted = true;
        velocity = { x: 0, y: 0 };
        addTransitionReward(singlePPOState, -4.5);
        if (singlePPOState.lastTransition) singlePPOState.lastTransition.done = true;


        if (lastInputs) {
            const depletionInputs = lastInputs.slice();
            depletionInputs[9] = 0;
            const reducedAction = [
                clamp(lastOutput[0] * 0.1, -1, 1),
                clamp(lastOutput[1] * 0.1, -1, 1),
                clamp(lastOutput[2] * 2 - 1, -1, 1)
            ];
            addMemory(memories, depletionInputs, reducedAction, "energy-depletion");
            rememberEpisode(episodes, singlePPOState.trajectory, "energy-depletion");
            singlePPOState.trajectory = [];
            lastInputs = depletionInputs;
            trainingExamples++;
            saveBrain();
            saveMemory();
            saveEpisodes();
        }


        updateDiagnostics(lastInputs || Array(INPUTS).fill(0), lastOutput);
        showFeedback("Energy depleted. The final action was marked as a terminal PPO experience.");
    }


    function consumeEnergy(distanceMoved, frameScale, additionalCost = 0) {
        if (energyDepleted) return;
        const thrustCost = (Math.abs(lastOutput[0]) + Math.abs(lastOutput[1])) *
            ENERGY_THRUST_COST * frameScale;
        const brakingCost = clamp(lastOutput[2], 0, 1) * ENERGY_BRAKING_COST * frameScale;
        const movementCost = distanceMoved * ENERGY_MOVEMENT_COST;
        const energySpent = movementCost + thrustCost + brakingCost + additionalCost;
        const instantUsageRate = clamp(energySpent / 0.18, 0, 1);
        energyUsageRate = clamp(energyUsageRate * 0.65 + instantUsageRate * 0.35, 0, 1);
        energy = Math.max(0, energy - energySpent);
        if (energy <= 0) applyEnergyDepletionPunishment();
    }


    function simulateStep(now) {
        const frameScale = 1;
        if (evolutionaryMode) {
            updateEvolution(now, frameScale);
            return;
        }


        if (energyDepleted) {
            if (!singlePPOState.ppoPending && singlePPOState.rolloutBuffer.length) {
                const finalTraining = schedulePPOTraining(singlePPOState, "single", 0, true);
                if (finalTraining) dispatchPPOJobs([finalTraining]);
            }
            if (lastInputs) updateDiagnostics(lastInputs, lastOutput);
            draw();
            saveCountdown++;
            if (saveCountdown >= 300) {
                saveCountdown = 0;
                saveBrain();
            }
            return;
        }


        const previousX = agent.x;
        const previousY = agent.y;
        const goalDistanceBefore = distance(goal.x - agent.x, goal.y - agent.y);
        let additionalEnergyCost = 0;
        lastInputs = observe(agent, goal, obstacle);
        const decision = planAndSampleAction(
            brain, critic, lastInputs, agent, velocity,
            { goal, obstacle }, energy, memories, episodes
        );
        lastOutput = decision.action;
        lastMemoryRecall = decision.memoryRecall.strength;
        lastMemoryRecallCount = decision.memoryRecall.count;
        lastPlannedMean = decision.mean || lastPlannedMean;
        lastTransition = createPPOTransition(lastInputs, decision, singlePPOState);


        if (typeof ui.updateNeuralNetwork === "function") {
            ui.updateNeuralNetwork(decision.activations, brain.weights);
        }


        const brake = clamp(lastOutput[2], 0, 1);
        const damping = 1 - brake * BRAKING_STRENGTH;
        velocity.x = clamp((velocity.x + lastOutput[0] * ACCELERATION) * damping, -SPEED, SPEED);
        velocity.y = clamp((velocity.y + lastOutput[1] * ACCELERATION) * damping, -SPEED, SPEED);
        agent.x += velocity.x;
        agent.y += velocity.y;
        if (agent.x < agent.radius || agent.x > world.width - agent.radius) {
            agent.x = clamp(agent.x, agent.radius, world.width - agent.radius);
            velocity.x = 0;
        }
        if (agent.y < agent.radius || agent.y > world.height - agent.radius) {
            agent.y = clamp(agent.y, agent.radius, world.height - agent.radius);
            velocity.y = 0;
        }


        const wallContact = agent.x <= agent.radius + 0.001 ||
            agent.x >= world.width - agent.radius - 0.001 ||
            agent.y <= agent.radius + 0.001 ||
            agent.y >= world.height - agent.radius - 0.001;
        if (wallContact) {
            applyWallContactPunishment(
                brain, agent, agent.radius, false, memories, !touchingWall, singlePPOState
            );
            if (!touchingWall) saveMemory();
        }
        if (wallContact && !touchingWall) {
            stillSince = now;
            additionalEnergyCost += ENERGY_WALL_IMPACT_COST;
        }
        touchingWall = wallContact;


        const movedThisFrame = distance(agent.x - previousX, agent.y - previousY) > MOVEMENT_THRESHOLD;
        if (movedThisFrame) {
            stillSince = null;
        } else if (stillSince === null) {
            stillSince = now;
        } else if (now - stillSince >= STILL_PUNISHMENT_MS) {
            if (!wallContact) reinforce(false);
            stillSince = now;
        }


        const goalDistanceAfterMovement = distance(goal.x - agent.x, goal.y - agent.y);
        if (goalDistanceAfterMovement < agent.radius + GOAL_RADIUS) {
            reinforce(true);
            addTransitionReward(singlePPOState, 3.5);
            energy = MAX_ENERGY;
            energyDepleted = false;
            rememberEpisode(episodes, singlePPOState.trajectory, "goal");
            singlePPOState.trajectory = [];
            saveEpisodes();
            moveGoal();
            stillSince = null;
        }


        if (distance(obstacle.x - agent.x, obstacle.y - agent.y) < agent.radius + OBSTACLE_RADIUS) {
            const reboundX = agent.x - obstacle.x || 1;
            const reboundY = agent.y - obstacle.y || 1;
            const magnitude = Math.hypot(reboundX, reboundY) || 1;
            const push = 26;
            const normalX = reboundX / magnitude;
            const normalY = reboundY / magnitude;
            agent.x = clamp(agent.x + normalX * push, agent.radius, world.width - agent.radius);
            agent.y = clamp(agent.y + normalY * push, agent.radius, world.height - agent.radius);
            velocity.x += normalX * 0.5;
            velocity.y += normalY * 0.5;
            reinforce(false);
            stillSince = now;
            additionalEnergyCost += ENERGY_OBSTACLE_IMPACT_COST;
        }


        const distanceMoved = distance(agent.x - previousX, agent.y - previousY);
        const energyBefore = energy;
        consumeEnergy(distanceMoved, frameScale, additionalEnergyCost);
        const progressReward = clamp((goalDistanceBefore - goalDistanceAfterMovement) / 90, -0.10, 0.10);
        lastTransition.reward = clamp(
            lastTransition.reward + progressReward - (energyBefore - energy) * 0.003,
            -8, 8
        );
        if (singlePPOState.lastTrajectoryEntry) {
            singlePPOState.lastTrajectoryEntry.reward = lastTransition.reward;
        }


        const training = schedulePPOTraining(
            singlePPOState, "single", 0, energyDepleted
        );
        if (training) dispatchPPOJobs([training]);


        updateDiagnostics(lastInputs, lastOutput);
        draw();


        saveCountdown++;
        if (saveCountdown >= 300) {
            saveCountdown = 0;
            saveBrain();
        }
    }


    // A fixed 60 Hz simulation step keeps movement, energy costs, and learning
    // independent of variable frame-scheduler delivery rates.
    function update(timestamp) {
        if (!running) return;
        frameHandle = null;
        const frameNow = Number.isFinite(timestamp) ? timestamp : getNow();
        if (lastFrameTimestamp === null) {
            lastFrameTimestamp = frameNow;
            simulationClock = frameNow - SIMULATION_STEP_MS;
            simulationAccumulator = SIMULATION_STEP_MS;
        } else {
            const elapsed = clamp(frameNow - lastFrameTimestamp, 0, MAX_FRAME_DELTA_MS);
            lastFrameTimestamp = frameNow;
            simulationAccumulator += elapsed;
        }


        let steps = 0;
        while (simulationAccumulator >= SIMULATION_STEP_MS &&
            steps < MAX_SIMULATION_STEPS_PER_FRAME) {
            simulationClock += SIMULATION_STEP_MS;
            simulateStep(simulationClock);
            simulationAccumulator -= SIMULATION_STEP_MS;
            steps++;
        }


        // Drop excess backlog after a long tab stall instead of entering an
        // endless catch-up loop that would freeze the page.
        if (steps === MAX_SIMULATION_STEPS_PER_FRAME &&
            simulationAccumulator >= SIMULATION_STEP_MS) {
            simulationAccumulator = 0;
        }


        drainFallbackPPOQueue(
            evolutionaryMode && evolutionPopulationSize > 100 ? 1 : 2
        );
        if (running) frameHandle = requestFrame(update);
    }


    function clearMemory() {
        memories = [];
        episodes = [];
        lastMemoryRecall = 0;
        lastMemoryRecallCount = 0;
        saveMemory();
        saveEpisodes();
        updateDiagnostics(lastInputs || Array(INPUTS).fill(0), lastOutput || [0, 0, 0]);
        showFeedback("State and episodic memories cleared. The actor and critic were kept.");
    }


    function resetBrainAndMemory() {
        try {
            [
                STORAGE_KEY, LEGACY_STORAGE_KEY_V7, LEGACY_STORAGE_KEY_V6,
                LEGACY_STORAGE_KEY_V5, MEMORY_STORAGE_KEY,
                LEGACY_MEMORY_STORAGE_KEY_V1, EPISODE_STORAGE_KEY,
                LEGACY_EPISODE_STORAGE_KEY_V1
            ].forEach(key => storage.removeItem(key));
        } catch (error) {
            console.warn("Could not clear saved model.", error);
        }
        singlePPOState.ppoToken++;
        singlePPOState.ppoPending = false;
        singlePPOState.rolloutBuffer = [];
        singlePPOState.trajectory = [];
        singlePPOState.lastTransition = null;
        singlePPOState.lastTrajectoryEntry = null;
        singlePPOState.pendingReward = 0;
        lastTransition = null;
        brain = new NeuralNetwork(null);
        critic = new NeuralNetwork(null, 1);
        memories = [];
        episodes = [];
        trainingExamples = 0;
        trainingRewards = 0;
        trainingCorrections = 0;
        wallPenaltyFrames = 0;
        ppoUpdateCount = 0;
        ppoRewardAverage = 0;
        agent = { x: 90, y: 410, radius: DEFAULT_AGENT_RADIUS };
        velocity = { x: 0, y: 0 };
        energy = MAX_ENERGY;
        energyUsageRate = 0;
        energyDepleted = false;
        lastFrameTimestamp = null;
        simulationAccumulator = 0;
        stillSince = null;
        touchingWall = false;
        setAgentRadius(DEFAULT_AGENT_RADIUS);
        moveGoal();
        saveBrain();
        saveMemory();
        saveEpisodes();
        updateEvolutionStatus();
        draw();
        showFeedback("Fresh actor, critic, state memory, and episodic memory created.");
    }


    function exportModelData() {
        const selectedEvolutionAgent = evolutionaryMode ?
            (evolutionSuccessor || getEvolutionDisplayAgent()) : null;
        const exportSource = selectedEvolutionAgent ? selectedEvolutionAgent.network : brain;
        const exportCritic = selectedEvolutionAgent ? selectedEvolutionAgent.critic : critic;
        const exportMemories = selectedEvolutionAgent && selectedEvolutionAgent.memories ?
            selectedEvolutionAgent.memories : memories;
        const exportEpisodes = selectedEvolutionAgent && selectedEvolutionAgent.episodes ?
            selectedEvolutionAgent.episodes : episodes;
        return {
            ...exportSource.export(),
            critic: exportCritic.export(),
            memoryVersion: 3,
            memory: cloneMemories(exportMemories),
            episodes: cloneEpisodes(exportEpisodes)
        };
    }


    function importModelData(data, fileName = "imported file") {
        if (!data || typeof data !== "object" || Array.isArray(data)) {
            throw new Error("The selected file does not contain a valid Agent Five model.");
        }
        const validator = new NeuralNetwork(null);
        const importedModel = validator.isValid(data) ? data : migrateLegacyBrain(data);
        if (!importedModel || importedModel.modelType !== "policy") {
            throw new Error("This JSON is not a compatible Agent Five v8 actor or a migratable v5/v6/v7 brain.");
        }
        const criticValidator = new NeuralNetwork(null, 1);
        const importedCriticModel = criticValidator.isValid(data.critic) ?
            data.critic : migrateLegacyBrain(data.critic);
        const importedCritic = importedCriticModel && importedCriticModel.modelType === "critic" ?
            new NeuralNetwork(importedCriticModel, 1) : new NeuralNetwork(null, 1);
        const remapLegacyMemoryInputs = data.version === 7;
        const includedMemory = Object.prototype.hasOwnProperty.call(data, "memory");
        const includedEpisodes = Object.prototype.hasOwnProperty.call(data, "episodes");
        let importedMemories = memories;
        let importedEpisodes = episodes;


        if (includedMemory) {
            if (!Array.isArray(data.memory) || data.memory.length > MEMORY_CAPACITY * 4) {
                throw new Error("The included state memory section is invalid or too large.");
            }
            importedMemories = data.memory.map(item => normalizeMemoryItem(item, remapLegacyMemoryInputs));
            if (importedMemories.some(item => !item)) {
                throw new Error("The included state memory contains an invalid experience.");
            }
            importedMemories = normalizeMemoryCollection(importedMemories);
        }
        if (includedEpisodes) {
            if (!Array.isArray(data.episodes) || data.episodes.length > EPISODE_MEMORY_CAPACITY * 4) {
                throw new Error("The included episodic memory section is invalid or too large.");
            }
            importedEpisodes = data.episodes.map(item => normalizeEpisode(item, remapLegacyMemoryInputs));
            if (importedEpisodes.some(item => !item)) {
                throw new Error("The included episodic memory contains an invalid sequence.");
            }
            importedEpisodes = cloneEpisodes(importedEpisodes);
        }


        singlePPOState.ppoToken++;
        singlePPOState.ppoPending = false;
        singlePPOState.rolloutBuffer = [];
        singlePPOState.trajectory = [];
        singlePPOState.pendingReward = 0;
        singlePPOState.lastTransition = null;
        singlePPOState.lastTrajectoryEntry = null;
        lastTransition = null;
        brain = new NeuralNetwork(importedModel);
        critic = importedCritic;
        if (includedMemory) {
            memories = importedMemories;
            lastMemoryRecall = 0;
            lastMemoryRecallCount = 0;
            saveMemory();
        }
        if (includedEpisodes) {
            episodes = importedEpisodes;
            saveEpisodes();
        }
        trainingExamples = 0;
        saveBrain();
        const importedParts = [
            "actor and critic",
            includedMemory ? memories.length + " state memories" : null,
            includedEpisodes ? episodes.length + " episodic sequences" : null
        ].filter(Boolean).join(", ");
        showFeedback("Loaded " + importedParts + " from " + fileName + ".");
        updateDiagnostics(lastInputs || Array(INPUTS).fill(0), lastOutput || [0, 0, 0]);
        return importedParts;
    }


    function setPopulationSize(value) {
        const requestedSize = Math.round(Number(value));
        evolutionPopulationSize = clamp(
            Number.isFinite(requestedSize) ? requestedSize : DEFAULT_EVOLUTION_POPULATION_SIZE,
            MIN_EVOLUTION_POPULATION_SIZE, MAX_EVOLUTION_POPULATION_SIZE
        );
        updateEvolutionStatus();
        showFeedback("Evolutionary population set to " + evolutionPopulationSize +
            ". It applies to the next generation.");
        return evolutionPopulationSize;
    }


    function start() {
        if (running) return;
        running = true;
        lastFrameTimestamp = null;
        simulationAccumulator = 0;
        setAgentRadius(DEFAULT_AGENT_RADIUS);
        if (typeof ui.setModelStatus === "function") ui.setModelStatus("Ready");
        initializeLearningWorker();
        update(getNow());
    }


    function stop() {
        running = false;
        if (frameHandle !== null) {
            try { cancelFrame(frameHandle); } catch (error) {}
            frameHandle = null;
        }
        if (learningWorker) {
            try { learningWorker.terminate(); } catch (error) {}
            learningWorker = null;
        }
        ppoWorkerBusy = false;
        ppoPendingJobs.clear();
        ppoWorkerQueue = [];
        ppoFallbackQueue = [];
    }


    function setEvolutionMode(enabled) {
        if (enabled) startEvolutionMode();
        else stopEvolutionMode();
        return evolutionaryMode;
    }


    function applyManualCorrection() {
        reinforce(false);
        teleportAfterManualPunishment();
        showFeedback("Manual punishment applied. Agent relocated.");
    }


    const api = {
        start, stop, getSnapshot: createSnapshot, setPopulationSize, setEvolutionMode, setAgentRadius,
        reward: () => reinforce(true), correct: applyManualCorrection, clearMemory,
        reset: resetBrainAndMemory, exportData: exportModelData, importData: importModelData
    };


    setAgentRadius(DEFAULT_AGENT_RADIUS);
    if (typeof ui.setModelStatus === "function") ui.setModelStatus("Ready");
    updateEvolutionStatus();
    draw();
    return Object.freeze(api);
    }


    root.AgentFiveCore = Object.freeze({ create: createAgentFive });
})(globalThis);
