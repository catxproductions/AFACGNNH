/* Agent-Five-Nexus-1.0.0 language subsystem.
 * Pure JavaScript tokenizer and compact GRU decoder; no DOM or external dependencies.
 */
(function (root) {
    "use strict";

    const SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<USER>", "<AGENT>", "<ENV>", "<TOOL>", "<RES>"];
    const SEED_VOCAB = [
        "Agent", "Five", "Nexus", "I", "am", "at", "the", "goal", "obstacle", "energy",
        "remaining", "position", "coordinates", "environment", "report", "waypoint", "set",
        "carefully", "safely", "navigate", "moving", "waiting", "monitoring", "risk",
        "low", "high", "clearance", "distance", "velocity", "tool", "result", "success",
        "error", "understood", "instruction", "can", "help", "please", "toward", "from",
        "with", "and", "is", "are", "was", "will", "continue", "avoid", "collect",
        "goals", "energy", "level", "calculated", "memory", "found", "no", "data",
        ".", ",", ":", ";", "(", ")", "-", "=", "?"
    ];

    class NexusTokenizer {
        constructor(saved) {
            this.vocabulary = SPECIAL_TOKENS.concat(SEED_VOCAB);
            this.tokenToId = new Map(this.vocabulary.map((token, index) => [token, index]));
            if (saved && Array.isArray(saved.vocabulary) &&
                saved.vocabulary.slice(0, SPECIAL_TOKENS.length).every((token, index) => token === SPECIAL_TOKENS[index])) {
                this.vocabulary = saved.vocabulary.slice();
                this.tokenToId = new Map(this.vocabulary.map((token, index) => [token, index]));
            }
        }

        add(token) {
            if (!this.tokenToId.has(token)) {
                this.tokenToId.set(token, this.vocabulary.length);
                this.vocabulary.push(token);
            }
            return this.tokenToId.get(token);
        }

        tokenize(text) {
            return String(text || "").match(/<[^>]+>|[A-Za-z0-9_']+|[^\sA-Za-z0-9_]/g) || [];
        }

        encode(text, addTokens = true) {
            return this.tokenize(text).map(token => {
                if (this.tokenToId.has(token)) return this.tokenToId.get(token);
                return addTokens ? this.add(token) : 1;
            });
        }

        decode(ids) {
            return (Array.isArray(ids) ? ids : []).map(id =>
                this.vocabulary[id] || SPECIAL_TOKENS[1]
            ).filter(token => !SPECIAL_TOKENS.includes(token)).join(" ")
                .replace(/\s+([.,!?;:])/g, "$1");
        }

        export() {
            return { version: 1, vocabulary: this.vocabulary.slice() };
        }
    }

    function sigmoid(value) {
        return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, value))));
    }

    function makeVector(length, scale, seed) {
        const result = new Float32Array(length);
        let state = seed >>> 0;
        for (let i = 0; i < length; i++) {
            state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
            result[i] = ((state >>> 0) / 4294967295 * 2 - 1) * scale;
        }
        return result;
    }

    class NexusDecoderNetwork {
        constructor(options = {}) {
            this.embeddingSize = 24;
            this.hiddenSize = 32;
            this.contextSize = 8;
            this.trainedTokens = 0;
            this.tokenizer = options.tokenizer || new NexusTokenizer(options.tokenizerData);
            const saved = options.saved;
            if (saved && this.import(saved)) return;
            this.initialize(this.tokenizer.vocabulary.length);
        }

        initialize(vocabSize) {
            const e = this.embeddingSize, h = this.hiddenSize, c = this.contextSize;
            this.embeddings = makeVector(vocabSize * e, 0.12, 101);
            this.Wz = makeVector(h * e, 0.08, 102); this.Uz = makeVector(h * h, 0.06, 103); this.bz = new Float32Array(h);
            this.Wr = makeVector(h * e, 0.08, 104); this.Ur = makeVector(h * h, 0.06, 105); this.br = new Float32Array(h);
            this.Wh = makeVector(h * e, 0.08, 106); this.Uh = makeVector(h * h, 0.06, 107); this.bh = new Float32Array(h);
            this.Wc = makeVector(h * c, 0.04, 108);
            this.Wo = makeVector(vocabSize * h, 0.025, 109);
            this.bo = new Float32Array(vocabSize);
        }

        ensureVocabulary() {
            const needed = this.tokenizer.vocabulary.length;
            const current = this.bo.length;
            if (needed <= current) return;
            const nextEmbeddings = new Float32Array(needed * this.embeddingSize);
            nextEmbeddings.set(this.embeddings);
            nextEmbeddings.set(makeVector((needed - current) * this.embeddingSize, 0.12, 201), current * this.embeddingSize);
            this.embeddings = nextEmbeddings;
            const nextOutput = new Float32Array(needed * this.hiddenSize);
            nextOutput.set(this.Wo);
            nextOutput.set(makeVector((needed - current) * this.hiddenSize, 0.025, 202), current * this.hiddenSize);
            this.Wo = nextOutput;
            const nextBias = new Float32Array(needed);
            nextBias.set(this.bo);
            this.bo = nextBias;
        }

        step(tokenId, priorState, context) {
            this.ensureVocabulary();
            const e = this.embeddingSize, h = this.hiddenSize, c = this.contextSize;
            const state = priorState || new Float32Array(h);
            const xOffset = Math.max(0, Math.min(this.bo.length - 1, tokenId)) * e;
            const x = this.embeddings.subarray(xOffset, xOffset + e);
            const ctx = Array.from({ length: c }, (_, i) => Number(context && context[i]) || 0);
            const z = new Float32Array(h), r = new Float32Array(h), candidate = new Float32Array(h);
            for (let j = 0; j < h; j++) {
                let az = this.bz[j], ar = this.br[j], ah = this.bh[j];
                for (let i = 0; i < e; i++) {
                    az += this.Wz[j * e + i] * x[i];
                    ar += this.Wr[j * e + i] * x[i];
                    ah += this.Wh[j * e + i] * x[i];
                }
                for (let i = 0; i < h; i++) {
                    az += this.Uz[j * h + i] * state[i];
                    ar += this.Ur[j * h + i] * state[i];
                    ah += this.Uh[j * h + i] * state[i] * sigmoid(ar);
                }
                for (let i = 0; i < c; i++) ah += this.Wc[j * c + i] * ctx[i];
                z[j] = sigmoid(az);
                r[j] = sigmoid(ar);
                candidate[j] = Math.tanh(ah);
            }
            const next = new Float32Array(h);
            for (let j = 0; j < h; j++) next[j] = (1 - z[j]) * state[j] + z[j] * candidate[j];
            const logits = new Float32Array(this.bo.length);
            for (let token = 0; token < logits.length; token++) {
                let value = this.bo[token];
                const offset = token * h;
                for (let j = 0; j < h; j++) value += this.Wo[offset + j] * next[j];
                logits[token] = value;
            }
            return { state: next, logits };
        }

        trainSequence(text, context) {
            const ids = this.tokenizer.encode(text);
            this.ensureVocabulary();
            if (ids.length < 2) return 0;
            let state = new Float32Array(this.hiddenSize);
            let updates = 0;
            for (let i = 0; i < ids.length - 1; i++) {
                const result = this.step(ids[i], state, context);
                state = result.state;
                const target = ids[i + 1];
                let max = -Infinity;
                for (let j = 0; j < result.logits.length; j++) if (result.logits[j] > max) max = result.logits[j];
                let total = 0;
                const probabilities = new Float32Array(result.logits.length);
                for (let j = 0; j < probabilities.length; j++) {
                    probabilities[j] = Math.exp(Math.max(-30, result.logits[j] - max));
                    total += probabilities[j];
                }
                const rate = 0.015;
                for (let j = 0; j < probabilities.length; j++) {
                    const gradient = (j === target ? 1 : 0) - probabilities[j] / total;
                    this.bo[j] += rate * gradient;
                    const offset = j * this.hiddenSize;
                    for (let k = 0; k < this.hiddenSize; k++) {
                        this.Wo[offset + k] += rate * gradient * state[k];
                    }
                }
                updates++;
            }
            this.trainedTokens += updates;
            return updates;
        }

        generate(prompt, context, maximumTokens = 24) {
            const ids = this.tokenizer.encode(prompt);
            if (this.trainedTokens < 8 || !ids.length) return [];
            let state = new Float32Array(this.hiddenSize);
            let result = null;
            for (const id of ids) {
                result = this.step(id, state, context);
                state = result.state;
            }
            const output = [];
            let current = ids[ids.length - 1];
            for (let i = 0; i < maximumTokens; i++) {
                result = this.step(current, state, context);
                state = result.state;
                let best = 1, bestScore = -Infinity;
                for (let id = SPECIAL_TOKENS.length; id < result.logits.length; id++) {
                    const score = result.logits[id] - (output.includes(id) ? 0.12 : 0);
                    if (score > bestScore) { bestScore = score; best = id; }
                }
                if (best === 3) break;
                output.push(best);
                current = best;
            }
            return output;
        }

        export() {
            const array = value => Array.from(value);
            return {
                version: 1, embeddingSize: this.embeddingSize, hiddenSize: this.hiddenSize,
                contextSize: this.contextSize, trainedTokens: this.trainedTokens,
                tokenizer: this.tokenizer.export(),
                embeddings: array(this.embeddings),
                Wz: array(this.Wz), Uz: array(this.Uz), bz: array(this.bz),
                Wr: array(this.Wr), Ur: array(this.Ur), br: array(this.br),
                Wh: array(this.Wh), Uh: array(this.Uh), bh: array(this.bh),
                Wc: array(this.Wc), Wo: array(this.Wo), bo: array(this.bo)
            };
        }

        import(saved) {
            if (!saved || saved.version !== 1 || saved.embeddingSize !== 24 ||
                saved.hiddenSize !== 32 || saved.contextSize !== 8 ||
                !saved.tokenizer || !Array.isArray(saved.tokenizer.vocabulary)) return false;
            const vocabSize = saved.tokenizer.vocabulary.length, e = 24, h = 32, c = 8;
            const specs = {
                embeddings: vocabSize * e, Wz: h * e, Uz: h * h, bz: h,
                Wr: h * e, Ur: h * h, br: h, Wh: h * e, Uh: h * h, bh: h,
                Wc: h * c, Wo: vocabSize * h, bo: vocabSize
            };
            for (const [key, length] of Object.entries(specs)) {
                if (!Array.isArray(saved[key]) || saved[key].length !== length ||
                    !saved[key].every(Number.isFinite)) return false;
            }
            this.tokenizer = new NexusTokenizer(saved.tokenizer);
            for (const [key] of Object.entries(specs)) this[key] = Float32Array.from(saved[key]);
            this.trainedTokens = Math.max(0, Number(saved.trainedTokens) || 0);
            return true;
        }
    }

    root.AgentFiveNexusLanguage = Object.freeze({
        NexusTokenizer, NexusDecoderNetwork, SPECIAL_TOKENS
    });
})(typeof self !== "undefined" ? self : globalThis);
