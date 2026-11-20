(function (root) {
   "use strict";
   const DEFAULTS = {
       gamma: 0.99,
       gaeLambda: 0.95,
       clipRange: 0.20,
       policyRate: 0.00045,
       valueRate: 0.0012,
       policyStandardDeviation: 0.20,
       epochs: 2,
       minibatchSize: 32
   };
   function clamp(value, minimum, maximum) {
       return Math.max(minimum, Math.min(maximum, value));
   }
   function forwardModel(model, input) {
       const activations = [input];
       for (let layer = 0; layer < model.weights.length; layer++) {
           const prior = activations[layer];
           const next = model.biases[layer].map((bias, targetIndex) => {
               let sum = bias;
               for (let sourceIndex = 0; sourceIndex < prior.length; sourceIndex++) {
                   sum += prior[sourceIndex] * model.weights[layer][sourceIndex][targetIndex];
               }
               return Math.tanh(sum);
           });
           activations.push(next);
       }
       return { output: activations[activations.length - 1], activations };
   }
   function applyOutputGradient(model, input, outputGradient, rate) {
       const result = forwardModel(model, input);
       const deltas = Array(model.weights.length);
       const last = model.weights.length - 1;
       deltas[last] = result.output.map((value, index) =>
           clamp(outputGradient[index], -5, 5) * (1 - value * value)
       );
       for (let layer = last - 1; layer >= 0; layer--) {
           const current = result.activations[layer + 1];
           deltas[layer] = current.map((value, sourceIndex) => {
               let propagated = 0;
               for (let targetIndex = 0; targetIndex < deltas[layer + 1].length; targetIndex++) {
                   propagated += model.weights[layer + 1][sourceIndex][targetIndex] *
                       deltas[layer + 1][targetIndex];
               }
               return propagated * (1 - value * value);
           });
       }
       for (let layer = 0; layer < model.weights.length; layer++) {
           const prior = result.activations[layer];
           for (let sourceIndex = 0; sourceIndex < prior.length; sourceIndex++) {
               for (let targetIndex = 0; targetIndex < deltas[layer].length; targetIndex++) {
                   const change = rate * deltas[layer][targetIndex] * prior[sourceIndex];
                   model.weights[layer][sourceIndex][targetIndex] =
                       clamp(model.weights[layer][sourceIndex][targetIndex] + change, -4, 4);
               }
           }
           for (let targetIndex = 0; targetIndex < deltas[layer].length; targetIndex++) {
               model.biases[layer][targetIndex] =
                   clamp(model.biases[layer][targetIndex] + rate * deltas[layer][targetIndex], -4, 4);
           }
       }
   }
   function trainValue(model, input, target, rate) {
       const result = forwardModel(model, input);
       const outputGradient = result.output.map((value, index) =>
           clamp(target[index] - value, -1, 1)
       );
       applyOutputGradient(model, input, outputGradient, rate);
   }
   function randomNormal() {
       let first = 0;
       let second = 0;
       while (first === 0) first = Math.random();
       while (second === 0) second = Math.random();
       return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
   }
   function normalLogProbability(latent, mean, standardDeviation) {
       const variance = standardDeviation * standardDeviation;
       let logProbability = 0;
       for (let index = 0; index < latent.length; index++) {
           const difference = latent[index] - mean[index];
           logProbability += -0.5 * (difference * difference / variance) -
               Math.log(standardDeviation * Math.sqrt(2 * Math.PI));
       }
       return logProbability;
   }
   function sampleAction(policy, critic, inputs, meanOverride, standardDeviation, precomputedPolicyResult) {
// Reuse perception's forward pass during normal action selection.
       const policyResult = precomputedPolicyResult || policy.forward(inputs);
       const baseMean = policyResult.output;
       const mean = Array.isArray(meanOverride) && meanOverride.length === baseMean.length ?
           meanOverride.map((value, index) => clamp(value, -0.98, 0.98)) :
           baseMean.slice();
       const deviation = clamp(Number(standardDeviation) || DEFAULTS.policyStandardDeviation, 0.05, 0.5);
       const latent = mean.map(value => value + randomNormal() * deviation);
       const squashed = latent.map(Math.tanh);
       const value = critic ? critic.forward(inputs).output[0] : 0;
       return {
           action: [squashed[0], squashed[1], (squashed[2] + 1) * 0.5],
           latent,
           mean,
           meanOffset: mean.map((value, index) => value - baseMean[index]),
           logProbability: normalLogProbability(latent, mean, deviation),
           value,
           activations: policyResult.activations
       };
   }
   function trainPPO(policy, critic, rollout, suppliedOptions) {
       const options = { ...DEFAULTS, ...(suppliedOptions || {}) };
       const samples = Array.isArray(rollout) ? rollout.filter(item =>
           item && Array.isArray(item.inputs) && Array.isArray(item.latent) &&
           Array.isArray(item.meanOffset) && item.inputs.every(Number.isFinite) &&
           item.latent.every(Number.isFinite) && item.meanOffset.every(Number.isFinite) &&
           Number.isFinite(item.logProbability) && Number.isFinite(item.value) &&
           Number.isFinite(item.reward)
       ) : [];
       if (!samples.length) {
           return { policy, critic, stats: { updates: 0, meanReward: 0, meanAdvantage: 0 } };
       }
       const advantages = Array(samples.length).fill(0);
       const returns = Array(samples.length).fill(0);
       let nextValue = 0;
       let runningAdvantage = 0;
       for (let index = samples.length - 1; index >= 0; index--) {
           const item = samples[index];
           const terminal = Boolean(item.done);
           const followingValue = terminal ? 0 :
               (index === samples.length - 1 ? clamp(item.nextValue || 0, -1, 1) : nextValue);
           const delta = item.reward + options.gamma * followingValue - item.value;
           runningAdvantage = delta + options.gamma * options.gaeLambda *
               (terminal ? 0 : 1) * runningAdvantage;
           advantages[index] = runningAdvantage;
           returns[index] = clamp(runningAdvantage + item.value, -1, 1);
           nextValue = item.value;
       }
       const averageAdvantage = advantages.reduce((sum, value) => sum + value, 0) / advantages.length;
       const variance = advantages.reduce((sum, value) =>
           sum + (value - averageAdvantage) * (value - averageAdvantage), 0
       ) / advantages.length;
       const standardDeviation = Math.sqrt(variance) + 1e-6;
       const normalizedAdvantages = advantages.map(value =>
           clamp((value - averageAdvantage) / standardDeviation, -4, 4)
       );
       const deviation = clamp(options.policyStandardDeviation, 0.05, 0.5);
       const indices = Array.from({ length: samples.length }, (_, index) => index);
       let updateCount = 0;
       for (let epoch = 0; epoch < options.epochs; epoch++) {
           for (let index = indices.length - 1; index > 0; index--) {
               const swapIndex = Math.floor(Math.random() * (index + 1));
               [indices[index], indices[swapIndex]] = [indices[swapIndex], indices[index]];
           }
           const batchSize = Math.max(1, Math.floor(options.minibatchSize));
           for (let offset = 0; offset < indices.length; offset += batchSize) {
               const end = Math.min(indices.length, offset + batchSize);
               for (let batchIndex = offset; batchIndex < end; batchIndex++) {
                   const sampleIndex = indices[batchIndex];
                   const item = samples[sampleIndex];
                   const baseOutput = forwardModel(policy, item.inputs).output;
                   const mean = baseOutput.map((value, index) =>
                       clamp(value + item.meanOffset[index], -0.98, 0.98)
                   );
                   const newLogProbability = normalLogProbability(item.latent, mean, deviation);
                   const logRatio = clamp(newLogProbability - item.logProbability, -5, 5);
                   const ratio = Math.exp(logRatio);
                   const advantage = normalizedAdvantages[sampleIndex];
                   const clippedRatio = clamp(ratio, 1 - options.clipRange, 1 + options.clipRange);
                   const unclippedObjective = ratio * advantage;
                   const clippedObjective = clippedRatio * advantage;
                   const objectiveGradient = unclippedObjective <= clippedObjective ? unclippedObjective : 0;
                   const outputGradient = mean.map((value, index) =>
                       clamp(
                           objectiveGradient * (item.latent[index] - value) /
                               (deviation * deviation),
                           -5,
                           5
                       )
                   );
                   applyOutputGradient(policy, item.inputs, outputGradient, options.policyRate);
                   trainValue(critic, item.inputs, [returns[sampleIndex]], options.valueRate);
                   updateCount++;
               }
           }
       }
       const meanReward = samples.reduce((sum, item) => sum + item.reward, 0) / samples.length;
       return {
           policy,
           critic,
           stats: {
               updates: updateCount,
               samples: samples.length,
               meanReward,
               meanAdvantage: averageAdvantage
           }
       };
   }
   function planAction(policyMean, state, settings) {
       if (!state || !state.goal || !state.obstacle || !policyMean || policyMean.length < 3) {
           return policyMean ? policyMean.slice() : [0, 0, 0];
       }
       const config = {
           width: 500,
           height: 500,
           speed: 2.4,
           acceleration: 0.16,
           brakingStrength: 0.18,
           agentRadius: 10,
           obstacleRadius: 30,
           avoidanceRange: 115,
           horizon: 12,
           blend: 0.24,
           ...(settings || {})
       };
       const offsets = [0, 0.48, -0.48, 0.92, -0.92, 1.35, -1.35];
       const startDistance = Math.hypot(state.goal.x - state.x, state.goal.y - state.y);
       let bestScore = -Infinity;
       let bestSuggestion = policyMean.slice();
       for (const offset of offsets) {
           let x = state.x;
           let y = state.y;
           let velocityX = state.velocity.x;
           let velocityY = state.velocity.y;
           let score = 0;
           let previousGoalDistance = startDistance;
           let minimumObstacleGap = Infinity;
           let crashed = false;
           let firstAction = null;
           for (let step = 0; step < config.horizon; step++) {
               const goalX = state.goal.x - x;
               const goalY = state.goal.y - y;
               const goalAngle = Math.atan2(goalY, goalX) + offset;
               let thrustX = Math.cos(goalAngle) * 0.88;
               let thrustY = Math.sin(goalAngle) * 0.88;
               const obstacleX = state.obstacle.x - x;
               const obstacleY = state.obstacle.y - y;
               const obstacleDistance = Math.hypot(obstacleX, obstacleY);
               const obstacleGap = obstacleDistance - config.obstacleRadius - config.agentRadius;
               minimumObstacleGap = Math.min(minimumObstacleGap, obstacleGap);
               const desiredX = Math.cos(goalAngle);
               const desiredY = Math.sin(goalAngle);
               const obstacleDirectionX = obstacleDistance > 0.001 ? obstacleX / obstacleDistance : 0;
               const obstacleDirectionY = obstacleDistance > 0.001 ? obstacleY / obstacleDistance : 0;
               const headingTowardObstacle = desiredX * obstacleDirectionX + desiredY * obstacleDirectionY;
               if (obstacleGap < config.avoidanceRange && headingTowardObstacle > 0.15) {
                   const urgency = clamp((config.avoidanceRange - obstacleGap) / config.avoidanceRange, 0, 1);
                   const awayX = -obstacleDirectionX;
                   const awayY = -obstacleDirectionY;
                   thrustX = thrustX * (1 - urgency * 0.85) + awayX * urgency * 0.85;
                   thrustY = thrustY * (1 - urgency * 0.85) + awayY * urgency * 0.85;
               }
               const thrustMagnitude = Math.hypot(thrustX, thrustY) || 1;
               thrustX = clamp(thrustX / thrustMagnitude, -1, 1);
               thrustY = clamp(thrustY / thrustMagnitude, -1, 1);
               const speed = Math.hypot(velocityX, velocityY);
               const brake = speed > config.speed * 0.82 ? 0.35 : 0;
               if (!firstAction) firstAction = [thrustX, thrustY, brake * 2 - 1];
               const damping = 1 - brake * config.brakingStrength;
               velocityX = clamp((velocityX + thrustX * config.acceleration) * damping, -config.speed, config.speed);
               velocityY = clamp((velocityY + thrustY * config.acceleration) * damping, -config.speed, config.speed);
               x += velocityX;
               y += velocityY;
               if (x < config.agentRadius || x > config.width - config.agentRadius ||
                   y < config.agentRadius || y > config.height - config.agentRadius) {
                   score -= 90;
                   crashed = true;
                   break;
               }
               const updatedGap = Math.hypot(state.obstacle.x - x, state.obstacle.y - y) -
                   config.obstacleRadius - config.agentRadius;
               minimumObstacleGap = Math.min(minimumObstacleGap, updatedGap);
               if (updatedGap < 0) {
                   score -= 120;
                   crashed = true;
                   break;
               }
               const currentGoalDistance = Math.hypot(state.goal.x - x, state.goal.y - y);
               score += previousGoalDistance - currentGoalDistance;
               previousGoalDistance = currentGoalDistance;
               if (updatedGap < config.avoidanceRange) {
                   score -= (1 - clamp(updatedGap / config.avoidanceRange, 0, 1)) * 0.9;
               }
               score -= (Math.abs(thrustX) + Math.abs(thrustY)) * 0.012 + brake * 0.006;
               if (currentGoalDistance < config.agentRadius + 12) {
                   score += 18;
                   break;
               }
           }
           if (!crashed && minimumObstacleGap > config.avoidanceRange) score += 0.1;
           if (score > bestScore && firstAction) {
               bestScore = score;
               bestSuggestion = firstAction;
           }
       }
       const blend = clamp(config.blend, 0, 0.6);
       return policyMean.map((value, index) =>
           clamp(value * (1 - blend) + bestSuggestion[index] * blend, -0.98, 0.98)
       );
   }
   Object.defineProperty(root, "AgentLearning", {
       value: {
           sampleAction,
           trainPPO,
           planAction,
           forwardModel
       },
       configurable: true,
       enumerable: true,
       writable: true
   });
})(typeof self !== "undefined" ? self : globalThis);