/* Agent-Five-Nexus-1.0.0 PPO worker.
 * This is a classic worker so it can load the companion pure-JS learning module.
 */
"use strict";

try {
    importScripts("agent-learning.js");
} catch (error) {
    self.postMessage({
        requestId: null,
        error: "Unable to load agent-learning.js: " + String(error && error.message || error)
    });
}

self.addEventListener("message", event => {
    const message = event.data || {};
    const requestId = message.requestId;
    const jobs = Array.isArray(message.jobs) ? message.jobs : [];
    const learning = self.AgentLearning;
    if (!learning || typeof learning.trainPPO !== "function") {
        self.postMessage({
            requestId,
            results: jobs.map(job => ({
                token: job && job.token,
                error: "AgentLearning.trainPPO is unavailable in the worker."
            }))
        });
        return;
    }

    const results = jobs.map(job => {
        try {
            const trained = learning.trainPPO(
                job.policy,
                job.critic,
                job.rollout,
                job.options
            );
            return {
                token: job.token,
                policy: trained.policy,
                critic: trained.critic,
                stats: trained.stats
            };
        } catch (error) {
            return {
                token: job && job.token,
                error: String(error && error.message || error)
            };
        }
    });

    self.postMessage({ requestId, results });
});
