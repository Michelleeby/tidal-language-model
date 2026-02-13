import fp from "fastify-plugin";
import type { FastifyInstance } from "fastify";
import { ProvisioningChain } from "../services/provisioning-chain.js";
import { LocalProvider } from "../services/providers/local-provider.js";
import { AWSProvider } from "../services/providers/aws-provider.js";
import { VastAIProvider } from "../services/providers/vastai-provider.js";
import { WorkerSpawner } from "../services/worker-spawner.js";

declare module "fastify" {
  interface FastifyInstance {
    provisioningChain: ProvisioningChain;
    workerSpawner: WorkerSpawner;
  }
}

export default fp(async function provisioningPlugin(fastify: FastifyInstance) {
  const config = fastify.serverConfig;

  const spawner = new WorkerSpawner(
    config.projectRoot,
    config.pythonBin,
    config.redisUrl,
    fastify.log,
  );

  const chain = new ProvisioningChain([
    new LocalProvider(spawner),
    new AWSProvider(),
    new VastAIProvider({
      apiKey: config.vastaiApiKey,
      dashboardUrl: config.dashboardUrl,
      authToken: config.authToken,
      repoUrl: config.repoUrl,
      log: fastify.log,
    }),
  ]);

  fastify.decorate("provisioningChain", chain);
  fastify.decorate("workerSpawner", spawner);

  fastify.addHook("onClose", () => spawner.cleanup());
});
