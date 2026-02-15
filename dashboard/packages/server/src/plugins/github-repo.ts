import fp from "fastify-plugin";
import type { FastifyInstance } from "fastify";
import { GitHubRepoService } from "../services/github-repo.js";

declare module "fastify" {
  interface FastifyInstance {
    githubRepo: GitHubRepoService;
  }
}

export default fp(async function githubRepoPlugin(
  fastify: FastifyInstance,
) {
  const service = new GitHubRepoService();
  fastify.decorate("githubRepo", service);
});
