import { execFile as defaultExecFile } from "node:child_process";
import { promisify } from "node:util";

const GITHUB_API = "https://api.github.com";

export interface CreateRepoResult {
  htmlUrl: string;
  cloneUrl: string;
}

export interface GitStatus {
  dirty: boolean;
  files: string[];
}

/**
 * Service for GitHub repo creation and git operations on local clones.
 * Accepts an optional execFile override for testing.
 */
export class GitHubRepoService {
  private execFileAsync: (
    file: string,
    args: string[],
    opts: { cwd?: string },
  ) => Promise<{ stdout: string; stderr: string }>;

  constructor(execFileFn?: typeof defaultExecFile) {
    this.execFileAsync = promisify(execFileFn ?? defaultExecFile) as typeof this.execFileAsync;
  }

  // -------------------------------------------------------------------------
  // GitHub API
  // -------------------------------------------------------------------------

  async createRepo(
    token: string,
    pluginName: string,
    description: string,
  ): Promise<CreateRepoResult> {
    const repoName = `tidal-plugin-${pluginName}`;

    const res = await fetch(`${GITHUB_API}/user/repos`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        name: repoName,
        description,
        private: false,
        auto_init: false,
      }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      const msg = (body as { message?: string }).message ?? res.statusText;
      throw new Error(`Failed to create GitHub repo (${res.status}): ${msg}`);
    }

    const data = (await res.json()) as {
      html_url: string;
      clone_url: string;
    };

    return {
      htmlUrl: data.html_url,
      cloneUrl: data.clone_url,
    };
  }

  // -------------------------------------------------------------------------
  // Git operations on local clones
  // -------------------------------------------------------------------------

  async cloneRepo(repoUrl: string, destDir: string): Promise<void> {
    await this.execFileAsync("git", ["clone", repoUrl, destDir], {});
  }

  async configureGitUser(repoDir: string, login: string): Promise<void> {
    await this.git(repoDir, ["config", "user.name", login]);
    await this.git(repoDir, [
      "config",
      "user.email",
      `${login}@users.noreply.github.com`,
    ]);
  }

  async commitAndPush(
    repoDir: string,
    token: string,
    login: string,
    repoUrl: string,
    message: string,
  ): Promise<void> {
    await this.git(repoDir, ["add", "-A"]);
    await this.git(repoDir, ["commit", "-m", message]);

    // Set auth remote URL for push
    const authUrl = this.buildAuthUrl(repoUrl, login, token);
    await this.git(repoDir, ["remote", "set-url", "origin", authUrl]);
    await this.git(repoDir, ["push", "origin", "main"]);
    // Reset to non-auth URL to avoid storing token on disk
    await this.git(repoDir, ["remote", "set-url", "origin", repoUrl]);
  }

  async pull(repoDir: string): Promise<void> {
    await this.git(repoDir, ["pull", "origin", "main"]);
  }

  async getStatus(repoDir: string): Promise<GitStatus> {
    const { stdout } = await this.git(repoDir, ["status", "--porcelain"]);
    const lines = stdout
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length > 0);
    return {
      dirty: lines.length > 0,
      files: lines,
    };
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  private async git(
    cwd: string,
    args: string[],
  ): Promise<{ stdout: string; stderr: string }> {
    return this.execFileAsync("git", args, { cwd });
  }

  private buildAuthUrl(
    repoUrl: string,
    login: string,
    token: string,
  ): string {
    // https://github.com/user/repo.git → https://login:token@github.com/user/repo.git
    return repoUrl.replace("https://", `https://${login}:${token}@`);
  }
}
