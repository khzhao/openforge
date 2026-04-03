const PLUGIN_ID = "rl-training-headers";
const STATE_KEY = "__openforgeRlTrainingHeadersState";
const SIDE_TRIGGERS = new Set(["heartbeat", "memory", "cron"]);

type HeaderState = {
  current: Record<string, string> | null;
  installed: boolean;
  originalFetch: typeof fetch | null;
};

function getState(): HeaderState {
  const globalScope = globalThis as typeof globalThis & {
    [STATE_KEY]?: HeaderState;
  };
  if (!globalScope[STATE_KEY]) {
    globalScope[STATE_KEY] = {
      current: null,
      installed: false,
      originalFetch: null,
    };
  }
  return globalScope[STATE_KEY]!;
}

function pluginConfig(api: any): { urlIncludes: string[] } {
  const config =
    api?.config?.plugins?.entries?.[PLUGIN_ID]?.config ??
    api?.pluginConfig ??
    {};
  return {
    urlIncludes: Array.isArray(config?.urlIncludes)
      ? config.urlIncludes.filter((value: unknown) => typeof value === "string")
      : [],
  };
}

function sessionIdFrom(event: any, ctx: any): string | null {
  const candidates = [
    event?.sessionId,
    event?.session?.id,
    event?.session?.sessionId,
    event?.run?.sessionId,
    event?.user,
    event?.message?.user,
    event?.payload?.user,
    ctx?.sessionId,
    ctx?.session?.id,
    ctx?.session?.sessionId,
    ctx?.run?.sessionId,
    ctx?.user,
  ];
  for (const value of candidates) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return null;
}

function triggerFrom(event: any, ctx: any): string {
  const candidates = [
    event?.trigger,
    event?.message?.trigger,
    event?.run?.trigger,
    ctx?.trigger,
  ];
  for (const value of candidates) {
    if (typeof value === "string" && value.trim()) {
      return value.trim().toLowerCase();
    }
  }
  return "user";
}

function turnTypeFrom(event: any, ctx: any): "main" | "side" {
  return SIDE_TRIGGERS.has(triggerFrom(event, ctx)) ? "side" : "main";
}

function shouldInject(url: string, api: any): boolean {
  const { urlIncludes } = pluginConfig(api);
  if (urlIncludes.length > 0) {
    return urlIncludes.some((piece) => url.includes(piece));
  }
  return url.includes("/v1/chat/completions");
}

function patchFetch(api: any): void {
  const state = getState();
  if (state.installed) {
    return;
  }
  state.installed = true;
  state.originalFetch = globalThis.fetch.bind(globalThis);

  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const headersToInject = state.current;
    if (!headersToInject) {
      return state.originalFetch!(input, init);
    }

    const request = input instanceof Request ? input : null;
    const method = String(init?.method ?? request?.method ?? "GET").toUpperCase();
    const url =
      typeof input === "string" || input instanceof URL
        ? String(input)
        : request?.url ?? "";

    if (method !== "POST" || !shouldInject(url, api)) {
      return state.originalFetch!(input, init);
    }

    const headers = new Headers(request?.headers ?? init?.headers ?? undefined);
    for (const [key, value] of Object.entries(headersToInject)) {
      if (!headers.has(key)) {
        headers.set(key, value);
      }
    }

    return state.originalFetch!(input, {
      ...init,
      headers,
    });
  };
}

export default function register(api: any) {
  patchFetch(api);

  api.on(
    "before_prompt_build",
    (event: any, ctx: any) => {
      const sessionId = sessionIdFrom(event, ctx);
      const headers: Record<string, string> = {
        "X-Turn-Type": turnTypeFrom(event, ctx),
      };
      if (sessionId) {
        headers["X-Session-Id"] = sessionId;
      }
      if (Object.keys(headers).length === 0) {
        return null;
      }
      getState().current = headers;
      return null;
    },
    { priority: 100 },
  );

  api.on(
    "agent_end",
    () => {
      getState().current = null;
      return null;
    },
    { priority: -100 },
  );
}
