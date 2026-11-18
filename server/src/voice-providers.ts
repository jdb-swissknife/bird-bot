// Voice providers for calls, previews, dictation and call summaries.
//
//   speko  — primary. A hosted Speko voice agent ("Bird-bot Voice") joined over WebRTC from the browser.
//            Its relay tools are Speko *webhook* tools that call back into this server (/hooks/speko/tool),
//            so real work still lands in the bot's own text agent, exactly like the xAI design.
//   openai — backup. OpenAI realtime over WebSocket, same event protocol the xAI path already speaks.
//   xai    — backup (needs XAI_API_KEY). The original Grok Bot path.
import crypto from 'node:crypto';
import { Speko } from '@spekoai/sdk';
import { env } from './env.js';
import type { VoiceOption } from './types.js';

export type VoiceProvider = 'speko' | 'openai' | 'xai';
export const PROVIDERS: VoiceProvider[] = ['speko', 'openai', 'xai'];

export function providerAvailable(p: VoiceProvider): boolean {
  return p === 'speko' ? !!env.spekoKey : p === 'openai' ? !!env.openaiKey : !!env.xaiKey;
}
export const availableProviders = () => PROVIDERS.filter(providerAvailable);

export const PROVIDER_LABEL: Record<VoiceProvider, string> = { speko: 'Speko', openai: 'OpenAI', xai: 'xAI' };
const FALLBACK_VOICE: Record<VoiceProvider, string> = { speko: 'a5136bf9-224c-4d76-b823-52bd5efcffcc', openai: 'marin', xai: 'luna' };

/** "speko:abc" -> { provider: 'speko', id: 'abc' }. Legacy bare ids (old xAI voices) map to xAI when it is set up. */
export function parseVoice(raw: string | null | undefined): { provider: VoiceProvider; id: string } {
  const s = String(raw ?? '').trim();
  const m = /^(speko|openai|xai):(.+)$/.exec(s);
  if (m && providerAvailable(m[1] as VoiceProvider)) return { provider: m[1] as VoiceProvider, id: m[2] };
  if (s && !m && providerAvailable('xai')) return { provider: 'xai', id: s };
  const d = /^(speko|openai|xai):(.+)$/.exec(env.defaultVoice);
  if (d && providerAvailable(d[1] as VoiceProvider)) return { provider: d[1] as VoiceProvider, id: d[2] };
  const first = availableProviders()[0] ?? 'speko';
  return { provider: first, id: FALLBACK_VOICE[first] };
}

/** The voice to use when falling back to provider p. */
export const fallbackVoice = (p: VoiceProvider, preferred?: { provider: VoiceProvider; id: string }) =>
  preferred?.provider === p ? preferred.id : p === 'speko' && env.defaultVoice.startsWith('speko:') ? env.defaultVoice.slice(6) : FALLBACK_VOICE[p];

/** Providers to try for a call, best first, skipping excluded ones. */
export function providerOrder(preferred: VoiceProvider, exclude: string[] = []): VoiceProvider[] {
  const order = [preferred, ...(env.voiceOrder as VoiceProvider[]), ...PROVIDERS];
  return [...new Set(order)].filter((p) => PROVIDERS.includes(p) && providerAvailable(p) && !exclude.includes(p));
}

/** Speko wants a full locale. */
export const spekoLanguage = (lang: string) => (!lang || lang === 'auto' || lang === 'en' ? 'en-US' : lang);

// ---------- Speko ----------

let spekoClient: Speko | null = null;
export const speko = () => (spekoClient ??= new Speko({ apiKey: env.spekoKey, baseUrl: env.spekoBaseUrl }));

/** Webhook signing secret for the Speko tools, derived from the session secret so it survives restarts. */
export const spekoWebhookSecret = () =>
  `whsec_${crypto.createHmac('sha256', env.sessionSecret).update('speko-webhook-v1').digest('base64')}`;

export const SPEKO_TOOL_PATH = '/hooks/speko/tool';

/** The relay tools, as Speko webhook tools. Kept in step with VOICE_TOOLS in voice.ts (minus end_the_call). */
export function spekoToolDefs(tools: { name: string; description: string; parameters: Record<string, unknown> }[]) {
  const url = `${env.serverUrl.replace(/\/+$/, '')}${SPEKO_TOOL_PATH}`;
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    parameters: t.parameters,
    preToolSpeech: (t.name === 'send_task' ? 'auto' : 'never') as 'auto' | 'never',
    source: {
      kind: 'webhook' as const,
      url,
      secret: spekoWebhookSecret(),
      // Per-call secret (session toolSecrets): never enters the prompt or transcript, only tool requests.
      headers: { 'X-Birdbot-Call': '{{call_token}}' },
      timeoutMs: 4_000, // Speko caps webhook tools at 4s; these all answer at once (send_task hands off and returns)
    },
  }));
}

let agentPromise: Promise<string> | null = null;

/** Find (or create) the hosted "Bird-bot Voice" agent and make sure its relay tools point at this server. */
export function ensureSpekoAgent(tools: { name: string; description: string; parameters: Record<string, unknown> }[]): Promise<string> {
  agentPromise ??= (async () => {
    const s = speko();
    let agent = (await s.agents.list()).find((a) => a.name === env.spekoAgentName);
    if (!agent) {
      agent = (await s.agents.create({
        name: env.spekoAgentName,
        systemPrompt: '{{bot_prompt}}',
        promptVariables: [{ name: 'bot_prompt', defaultValue: 'You are Bird-bot, a friendly voice assistant. Keep replies short.', description: 'Per-call instructions built by the Bird-bot app' }],
        voice: FALLBACK_VOICE.speko,
        intent: { language: 'en-US', optimizeFor: 'latency' },
      })) as any;
      console.log('[voice] created Speko agent', agent!.id);
    }
    const agentId = agent!.id;
    const existing = await s.agents.tools.list(agentId).catch(() => []);
    for (const def of spekoToolDefs(tools)) {
      const have = existing.find((t) => t.name === def.name);
      try {
        if (have) await s.agents.tools.update(agentId, have.id, def as any);
        else await s.agents.tools.create(agentId, def as any);
      } catch (e: any) {
        console.warn(`[voice] Speko tool ${def.name} not synced:`, e?.message ?? e);
      }
    }
    if (!existing.some((t) => t.name === 'end_call')) {
      await s.agents.tools.create(agentId, {
        name: 'end_call', description: 'Hang up when the user says goodbye or asks to end the call. Say a short goodbye first.',
        parameters: { type: 'object', properties: {} }, source: { kind: 'builtin', name: 'end_call' },
      } as any).catch((e: any) => console.warn('[voice] Speko end_call tool not added:', e?.message ?? e));
    }
    return agentId;
  })().catch((e) => { agentPromise = null; throw e; });
  return agentPromise;
}

export interface SpekoSession { sessionId: string; transportToken: string; transportUrl: string; transport: string }

export async function mintSpekoSession(p: { agentId: string; instructions: string; voice: string; speed: number; language: string; callToken: string; firstMessage: string }): Promise<SpekoSession> {
  const body: Record<string, unknown> = {
    agentId: p.agentId,
    intent: { language: spekoLanguage(p.language), optimizeFor: 'latency' },
    variables: { bot_prompt: p.instructions },
    toolSecrets: { call_token: p.callToken },
    voice: p.voice,
    firstMessage: p.firstMessage,
    ttsOptions: { speed: p.speed },
    timezone: process.env.TZ || 'America/Chicago',
    maxDurationSeconds: 3600,
  };
  const r = await fetch(`${env.spekoBaseUrl}/v1/sessions`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${env.spekoKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(20_000),
  });
  const j: any = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(`Speko session ${r.status}: ${JSON.stringify(j).slice(0, 300)}`);
  const token = j.transportToken ?? j.conversationToken;
  const url = j.transportUrl ?? j.livekitUrl;
  if (!token || !url) throw new Error('Speko session missing transport credentials');
  return { sessionId: j.sessionId ?? j.id, transportToken: token, transportUrl: url, transport: j.transport ?? 'livekit' };
}

export async function endSpekoSession(sessionId: string) {
  await speko().sessions.end(sessionId).catch((e: any) => console.warn('[voice] Speko end failed:', e?.message ?? e));
}

/**
 * Standard Webhooks signature check (webhook-id / webhook-timestamp / webhook-signature).
 * Returns true/false when the headers are present, null when Speko sent none.
 */
export function verifySpekoSignature(headers: Headers, rawBody: string): boolean | null {
  const id = headers.get('webhook-id'); const ts = headers.get('webhook-timestamp'); const sig = headers.get('webhook-signature');
  if (!id || !ts || !sig) return null;
  if (Math.abs(Date.now() / 1000 - Number(ts)) > 600) return false;
  const secret = spekoWebhookSecret();
  const keys = [Buffer.from(secret.slice(6), 'base64'), Buffer.from(secret, 'utf8')];
  const expected = keys.map((k) => crypto.createHmac('sha256', k).update(`${id}.${ts}.${rawBody}`).digest('base64'));
  const given = sig.split(' ').map((s) => s.split(',')[1] ?? s);
  return given.some((g) => expected.some((e) => g.length === e.length && crypto.timingSafeEqual(Buffer.from(g), Buffer.from(e))));
}

const SPEKO_VENDORS = ['cartesia', 'elevenlabs', 'rime', 'inworld', 'deepgram', 'openai', 'xai', 'google', 'minimax'];
const VENDOR_NAME: Record<string, string> = { cartesia: 'Cartesia', elevenlabs: 'ElevenLabs', rime: 'Rime', inworld: 'Inworld', deepgram: 'Deepgram', openai: 'OpenAI', xai: 'xAI', google: 'Google', minimax: 'MiniMax' };

export async function spekoVoices(): Promise<VoiceOption[]> {
  const { voices } = await speko().voices.list();
  return voices
    .filter((v) => SPEKO_VENDORS.includes(v.vendor))
    .sort((a, b) => SPEKO_VENDORS.indexOf(a.vendor) - SPEKO_VENDORS.indexOf(b.vendor) || a.name.localeCompare(b.name))
    .map((v) => ({ id: `speko:${v.id}`, name: `${v.name} · ${VENDOR_NAME[v.vendor] ?? v.vendor}`, provider: 'speko' }));
}

// ---------- OpenAI realtime ----------

export const OPENAI_VOICES = ['marin', 'cedar', 'alloy', 'ash', 'ballad', 'coral', 'echo', 'sage', 'shimmer', 'verse'];

export async function mintOpenAISession(session: Record<string, unknown>): Promise<{ value: string; expiresAt: number }> {
  const r = await fetch('https://api.openai.com/v1/realtime/client_secrets', {
    method: 'POST',
    headers: { Authorization: `Bearer ${env.openaiKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ expires_after: { anchor: 'created_at', seconds: 300 }, session: { ...session, model: env.openaiVoiceModel } }),
    signal: AbortSignal.timeout(15_000),
  });
  const j: any = await r.json().catch(() => ({}));
  if (!r.ok || !j.value) throw new Error(`OpenAI client secret ${r.status}: ${JSON.stringify(j).slice(0, 300)}`);
  return { value: j.value, expiresAt: j.expires_at };
}

/** One short answer from whichever text model is configured (call summaries). */
export async function quickText(system: string, user: string): Promise<string> {
  if (env.spekoKey) {
    try {
      const r: any = await speko().complete({ messages: [{ role: 'system', content: system }, { role: 'user', content: user }], intent: { language: 'en' } } as any);
      if (r?.text) return String(r.text);
    } catch (e: any) { console.warn('[voice] Speko complete failed:', e?.message); }
  }
  if (env.openaiKey) {
    const r = await fetch('https://api.openai.com/v1/responses', {
      method: 'POST', headers: { Authorization: `Bearer ${env.openaiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: process.env.OPENAI_TEXT_MODEL ?? 'gpt-5-mini', store: false, input: [{ role: 'system', content: system }, { role: 'user', content: user }] }),
      signal: AbortSignal.timeout(45_000),
    });
    const j: any = await r.json().catch(() => ({}));
    const text = j.output_text ?? (j.output ?? []).flatMap((o: any) => o.content ?? []).map((c: any) => c.text ?? '').join('');
    if (text) return String(text);
  }
  if (env.xaiKey) {
    const r = await fetch('https://api.x.ai/v1/responses', {
      method: 'POST', headers: { Authorization: `Bearer ${env.xaiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: env.xaiModel, store: false, reasoning: { effort: 'low' }, input: [{ role: 'system', content: system }, { role: 'user', content: user }] }),
      signal: AbortSignal.timeout(45_000),
    });
    const j: any = await r.json().catch(() => ({}));
    const text = j.output_text ?? (j.output ?? []).flatMap((o: any) => o.content ?? []).map((c: any) => c.text ?? '').join('');
    if (text) return String(text);
  }
  return '';
}
