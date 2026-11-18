import type { VoiceOption } from '@shared/types';

export type VoiceProviderId = 'speko' | 'openai' | 'xai';

export interface VoiceSession {
  callId: string;
  provider: VoiceProviderId;
  providerName: string;
  /** Set when the bot's own voice provider was unavailable and this one stepped in. */
  fallbackFrom?: VoiceProviderId;
  // WebSocket realtime providers (OpenAI, xAI)
  clientSecret?: string;
  expiresAt?: number;
  model?: string;
  websocketUrl?: string;
  protocols?: string[];
  // Speko (WebRTC via @spekoai/client)
  transportToken?: string;
  transportUrl?: string;
  transport?: 'livekit' | 'daily';
  spekoSessionId?: string;
  sampleRate: number;
  instructions: string;
  voice: string;
  speed: number;
  language: string;
  tools: unknown[];
  session: Record<string, unknown>;
}

export interface VoiceOptions { voices: VoiceOption[]; languages: { code: string; name: string }[]; speeds: number[]; providers?: { id: VoiceProviderId; name: string }[]; defaultVoice?: string }

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`/api/voice${path}`, { method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = ((await res.json()) as { error?: string }).error ?? msg; } catch { /* not json */ }
    throw new Error(msg || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

let optionsPromise: Promise<VoiceOptions> | null = null;

export const voiceApi = {
  options: () => (optionsPromise ??= fetch('/api/voice/options', { credentials: 'include' }).then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json() as Promise<VoiceOptions>;
  }).catch((e) => { optionsPromise = null; throw e; })),
  session: (botId: string, opts: { exclude?: string[]; replaceCallId?: string } = {}) => post<VoiceSession>('/session', { botId, ...opts }),
  tool: (botId: string, callId: string, name: string, args: unknown) => post<{ output: string; taskSeq?: number }>('/tool', { botId, callId, name, args }),
  end: (botId: string, callId: string, transcript: { role: 'user' | 'assistant'; text: string }[], error?: string) => post<{ ok: true }>('/end', { botId, callId, transcript, error }),
  /** Best-effort end on tab close. */
  endBeacon: (botId: string, callId: string, transcript: { role: 'user' | 'assistant'; text: string }[]) => {
    try { navigator.sendBeacon('/api/voice/end', new Blob([JSON.stringify({ botId, callId, transcript })], { type: 'application/json' })); } catch { /* ignore */ }
  },
  async preview(p: { botId: string; voice?: string; speed?: number; language?: string; text?: string }): Promise<Int16Array> {
    const res = await fetch('/api/voice/preview', { method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...p, codec: 'pcm' }) });
    if (!res.ok) throw new Error('Preview unavailable');
    const type = res.headers.get('Content-Type') ?? '';
    const buf = await res.arrayBuffer();
    if (type.startsWith('audio/pcm')) return new Int16Array(buf, 0, buf.byteLength >> 1);
    // Speko returns an encoded file (mp3/wav/ogg): decode it and convert to 24 kHz PCM16 for the shared player.
    const ctx = new OfflineAudioContext(1, 1, 24000);
    const audio = await ctx.decodeAudioData(buf.slice(0));
    const off = new OfflineAudioContext(1, Math.ceil(audio.duration * 24000), 24000);
    const src = off.createBufferSource(); src.buffer = audio; src.connect(off.destination); src.start();
    const f32 = (await off.startRendering()).getChannelData(0);
    const out = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) out[i] = Math.max(-32768, Math.min(32767, Math.round(f32[i] * 32767)));
    return out;
  },
  async transcribe(blob: Blob, language?: string): Promise<string> {
    const fd = new FormData();
    if (language) fd.append('language', language);
    const ext = blob.type.includes('ogg') ? 'ogg' : blob.type.includes('mp4') ? 'm4a' : 'webm';
    fd.append('file', blob, `dictation.${ext}`);
    const res = await fetch('/api/voice/stt', { method: 'POST', credentials: 'include', body: fd });
    if (!res.ok) {
      let msg = 'Transcription failed';
      try { msg = ((await res.json()) as { error?: string }).error ?? msg; } catch { /* not json */ }
      throw new Error(msg);
    }
    return ((await res.json()) as { text: string }).text;
  },
};
