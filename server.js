'use strict';
require('dotenv').config();
const express   = require('express');
const path      = require('path');
const rateLimit = require('express-rate-limit');

const app = express();

// ── Config ───────────────────────────────────────────────────────────────────
const GROQ_KEY       = process.env.GROQ_API_KEY       || '';
const MISTRAL_KEY    = process.env.MISTRAL_API_KEY    || '';
const WOLFRAM_ID     = process.env.WOLFRAM_APP_ID     || '';
const OPENROUTER_KEY = process.env.OPENROUTER_API_KEY || '';
const PORT           = process.env.PORT               || 3000;

if (!GROQ_KEY && !MISTRAL_KEY && !OPENROUTER_KEY) {
  console.error('[!] En az bir AI API anahtarı gerekli. .env dosyasını kontrol et.');
  process.exit(1);
}

// ── Rate Limiting ─────────────────────────────────────────────────────────────
const limiter = rateLimit({
  windowMs: 10 * 60 * 1000,
  max: 100,
  message: { ok: false, error: 'Çok fazla istek gönderildi, lütfen biraz bekleyin.' }
});

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(express.json({ limit: '10mb' }));
app.use('/api/', limiter);
app.use(express.static(path.join(__dirname, 'public')));

// ── Model Cascade Config ──────────────────────────────────────────────────────
const GROQ_RANKED = [
  'llama-3.3-70b-versatile',
  'deepseek-r1-distill-llama-70b',
  'qwen-2.5-32b',
  'mixtral-8x7b-32768',
  'llama3-70b-8192',
  'gemma2-9b-it',
  'llama-3.1-8b-instant',
  'llama3-8b-8192',
];

const MISTRAL_RANKED = [
  'mistral-large-latest',
  'mistral-small-latest',
  'open-mixtral-8x22b',
  'open-mixtral-8x7b',
  'open-mistral-7b',
];

const GROQ_HIGH_CUTOFF    = 4;
const MISTRAL_HIGH_CUTOFF = 2;

let orFreeModels = []; 

async function initOpenRouterModels() {
  if (!OPENROUTER_KEY) {
    console.warn('[~] OPENROUTER_API_KEY eksik — OpenRouter devre dışı.');
    return;
  }
  try {
    const r = await fetch('https://openrouter.ai/api/v1/models', {
      headers: { 'Authorization': `Bearer ${OPENROUTER_KEY}` }
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error?.message || r.statusText);

    orFreeModels = (d.data || [])
      .filter(m =>
        m.pricing &&
        (m.pricing.prompt === '0'     || m.pricing.prompt === 0) &&
        (m.pricing.completion === '0' || m.pricing.completion === 0)
      )
      .map(m => ({
        id:      m.id,
        name:    m.name || m.id,
        context: m.context_length || 0,
      }))
      .sort((a, b) => b.context - a.context);

    console.log(`[✓] OpenRouter: ${orFreeModels.length} ücretsiz model kaskada eklendi.`);
  } catch (err) {
    console.warn('[~] OpenRouter modelleri yüklenemedi:', err.message);
  }
}

function buildCascade(tier = 'all') {
  const list = [];
  if (GROQ_KEY) {
    const models = tier === 'high' ? GROQ_RANKED.slice(0, GROQ_HIGH_CUTOFF) : GROQ_RANKED;
    models.forEach(id => list.push({ provider: 'groq', id }));
  }
  if (MISTRAL_KEY) {
    const models = tier === 'high' ? MISTRAL_RANKED.slice(0, MISTRAL_HIGH_CUTOFF) : MISTRAL_RANKED;
    models.forEach(id => list.push({ provider: 'mistral', id }));
  }
  if (OPENROUTER_KEY && tier !== 'high') {
    orFreeModels.forEach(m => list.push({ provider: 'openrouter', id: m.id }));
  }
  return list;
}

// ── AI Provider Fonksiyonları ─────────────────────────────────────────────────

async function callGroq(messages, modelName) {
  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${GROQ_KEY}`
    },
    body: JSON.stringify({ model: modelName, messages, temperature: 0.6, max_tokens: 2048 }),
  });
  const d = await res.json();
  if (!res.ok) throw new Error(`Groq/${modelName}: ${d.error?.message || res.statusText}`);
  const content = d.choices?.[0]?.message?.content;
  if (!content) throw new Error(`Groq/${modelName}: boş yanıt`);
  return content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
}

async function callMistral(messages, modelName) {
  const res = await fetch('https://api.mistral.ai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${MISTRAL_KEY}`,
    },
    body: JSON.stringify({ model: modelName, messages, max_tokens: 2048 }),
  });
  const d = await res.json();
  if (!res.ok) throw new Error(`Mistral/${modelName}: ${d.message || res.statusText}`);
  const content = d.choices?.[0]?.message?.content;
  if (!content) throw new Error(`Mistral/${modelName}: boş yanıt`);
  return content.trim();
}

async function callOpenRouter(messages, modelId) {
  const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${OPENROUTER_KEY}`,
      'HTTP-Referer': 'https://cozergecer.app',
      'X-Title': 'ÇözerGeçer'
    },
    body: JSON.stringify({ model: modelId, messages, max_tokens: 2048, temperature: 0.6 })
  });
  const d = await res.json();
  if (!res.ok) throw new Error(`OpenRouter/${modelId}: ${d.error?.message || res.statusText}`);
  const content = d.choices?.[0]?.message?.content;
  if (!content) throw new Error(`OpenRouter/${modelId}: boş yanıt`);
  return content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
}

// Llama 4 Vision (Scout) — Görsel İşleme
async function callLlamaVision(imageB64, imageMime) {
  if (!OPENROUTER_KEY) throw new Error('OpenRouter API anahtarı eksik.');
  
  const dataUrl = `data:${imageMime};base64,${imageB64}`;
  const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${OPENROUTER_KEY}`,
      'HTTP-Referer': 'https://cozergecer.app',
      'X-Title': 'ÇözerGeçer'
    },
    body: JSON.stringify({
      model: 'meta-llama/llama-4-scout-17b-16e-instruct',
      messages: [{
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: dataUrl } },
          {
            type: 'text',
            text: [
              '⚠️ KESİN KURAL: Soruyu ÇÖZME. Cevap VERME. Yorum EKLEME. Adım YAZMA.',
              'Görevin YALNIZCA görseldeki içeriği olduğu gibi metne dönüştürmektir.',
              'Tüm matematiksel ifadeler için $$...$$ kullan. $...$, \\(...\\), \\[...\\] YASAK.',
              '',
              '## Çıktı sırası — her bölüm görselde varsa yaz, yoksa atla',
              '',
              '1. SORU METNİ',
              '   Görseldeki tüm yazıları (soru kökü, yönergeler, açıklamalar) kelimesi kelimesine aktar.',
              '   Seçenekler varsa A) B) C) D) E) sırasıyla yaz.',
              '',
              '2. ŞEKİL / DİYAGRAM (varsa)',
              '   "[Şekil: ...]" etiketiyle başla ve şu bilgileri ver:',
              '   - Şeklin türü (dik üçgen, çember, koordinat ekseni vb.)',
              '   - Köşe/nokta etiketleri: $$A$$, $$B$$, $$C$$',
              '   - Kenar uzunlukları: $$AB = 5$$, $$BC = 3$$',
              '   - Açılar: $$\\angle BAC = 60^\\circ$$, dik açı: $$\\angle B = 90^\\circ$$',
              '   - Koordinatlar: $$A(2,\\,3)$$',
              '   - Şekil üzerindeki tüm yazı ve sayılar',
              '',
              '⚠️ TEKRAR: Çözüm yazma. Cevap verme. Sadece görseldeki bilgiyi aktar.',
            ].join('\n'),
          }
        ],
      }],
      max_tokens: 2048,
    }),
  });

  const d = await res.json();
  if (!res.ok) throw new Error(`Llama-4-Scout: ${d.error?.message || res.statusText}`);
  const content = d.choices?.[0]?.message?.content;
  if (!content) throw new Error('Llama-4-Scout: Boş yanıt');
  return content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
}

// ── Kaskad Metin İşleyici ─────────────────────────────────────────────────────

const LATEX_RULE =
  'Matematiksel ifadeler için YALNIZCA $$...$$ kullan. $...$, \\(...\\) veya \\[...\\] ASLA kullanma.';

async function askWithFallback(messages, tier = 'all') {
  const cascade = buildCascade(tier);
  for (const { provider, id } of cascade) {
    try {
      console.log(`  -> [${provider}] ${id}`);
      if (provider === 'groq')       return await callGroq(messages, id);
      if (provider === 'mistral')    return await callMistral(messages, id);
      if (provider === 'openrouter') return await callOpenRouter(messages, id);
    } catch (err) {
      console.warn(`     [✗] ${err.message} — sonraki model...`);
    }
  }
  throw new Error('Tüm modeller başarısız oldu.');
}

function wrap(fn) {
  return async (req, res) => {
    try { await fn(req, res); }
    catch (err) {
      console.error('[Hata]', err.message);
      res.status(500).json({ ok: false, error: err.message });
    }
  };
}

// ── Routes ────────────────────────────────────────────────────────────────────

app.get('/api/health', (req, res) => {
  res.json({ ok: true, wolfram: !!WOLFRAM_ID });
});

app.post('/api/read-image', wrap(async (req, res) => {
  const { imageB64, imageMime } = req.body;
  if (!imageB64) return res.status(400).json({ ok: false, error: 'Görsel yok.' });
  const text = await callLlamaVision(imageB64, imageMime);
  res.json({ ok: true, text });
}));

app.post('/api/classify', wrap(async (req, res) => {
  const { question } = req.body;
  const safeQuestion = question.replace(/[\r\n\t]+/g, ' ').replace(/"/g, "'").trim();
  const system = "Sen bir JSON API'sın. SADECE ham JSON döndür.";
  const prompt = `Aşağıdaki soruyu analiz et ve SADECE şu şemaya uygun bir JSON nesnesi döndür:\n` +
                 `{"isMath":true,"wolframQuery":"ingilizce wolfram sorgusu veya boş string","type":"algebra|calculus|geometry|trigonometry|statistics|physics|chemistry|biology|history|literature|general"}\n\n` +
                 `Soru: "${safeQuestion}"`;
  const raw = await askWithFallback([{ role: 'system', content: system }, { role: 'user', content: prompt }]);
  const match = raw.match(/\{[\s\S]*?\}/);
  if (!match) throw new Error('Model geçerli JSON döndürmedi.');
  const parsed = JSON.parse(match[0].replace(/[\x00-\x1F\x7F]/g, ' '));
  res.json({ ok: true, ...parsed });
}));

app.post('/api/solve', wrap(async (req, res) => {
  const { question, isMath } = req.body;
  const system = `Sen zeki bir çözüm ortağısın. ${LATEX_RULE}`;
  const user   = isMath ? `Sorum: ${question}` : `Açıkla: ${question}`;
  const answer = await askWithFallback([{ role: 'system', content: system }, { role: 'user', content: user }], 'high');
  res.json({ ok: true, answer });
}));

app.post('/api/explain', wrap(async (req, res) => {
  const { question, answer } = req.body;
  const system = `Basitçe mantığını anlat. ${LATEX_RULE}`;
  const explanation = await askWithFallback([
    { role: 'system', content: system },
    { role: 'user',   content: `Soru: ${question}\nCevap: ${answer}\n\nKısaca anlat.` }
  ], 'high');
  res.json({ ok: true, explanation });
}));

app.post('/api/wolfram', wrap(async (req, res) => {
  const q = req.body.q;
  if (!WOLFRAM_ID) return res.status(501).json({ ok: false, error: 'Wolfram devre dışı.' });
  const url = `https://api.wolframalpha.com/v2/query?appid=${WOLFRAM_ID}&input=${encodeURIComponent(q)}&output=json&format=plaintext&units=metric`;
  const r = await fetch(url);
  const data = await r.json();
  const pods = data?.queryresult?.pods ?? [];
  if (!pods.length) return res.status(422).json({ ok: false, error: 'Sonuç bulunamadı.' });
  const chosen = pods.find(p => p.title === 'Result' || p.id === 'result') || pods[0];
  const result = (chosen.subpods ?? []).map(s => s.plaintext?.trim()).filter(Boolean).join(' | ');
  res.json({ ok: true, result });
}));

// ── Init & Start ──────────────────────────────────────────────────────────────
initOpenRouterModels().then(() => {
  app.listen(PORT, () => {
    console.log(`\n  🚀 ÇözerGeçer Sunucusu Hazır`);
    console.log(`  ➜ Vizyon  : Llama 4 Scout (17B)`);
    console.log(`  ➜ Wolfram : ${WOLFRAM_ID ? 'AKTİF' : 'DEVRE DIŞI'}\n`);
  });
});