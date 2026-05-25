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
//   Her liste kendi içinde en iyiden en kötüye sıralıdır.
//   Genel sıra: Groq → Mistral → OpenRouter (ücretsiz)

const GROQ_RANKED = [
  'llama-3.3-70b-versatile',       // En güçlü Groq modeli
  'deepseek-r1-distill-llama-70b', // Akıl yürütme odaklı
  'qwen-2.5-32b',                  // Çok dilli, güçlü
  'mixtral-8x7b-32768',            // MoE mimarisi
  'llama3-70b-8192',               // Stabil 70B
  'gemma2-9b-it',                  // Google, dengeli
  'llama-3.1-8b-instant',          // Hızlı, hafif
  'llama3-8b-8192',                // Yedek küçük model
];

const MISTRAL_RANKED = [
  'mistral-large-latest',  // Mistral'in en iyisi
  'mistral-small-latest',  // Hızlı ve dengeli
  'open-mixtral-8x22b',    // Büyük MoE
  'open-mixtral-8x7b',     // Klasik MoE
  'open-mistral-7b',       // Temel model
];

// OpenRouter ücretsiz modeller — sunucu başlarken çekilir
let orFreeModels = []; // [{ id, name, context }]

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
      // Büyük context penceresi genellikle daha güçlü modele işaret eder
      .sort((a, b) => b.context - a.context);

    console.log(`[✓] OpenRouter: ${orFreeModels.length} ücretsiz model kaskada eklendi.`);
  } catch (err) {
    console.warn('[~] OpenRouter modelleri yüklenemedi:', err.message);
  }
}

// Tam kaskad listesini döndür (her çağrıda güncel)
function buildCascade() {
  const list = [];
  if (GROQ_KEY)       GROQ_RANKED.forEach(id => list.push({ provider: 'groq',       id }));
  if (MISTRAL_KEY)    MISTRAL_RANKED.forEach(id => list.push({ provider: 'mistral',  id }));
  if (OPENROUTER_KEY) orFreeModels.forEach(m  => list.push({ provider: 'openrouter', id: m.id }));
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

// Mistral Vision (Pixtral) — kaskad dışı, sabit model
async function callMistralVision(imageB64, imageMime) {
  const dataUrl = `data:${imageMime};base64,${imageB64}`;
  const res = await fetch('https://api.mistral.ai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${MISTRAL_KEY}`,
    },
    body: JSON.stringify({
      model: 'pixtral-12b-2409',
      messages: [{
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: dataUrl } },
          {
            type: 'text',
            text:
              'Görseldeki yazıyı ve matematiksel ifadeleri OLDUĞU GİBİ oku ve yaz. ' +
              'SADECE metni aktar; soruyu çözme, cevaplamaya çalışma, yorum ekleme. ' +
              'Satır içi matematik için $...$, blok/display matematik için $$...$$ kullan. ' +
              'Başka hiçbir şey yazma.'
          }
        ],
      }],
      max_tokens: 1024,
    }),
  });
  const d = await res.json();
  if (!res.ok) throw new Error(`Mistral Vizyon: ${d.message || res.statusText}`);
  return d.choices[0].message.content.trim();
}

// ── Kaskad Metin İşleyici ─────────────────────────────────────────────────────
//   En iyi modelden başlar, hata alırsa bir sonrakine geçer.

const LATEX_RULE =
  'Matematiksel ifadeler için MUTLAKA $...$ (satır içi) veya $$...$$ (blok/display) kullan. ' +
  '\\(...\\) veya \\[...\\] formatını ASLA kullanma.';

async function askWithFallback(messages) {
  const cascade = buildCascade();
  if (cascade.length === 0) throw new Error('Hiçbir AI sağlayıcısı yapılandırılmamış.');

  for (const { provider, id } of cascade) {
    try {
      console.log(`  -> [${provider}] ${id}`);
      if (provider === 'groq')       return await callGroq(messages, id);
      if (provider === 'mistral')    return await callMistral(messages, id);
      if (provider === 'openrouter') return await callOpenRouter(messages, id);
    } catch (err) {
      console.warn(`     [✗] ${err.message} — sonraki model deneniyor…`);
    }
  }
  throw new Error('Tüm modeller başarısız oldu. Lütfen tekrar deneyin.');
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
  res.json({
    ok: true,
    status: 'online',
    cascade_size: buildCascade().length,
    openrouter_models: orFreeModels.length,
    wolfram: !!WOLFRAM_ID,
  });
});

// Kaskad listesini istemciye sun (bilgi/debug amaçlı)
app.get('/api/cascade', (req, res) => {
  res.json({ ok: true, models: buildCascade() });
});

// Görselden metin okuma
app.post('/api/read-image', wrap(async (req, res) => {
  const { imageB64, imageMime } = req.body;
  if (!imageB64) return res.status(400).json({ ok: false, error: 'Görsel verisi bulunamadı.' });
  const text = await callMistralVision(imageB64, imageMime);
  res.json({ ok: true, text });
}));

// Soru sınıflandırma
app.post('/api/classify', wrap(async (req, res) => {
  const { question } = req.body;
  const prompt =
    `Aşağıdaki soruyu analiz et ve YALNIZCA geçerli JSON döndür (markdown ya da kod bloğu kullanma).\n` +
    `Soru: "${question}"\n` +
    `{"isMath":true,"wolframQuery":"ingilizce wolfram sorgusu veya boş string","type":"algebra|calculus|geometry|trigonometry|statistics|physics|chemistry|biology|history|literature|general"}`;
  const raw       = await askWithFallback([{ role: 'user', content: prompt }]);
  const cleanJson = raw.replace(/```json|```/g, '').trim();
  res.json({ ok: true, ...JSON.parse(cleanJson) });
}));

// Çözüm
app.post('/api/solve', wrap(async (req, res) => {
  const { question, isMath } = req.body;
  const system = `Sen zeki ve samimi bir çözüm ortağısın. "Merhaba öğrenciler" gibi klişeleri asla kullanma, bir dost gibi konuş. ${LATEX_RULE}`;
  const user   = isMath
    ? `Şu matematik sorusunu adım adım çöz:\n\n${question}`
    : `Şu soruyu samimi bir dille açıkla:\n\n${question}`;
  const answer = await askWithFallback([
    { role: 'system', content: system },
    { role: 'user',   content: user },
  ]);
  res.json({ ok: true, answer });
}));

// Açıklama
app.post('/api/explain', wrap(async (req, res) => {
  const { question, answer } = req.body;
  const system = `Samimi bir abi/abla gibi davran; yapaylıktan kaçın, olayı basitçe mantığıyla anlat. ${LATEX_RULE}`;
  const explanation = await askWithFallback([
    { role: 'system', content: system },
    { role: 'user',   content: `Soru: ${question}\nCevap: ${answer}\n\nBu çözümü bana kısaca anlatır mısın?` },
  ]);
  res.json({ ok: true, explanation });
}));

// Wolfram Alpha
app.get('/api/wolfram', wrap(async (req, res) => {
  const q = req.query.q;
  if (!q) return res.status(400).json({ ok: false, error: 'Sorgu eksik.' });
  const url  = `https://api.wolframalpha.com/v1/result?appid=${WOLFRAM_ID}&i=${encodeURIComponent(q)}`;
  const r    = await fetch(url);
  const text = await r.text();
  res.json({ ok: true, result: text });
}));

// ── Init & Start ──────────────────────────────────────────────────────────────
initOpenRouterModels().then(() => {
  app.listen(PORT, () => {
    const cascade = buildCascade();
    console.log(`\n  🚀 ÇözerGeçer Sunucusu Hazır`);
    console.log(`  ➜ Adres   : http://localhost:${PORT}`);
    console.log(`  ➜ Kaskad  : ${cascade.length} model (en iyiden en kötüye)`);
    console.log(`  ➜ Vizyon  : Pixtral 12B`);
    console.log(`  ➜ Wolfram : ${WOLFRAM_ID ? 'AKTİF' : 'DEVRE DIŞI'}\n`);
    cascade.slice(0, 8).forEach((m, i) =>
      console.log(`    ${String(i + 1).padStart(2, ' ')}. [${m.provider.padEnd(10)}] ${m.id}`)
    );
    if (cascade.length > 8)
      console.log(`       … ve ${cascade.length - 8} model daha (OpenRouter ücretsiz)`);
    console.log();
  });
});