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

// Çözüm için kullanılacak minimum güçlü model eşiği
//   'high' tier: Groq'ta ilk 4 (70B sınıfı), Mistral'de ilk 2, OpenRouter yok
const GROQ_HIGH_CUTOFF    = 4;   // llama-3.3-70b / deepseek-r1 / qwen-32b / mixtral-8x7b
const MISTRAL_HIGH_CUTOFF = 2;   // mistral-large + mistral-small

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

// Kaskad listesini döndür
//   tier = 'all'  → tüm modeller (sınıflandırma, hafif görevler)
//   tier = 'high' → sadece güçlü modeller (çözüm, açıklama)
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
              '   - Özel elemanlar: $$r = 4$$, $$h = 6$$',
              '   - Koordinatlar: $$A(2,\\,3)$$',
              '   - Paralel/eşit işaretler: $$AB \\parallel CD$$, $$AB = CD$$',
              '   - Vektörler: $$\\vec{AB}$$',
              '   - Şekil üzerindeki tüm yazı ve sayılar',
              '',
              '⚠️ TEKRAR: Çözüm yazma. Cevap verme. Sadece görseldeki bilgiyi aktar.',
            ].join('\n'),
          }
        ],
      }],
      max_tokens: 1536,
    }),
  });
  const d = await res.json();
  if (!res.ok) throw new Error(`Mistral Vizyon: ${d.message || res.statusText}`);
  return d.choices[0].message.content.trim();
}

// ── Kaskad Metin İşleyici ─────────────────────────────────────────────────────
//   En iyi modelden başlar, hata alırsa bir sonrakine geçer.

const LATEX_RULE =
  'Matematiksel ifadeler için YALNIZCA $$...$$ kullan (hem satır içi hem blok/display). ' +
  '$...$, \\(...\\) veya \\[...\\] formatlarını ASLA kullanma.';

async function askWithFallback(messages, tier = 'all') {
  const cascade = buildCascade(tier);
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

  // Kontrol karakterlerini temizle: \n \r \t vb. prompt içinde JSON'u bozar
  const safeQuestion = question
    .replace(/[\r\n\t]+/g, ' ')  // satır sonu / sekme → boşluk
    .replace(/"/g, "'")           // çift tırnak → tek tırnak (prompt'u kırmaz)
    .trim();

  const system =
    "Sen bir JSON API'sın. Yalnızca ham JSON nesnesi döndür. " +
    'Markdown, kod bloğu (```), açıklama veya başka HİÇBİR şey yazma.';

  const prompt =
    `Aşağıdaki soruyu analiz et ve SADECE şu şemaya uygun bir JSON nesnesi döndür:\n` +
    `{"isMath":true,"wolframQuery":"ingilizce wolfram sorgusu veya boş string","type":"algebra|calculus|geometry|trigonometry|statistics|physics|chemistry|biology|history|literature|general"}\n\n` +
    `Soru: "${safeQuestion}"`;

  const raw = await askWithFallback([
    { role: 'system', content: system },
    { role: 'user',   content: prompt },
  ]);

  // Modelin eklediği açıklama/markdown ne olursa olsun { } bloğunu yakala
  const match = raw.match(/\{[\s\S]*?\}/);
  if (!match) {
    console.error('[classify] Geçersiz model yanıtı:', raw.slice(0, 200));
    throw new Error('Model geçerli JSON döndürmedi.');
  }

  // Model yanıtındaki kontrol karakterlerini temizle (JSON.parse bunlara tahammül etmez)
  const cleanJson = match[0].replace(/[\x00-\x1F\x7F]/g, ' ');

  const parsed = JSON.parse(cleanJson);
  res.json({ ok: true, ...parsed });
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
  ], 'high');
  res.json({ ok: true, answer });
}));

// Açıklama
app.post('/api/explain', wrap(async (req, res) => {
  const { question, answer } = req.body;
  const system = `Samimi bir abi/abla gibi davran; yapaylıktan kaçın, olayı basitçe mantığıyla anlat. ${LATEX_RULE}`;
  const explanation = await askWithFallback([
    { role: 'system', content: system },
    { role: 'user',   content: `Soru: ${question}\nCevap: ${answer}\n\nBu çözümü bana kısaca anlatır mısın?` },
  ], 'high');
  res.json({ ok: true, explanation });
}));

// Wolfram Alpha
app.post('/api/wolfram', wrap(async (req, res) => {
  const q = req.body.q;
  if (!q) return res.status(400).json({ ok: false, error: 'Sorgu eksik.' });

  // /v2/query — tam sonuç API'si; Result pod'unu önce dener,
  // bulamazsa DecimalApproximation veya ilk sayısal pod'u alır.
  const url = `https://api.wolframalpha.com/v2/query?appid=${WOLFRAM_ID}` +
              `&input=${encodeURIComponent(q)}&output=json&format=plaintext` +
              `&units=metric&podstate=Result__Step-by-step+solution`;

  const r = await fetch(url);
  if (!r.ok) throw new Error(`Wolfram HTTP ${r.status}`);
  const data = await r.json();

  const pods = data?.queryresult?.pods ?? [];
  if (!pods.length || data.queryresult.success === false) {
    return res.status(422).json({ ok: false, error: 'Wolfram bu sorguyu anlayamadı.' });
  }

  // Tercih sırası: Result > Simplification > DecimalApproximation > ilk pod
  const PREFERRED = ['Result', 'Simplification', 'Value', 'DecimalApproximation'];
  let chosen = null;
  for (const title of PREFERRED) {
    chosen = pods.find(p => p.title === title || p.id === title.toLowerCase());
    if (chosen) break;
  }
  if (!chosen) chosen = pods[0]; // en azından bir şey göster

  // Subpod'lardan düz metni topla
  const lines = (chosen.subpods ?? [])
    .map(s => s.plaintext?.trim())
    .filter(Boolean);

  if (!lines.length) {
    return res.status(422).json({ ok: false, error: 'Wolfram sonuç metni boş.' });
  }

  const result = lines.join('  |  ');
  console.log(`  [Wolfram] pod="${chosen.title}" → ${result}`);
  res.json({ ok: true, result, pod: chosen.title });
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