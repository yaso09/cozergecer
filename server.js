'use strict';
require('dotenv').config();
const express = require('express');
const path    = require('path');
const rateLimit = require('express-rate-limit');

const app = express();

// ── Config ──────────────────────────────────────────────────────────────────
const GROQ_KEY    = process.env.GROQ_API_KEY    || '';
const MISTRAL_KEY = process.env.MISTRAL_API_KEY || '';
const WOLFRAM_ID  = process.env.WOLFRAM_APP_ID  || '';
const PORT        = process.env.PORT            || 3000;

if (!GROQ_KEY || !MISTRAL_KEY) {
  console.error('[!] API Anahtarları eksik! .env dosyasını kontrol et.');
  process.exit(1);
}

// ── Rate Limiting (API Koruma) ───────────────────────────────────────────────
const limiter = rateLimit({
  windowMs: 10 * 60 * 1000,
  max: 100,
  message: { ok: false, error: "Çok fazla istek gönderildi, lütfen biraz bekleyin." }
});

// ── Middleware ───────────────────────────────────────────────────────────────
app.use(express.json({ limit: '10mb' }));
app.use('/api/', limiter);
app.use(express.static(path.join(__dirname, 'public')));

// ── AI Helpers (Kademeli Yedekleme / Fallback) ───────────────────────────────

async function callGroq(messages, modelName) {
  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${GROQ_KEY}`
    },
    body: JSON.stringify({
      model: modelName,
      messages: messages,
      temperature: 0.6,
      max_tokens: 2048
    }),
  });

  const d = await res.json();
  if (!res.ok) {
    // Rate limit veya Kota hatası kontrolü
    if (res.status === 429 || (d.error && d.error.type?.includes('quota'))) {
       throw new Error(`LIMIT_REACHED: ${modelName}`);
    }
    throw new Error(`Groq Hatası (${modelName}): ${d.error?.message || res.statusText}`);
  }

  let content = d.choices[0].message.content;
  // Qwen yedek olarak devreye girerse <think> bloklarını temizlemek için:
  return content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
}

async function callMistralText(messages) {
  const res = await fetch('https://api.mistral.ai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${MISTRAL_KEY}`,
    },
    body: JSON.stringify({
      model: "mistral-large-latest",
      messages: messages,
      max_tokens: 2048,
    }),
  });
  const d = await res.json();
  if (!res.ok) throw new Error("Mistral Fallback Hatası: " + (d.message || res.statusText));
  return d.choices[0].message.content.trim();
}

// Ana Metin İşleme (Llama 3.3 -> Qwen -> Mistral)
async function askTextAI(messages) {
  try {
    console.log("-> Deneniyor: Llama 3.3 70B...");
    return await callGroq(messages, "llama-3.3-70b-versatile");
  } catch (err) {
    if (err.message.includes("LIMIT_REACHED")) {
      try {
        console.warn("-> Llama limiti doldu. Deneniyor: Qwen 32B...");
        return await callGroq(messages, "qwen-2.5-32b");
      } catch (err2) {
        console.error("-> Groq limitleri tükendi. Deneniyor: Mistral Large...");
        return await callMistralText(messages);
      }
    }
    throw err;
  }
}

// Mistral Vision (Pixtral)
async function askMistralVision(imageB64, imageMime) {
  const dataUrl = `data:${imageMime};base64,${imageB64}`;
  const res = await fetch('https://api.mistral.ai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${MISTRAL_KEY}`,
    },
    body: JSON.stringify({
      model: "pixtral-12b-2409",
      messages: [{
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: dataUrl } },
          { type: 'text', text: "Bu görseldeki soruyu/metni oku. Matematiksel ifadeleri LaTeX ($...$) olarak yaz. Başka hiçbir açıklama ekleme." }
        ],
      }],
      max_tokens: 1024,
    }),
  });
  const d = await res.json();
  if (!res.ok) throw new Error(`Mistral Vizyon Hatası: ${d.message || res.statusText}`);
  return d.choices[0].message.content.trim();
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

// ── Routes ───────────────────────────────────────────────────────────────────

// Health Check
app.get('/api/health', (req, res) => {
  res.json({
    ok: true,
    status: "online",
    primary_model: "Llama 3.3 70B",
    vision: "Pixtral 12B",
    wolfram: !!WOLFRAM_ID
  });
});

// Görselden okuma
app.post('/api/read-image', wrap(async (req, res) => {
  const { imageB64, imageMime } = req.body;
  if (!imageB64) return res.status(400).json({ ok: false, error: "Görsel verisi bulunamadı." });
  const text = await askMistralVision(imageB64, imageMime);
  res.json({ ok: true, text });
}));

// Sınıflandırma
app.post('/api/classify', wrap(async (req, res) => {
  const { question } = req.body;
  const prompt = `Analiz et ve sadece saf JSON döndür. Markdown etiketleri kullanma.\nSoru: "${question}"\n{"isMath":true/false, "wolframQuery":"ingilizce matematik sorgusu", "type":"ders"}`;
  const raw = await askTextAI([{ role: 'user', content: prompt }]);
  const cleanJson = raw.replace(/```json|```/g, "").trim();
  res.json({ ok: true, ...JSON.parse(cleanJson) });
}));

// Çözüm
app.post('/api/solve', wrap(async (req, res) => {
  const { question, isMath } = req.body;
  const systemRole = "Sen zeki ve samimi bir çözüm ortağısın. 'Merhaba öğrenciler' gibi klişeleri asla kullanma. Bir dost gibi konuş.";
  const userPrompt = isMath ? `Şu matematik sorusunu LaTeX kullanarak adım adım çöz:\n\n${question}` : `Şu soruyu samimi bir dille açıkla:\n\n${question}`;
  const answer = await askTextAI([
    { role: 'system', content: systemRole },
    { role: 'user', content: userPrompt }
  ]);
  res.json({ ok: true, answer });
}));

// Açıklama
app.post('/api/explain', wrap(async (req, res) => {
  const { question, answer } = req.body;
  const systemRole = "Samimi bir abi/abla gibi davran. Yapaylıktan kaçın, olayı basitçe mantığıyla anlat.";
  const explanation = await askTextAI([
    { role: 'system', content: systemRole },
    { role: 'user', content: `Soru: ${question}\nCevap: ${answer}\n\nBu çözümü bana basitçe anlatır mısın?` }
  ]);
  res.json({ ok: true, explanation });
}));

// Wolfram Alpha
app.get('/api/wolfram', wrap(async (req, res) => {
  const q = req.query.q;
  if (!q) return res.status(400).json({ ok: false, error: "Sorgu eksik." });
  const url = `https://api.wolframalpha.com/v1/result?appid=${WOLFRAM_ID}&i=${encodeURIComponent(q)}`;
  const r = await fetch(url);
  const text = await r.text();
  res.json({ ok: true, result: text });
}));

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n  🚀 ÇözerGeçer Sunucusu Hazır`);
  console.log(`  ➜ Adres    : http://localhost:${PORT}`);
  console.log(`  ➜ Health   : http://localhost:${PORT}/api/health`);
  console.log(`  ➜ Ana Zeka : Llama 3.3 70B (Groq)`);
  console.log(`  ➜ Fallback : Qwen 32B & Mistral Large`);
  console.log(`  ➜ Vizyon   : Pixtral (Mistral)`);
  console.log(`  ➜ Durum    : ÇALIŞIYOR\n`);
});