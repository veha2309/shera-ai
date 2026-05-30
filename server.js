const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { Ollama } = require('ollama');
const { ChromaClient } = require('chromadb');
const { OllamaEmbeddingFunction } = require('@chroma-core/ollama');
const { LRUCache } = require('lru-cache');

/**
 * Shera AI - Hybrid Chroma + GraphRAG Backend Server
 * * ULTRA-LIGHTWEIGHT CONFIGURATION:
 * Optimizations applied for Qwen 2 (0.5b):
 * 1. Tiny 350MB footprint for ultra-low latency on weak VMs.
 * 2. Extreme sampling speed (Temp 0.7, Top P 0.8).
 * 3. LLM-Bypass Extraction: Skips LLM for short/simple subject queries.
 * 4. Graph traversal result cache.
 * 5. PROACTIVE MEMORY SWEEPER: Protects Node.js heap from OOM crashes.
 */

const app = express();
const port = 3000;
const host = '0.0.0.0';

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});
process.on('uncaughtException', (err) => {
    console.error('Uncaught Exception:', err);
});

// ─── Models ───────────────────────────────────────────────────────────────────
const EMBED_MODEL = 'nomic-embed-text';
// const CHAT_MODEL = 'gemma2:2b';          // Main answer model (quality)
const CHAT_MODEL = 'shera-llama:latest';          // Main answer model (quality)
const EXTRACTION_MODEL = 'qwen2:0.5b';   // Subject extractor (speed — 352MB vs 1.6GB)

// ─── OPTIMIZED Hindi Term Glossary ────────────────────────────────────────────
const HINDI_DICT = {
    // Habitats
    'dry forest': 'सूखे जंगल',
    'dry forests': 'सूखे जंगल',
    'tropical forest': 'उष्णकटिबंधीय वन',
    'tropical forests': 'उष्णकटिबंधीय वन',
    'tropical rainforest': 'उष्णकटिबंधीय वन',
    'tropical rainforests': 'उष्णकटिबंधीय वन',
    'scrubland': 'झाड़ीदार भूमि',
    'scrublands': 'झाड़ीदार भूमि',
    'grassland': 'घास के मैदान',
    'grasslands': 'घास के मैदान',
    'wetland': 'आर्द्रभूमि',
    'wetlands': 'आर्द्रभूमि',
    'mangrove': 'मैंग्रोव वन',
    'mangroves': 'मैंग्रोव वन',
    'savannah': 'सवाना',
    'savannahs': 'सवाना',
    'desert': 'रेगिस्तान',
    'deserts': 'रेगिस्तान',
    'river': 'नदियाँ',
    'rivers': 'नदियाँ',
    'lake': 'झीलें',
    'lakes': 'झीलें',
    'mountain': 'पहाड़',
    'mountains': 'पहाड़',

    // Diet
    'deer': 'हिरण',
    'antelope': 'मृग',
    'wild boar': 'जंगली सूअर',
    'buffalo': 'भैंस',
    'livestock': 'पालतू पशु',
    'fish': 'मछली',
    'fruit': 'फल',
    'fruits': 'फल',
    'leaf': 'पत्तियाँ',
    'leaves': 'पत्तियाँ',
    'grass': 'घास',
    'insect': 'कीड़े',
    'insects': 'कीड़े',
    'bark': 'छाल',
    'bamboo': 'बाँस',
    'small mammal': 'छोटे स्तनधारी',
    'small mammals': 'छोटे स्तनधारी',
    'bird': 'पक्षी',
    'birds': 'पक्षी',
    'egg': 'अंडे',
    'eggs': 'अंडे',

    // Biology
    'mammal': 'स्तनधारी',
    'reptile': 'सरीसृप',
    'reptiles': 'सरीसृप',
    'amphibian': 'उभयचर',
    'nocturnal': 'रात्रिचर',
    'diurnal': 'दिनचर',
    'herbivore': 'शाकाहारी',
    'carnivore': 'मांसाहारी',
    'omnivore': 'सर्वाहारी',
    'predator': 'शिकारी',
    'prey': 'शिकार',

    // Conservation
    'critically endangered': 'गंभीर रूप से संकटग्रस्त',
    'endangered': 'संकटग्रस्त',
    'vulnerable': 'संवेदनशील',
    'wildlife protection': 'वन्यजीव संरक्षण',

    // Animal names (base)
    'lion': 'शेर',
    'tiger': 'बाघ',
    'elephant': 'हाथी',
    'bear': 'भालू',
    'macaque': 'मकाक',
    'deer': 'हिरण',
    'crocodile': 'मगरमच्छ',
    'python': 'अजगर',
    'snake': 'साँप',

    // Animal names (full)
    'asiatic lion': 'एशियाई शेर',
    'white tiger': 'सफ़ेद बाघ',
    'bengal tiger': 'बंगाल बाघ',
    'indian elephant': 'भारतीय हाथी',
    'sloth bear': 'भालू',
    'lion tailed macaque': 'सिंहपुच्छी मकाक',
    'lion-tailed macaque': 'सिंहपुच्छी मकाक',
    'rhesus macaque': 'रीसस बंदर',
    'marsh crocodile': 'दलदली मगरमच्छ',
    'spectacled cobra': 'चशमेदार नाग',
    'indian rock python': 'भारतीय अजगर',
    'spotted deer': 'चीतल हिरण',
    'indian peafowl': 'भारतीय मोर',
    'white peafowl': 'सफ़ेद मोर',
    'common rat snake': 'साधारण साँप',
    'striped hyena': 'धारीदार लकड़बग्घा',
    'hyena': 'लकड़बग्घा',
    'gir forest': 'गिर वन',
    'wild cattle': 'जंगली मवेशी',
    'black headed ibis': 'काला सिर वाला इबिस',
    'bengal monitor': 'बंगाल मॉनिटर छिपकली',
    'emu': 'इमू',
    'black headed caique': 'काले सिर वाला काइक',
    'african grey parrot': 'अफ़्रीकी ग्रे तोता',
    'rose ringed parakeet': 'गुलाबी गर्दन वाला तोता',
    'silver pheasant': 'सिल्वर तीतर',
    'gharial': 'घड़ियाल',
    'indian birds': 'भारतीय पक्षी',
    'rhinoceros': 'गैंडा',
    'hippopotamus': 'दरियाई घोड़ा',
    'chimpanzee': 'चिंपैंजी',

    // Common descriptors
    'large': 'बड़ा',
    'small': 'छोटा',
    'powerful': 'शक्तिशाली',
    'beautiful': 'सुंदर',
    'found in': 'में पाया जाता है',
    'native to': 'का मूल निवासी',
    'located at': 'पर स्थित है',
    'enclosure': 'स्थान',
    'beat number': 'क्षेत्र',
    'location': 'स्थान',
    'weight': 'वजन',
    'lifespan': 'जीवनकाल'
};

// Sort keys descending by length to prevent partial word collisions
const glossaryKeys = Object.keys(HINDI_DICT).sort((a, b) => b.length - a.length);

// Escape standard string characters for safe regex injection
const escapeRegex = (str) => str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// Compile ONE highly-optimized regex pattern on server startup
const GLOSSARY_REGEX = new RegExp(`\\b(${glossaryKeys.map(escapeRegex).join('|')})\\b`, 'gi');

function applyHindiGlossary(text) {
    if (!text) return text;
    // Single-pass string replacement
    return text.replace(GLOSSARY_REGEX, (match) => {
        return HINDI_DICT[match.toLowerCase()] || match;
    });
}

function logResources(label) {
    const mem = process.memoryUsage();
    console.log(`[RESOURCES] ${label} - RSS: ${(mem.rss / 1024 / 1024).toFixed(2)}MB, Heap: ${(mem.heapUsed / 1024 / 1024).toFixed(2)}MB`);
}

// ─── Clients ──────────────────────────────────────────────────────────────────
const ollama = new Ollama();
const chroma = new ChromaClient({ path: 'http://localhost:8000' });
const embedder = new OllamaEmbeddingFunction({
    url: 'http://127.0.0.1:11434',
    model: 'nomic-embed-text'
});

// ─── Global Caches & Proactive Sweeper (Bulletproofed with LRU) ────────────────────────────────────────
const embeddingCache = new LRUCache({
    max: 1000, // Absolute cap. Evicts the oldest if it hits 1001.
    ttl: 1000 * 60 * 60, // 1 hour
});

const responseCache = new LRUCache({
    max: 500, // Capping responses protects heap memory during traffic spikes
    ttl: 1000 * 60 * 60, // 1 hour
});

const chromaSearchCache = new LRUCache({
    max: 500,
    ttl: 1000 * 60 * 30, // 30 minutes
});

const graphTraversalCache = new LRUCache({
    max: 200,
    ttl: 1000 * 60 * 60, // 1 hour
});

const RESPONSE_CACHE_TTL = 1000 * 60 * 60; // 1 hour
const CHROMA_CACHE_TTL = 1000 * 60 * 30;   // 30 minutes
const MAX_EMBEDDING_CACHE_SIZE = 1000;

// SWEEPER: Runs every 10 minutes to purge dead memory and enforce limits
setInterval(() => {
    const now = Date.now();

    let resPurged = 0;
    for (const [key, entry] of responseCache.entries()) {
        if (now - entry.ts > RESPONSE_CACHE_TTL) {
            responseCache.delete(key);
            resPurged++;
        }
    }

    let chromaPurged = 0;
    for (const [key, entry] of chromaSearchCache.entries()) {
        if (now - entry.ts > CHROMA_CACHE_TTL) {
            chromaSearchCache.delete(key);
            chromaPurged++;
        }
    }

    if (embeddingCache.size > MAX_EMBEDDING_CACHE_SIZE) {
        console.log(`[RESOURCES] Embedding cache hit cap (${embeddingCache.size}). Flushing to prevent OOM.`);
        embeddingCache.clear();
    }

    if (resPurged > 0 || chromaPurged > 0) {
        console.log(`[SWEEPER] Purged ${resPurged} responses and ${chromaPurged} chroma searches.`);
    }
}, 10 * 60 * 1000);

// Cache Accessors
function getCachedResponse(key) {
    return responseCache.get(key) || null;
}

function setCachedResponse(key, data) {
    responseCache.set(key, data);
}

function getCachedChromaSearch(cacheKey) {
    return chromaSearchCache.get(cacheKey) || null;
}

function setCachedChromaSearch(cacheKey, data) {
    chromaSearchCache.set(cacheKey, data);
}

async function getCachedEmbedding(text) {
    let embedding = embeddingCache.get(text);
    if (embedding) return embedding;

    try {
        const resp = await ollama.embed({ model: EMBED_MODEL, input: text, keep_alive: '1h' });
        embedding = resp.embeddings[0];
        embeddingCache.set(text, embedding);
        return embedding;
    } catch (e) {
        console.error(`Embedding error for "${text}":`, e.message);
        return null;
    }
}

let collection;

// ─── Middleware ───────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json());

// ─── Graph Store ──────────────────────────────────────────────────────────────
function loadAllGraphs() {
    const storeDir = path.join(__dirname, 'graph_data');
    if (!fs.existsSync(storeDir)) {
        console.warn('No graph_data found. Running in vector-only mode.');
        return { nodes: [], edges: [] };
    }

    const merged = { nodes: [], edges: [] };
    const nodeIds = new Set();
    const graphFiles = fs.readdirSync(storeDir).filter(f => f.endsWith('_graph.json'));

    for (const file of graphFiles) {
        try {
            const g = JSON.parse(fs.readFileSync(path.join(storeDir, file), 'utf8'));
            for (const node of g.nodes) {
                if (!nodeIds.has(node.id)) {
                    nodeIds.add(node.id);
                    merged.nodes.push({ ...node, source_graph: file });
                }
            }
            for (const edge of g.edges) {
                merged.edges.push({ ...edge, source_graph: file });
            }
        } catch (e) {
            console.error(`Failed to load graph: ${file}`);
        }
    }

    console.log(`Loaded ${merged.nodes.length} nodes, ${merged.edges.length} edges from ${graphFiles.length} graphs.`);
    return merged;
}

let graph = loadAllGraphs();
const nodeMap = {};
for (const node of graph.nodes) if (node.id) nodeMap[node.id] = node;

const adjacencyMap = {};
for (const edge of graph.edges) {
    if (!edge.source || !edge.target) continue;
    if (!adjacencyMap[edge.source]) adjacencyMap[edge.source] = [];
    if (!adjacencyMap[edge.target]) adjacencyMap[edge.target] = [];
    adjacencyMap[edge.source].push(edge);
    adjacencyMap[edge.target].push(edge);
}

function graphTraversal(startNodeId, maxHops = 2) {
    const cacheKey = `${startNodeId}:${maxHops}`;
    if (graphTraversalCache.has(cacheKey)) {
        return graphTraversalCache.get(cacheKey);
    }

    const visited = new Set();
    const results = [];
    const queue = [{ id: startNodeId, hop: 0 }];

    while (queue.length > 0) {
        const { id, hop } = queue.shift();
        if (!id || visited.has(id) || hop > maxHops) continue;
        visited.add(id);

        const node = nodeMap[id];
        if (node) results.push(node);

        const edges = adjacencyMap[id] || [];
        for (const edge of edges) {
            const nextId = edge.source === id ? edge.target : edge.source;
            if (nextId && !visited.has(nextId)) {
                queue.push({ id: nextId, hop: hop + 1 });
            }
        }
    }

    graphTraversalCache.set(cacheKey, results);
    return results;
}

// ─── Chroma Init ──────────────────────────────────────────────────────────────
async function initChroma() {
    collection = await chroma.getOrCreateCollection({
        name: 'zoo_collection',
        embeddingFunction: embedder
    });
    console.log('Connected to ChromaDB collection: zoo_collection');
}

async function getDynamicZooTimings(language = 'en') {
    try {
        const res = await collection.get({ ids: ['zootime_timings'] });
        if (res && res.metadatas && res.metadatas[0] && res.metadatas[0].full_data) {
            const timings = JSON.parse(res.metadatas[0].full_data);

            const dateIST = new Date(new Date().toLocaleString("en-US", { timeZone: "Asia/Kolkata" }));
            const todayStr = dateIST.toLocaleDateString("en-US", { weekday: 'long' });
            const today = timings.find(t => t.day === todayStr);

            const daysHi = { 'Monday': 'सोमवार', 'Tuesday': 'मंगलवार', 'Wednesday': 'बुधवार', 'Thursday': 'गुरुवार', 'Friday': 'शुक्रवार', 'Saturday': 'शनिवार', 'Sunday': 'रविवार' };

            let weekSchedule = timings.map(t => {
                if (t.day === 'Friday') {
                    return language === 'hi' ? `- ${daysHi['Friday']}: बंद` : `- Friday: Closed`;
                }
                return language === 'hi'
                    ? `- ${daysHi[t.day]}: ${t.openTime} - ${t.closeTime}`
                    : `- ${t.day}: ${t.openTime} - ${t.closeTime}`;
            }).join('\n');

            let todayScheduleHi = 'उपलब्ध नहीं';
            let todayScheduleEn = 'N/A';
            if (today) {
                if (today.day === 'Friday') {
                    todayScheduleHi = 'बंद';
                    todayScheduleEn = 'Closed';
                } else {
                    todayScheduleHi = `${today.openTime} से ${today.closeTime}`;
                    todayScheduleEn = `${today.openTime} to ${today.closeTime}`;
                }
            }

            if (language === 'hi') {
                return `**आज (${today ? daysHi[today.day] : ''}) का समय:** ${todayScheduleHi}।\n\n**पूरे सप्ताह का समय:**\n${weekSchedule}`;
            }
            return `**Today (${today ? today.day : ''}):** ${todayScheduleEn}.\n\n**Weekly Schedule:**\n${weekSchedule}`;
        }
    } catch (e) {
        console.error("[TIMINGS] Error fetching from Chroma:", e);
    }
    return null; // Fallback to LLM if data missing
}

const zooRegistry = {
    canonicalNames: [],
    lookup: {},
    metadata: {},
    sortedCanonical: [],
    eventNames: new Set(),
    rawNames: new Set(),
    rawToRender: {}
};
let venueTimings = [];

// ─── Priority Overrides ───────────────────────────────────────────────────────
const priorityOverrides = {
    'endangered': 'Endangered',
    'conservation': 'Endangered',
    'संकटग्रस्त': 'Endangered',
    'संकटग्रस्त जानवर': 'Endangered',
    'peacock': 'Indian Peafowl (Leucistic)',
    'मोर': 'Indian Peafowl (Leucistic)',
    'peacocks': 'Indian Peafowl (Leucistic)',
    'mor': 'Indian Peafowl (Leucistic)',
    'mora': 'Indian Peafowl (Leucistic)',
    'peafowl': 'Indian Peafowl (Leucistic)',
    'peafowls': 'Indian Peafowl (Leucistic)',
    'peahen': 'Indian Peafowl (Leucistic)',
    'peahens': 'Indian Peafowl (Leucistic)',
    'white peafowl': 'White Peafowl',
    'white peacock': 'White Peafowl',
    'lion tailed': 'Lion Tailed Macaque',
    'lion tail': 'Lion Tailed Macaque',
    'lion tailed monkey': 'Lion Tailed Macaque',
    'lion tailed monkeys': 'Lion Tailed Macaque',
    'lion tailed macaque': 'Lion Tailed Macaque',
    'lion tailed macaques': 'Lion Tailed Macaque',
    'lion tailed moneky': 'Lion Tailed Macaque',
    'lion macaque': 'Lion Tailed Macaque',
    'lion macaques': 'Lion Tailed Macaque',
    'lion monkey': 'Lion Tailed Macaque',
    'lion monkeys': 'Lion Tailed Macaque',
    'lion bandar': 'Lion Tailed Macaque',
    'lion bandaro': 'Lion Tailed Macaque',
    'lion bandaron': 'Lion Tailed Macaque',
    'sher bandar': 'Lion Tailed Macaque',
    'shera bandar': 'Lion Tailed Macaque',
    'lion': 'Asiatic Lion',
    'lions': 'Asiatic Lion',
    'monkey': 'macaque',
    'monkeys': 'Rhesus Macaque',
    'bandar': 'macaque',
    'bandaro': 'macaque',
    'bandaron': 'macaque',
    'बंदर': 'macaque',
    'बन्दर': 'macaque',
    'बंदरों': 'macaque',
    'tiger': 'White Tiger',
    'tigers': 'White Tiger',
    'elephant': 'Indian Elephant',
    'elephants': 'Indian Elephant',
    'sher': 'Asiatic Lion',
    'bagh': 'White Tiger',
    'hathi': 'Indian Elephant',
    'bhalu': 'Sloth Bear',
    'भालू': 'Sloth Bear',
    'rhino': 'Indian Rhinoceros',
    'rhinos': 'Indian Rhinoceros',
    'genda': 'Indian Rhinoceros',
    'गेंडा': 'Indian Rhinoceros',
    'hippo': 'Hippopotamus',
    'hippos': 'Hippopotamus',
    'dariyayi ghoda': 'Hippopotamus',
    'दरियाई घोड़ा': 'Hippopotamus',
    'croc': 'Marsh Crocodile',
    'crocs': 'Marsh Crocodile',
    'crocodile': 'Marsh Crocodile',
    'crocodiles': 'Marsh Crocodile',
    'gharial': 'Gharial',
    'gharials': 'Gharial',
    'owl': 'Great Horned Owl',
    'owls': 'Great Horned Owl',
    'fox': 'Common Fox (Indian Fox)',
    'foxes': 'Common Fox (Indian Fox)',
    'lomdi': 'Common Fox (Indian Fox)',
    'लोमड़ी': 'Common Fox (Indian Fox)',
    'leopard': 'Indian Leopard',
    'leopards': 'Indian Leopard',
    'tendua': 'Indian Leopard',
    'तेंदुआ': 'Indian Leopard',
    'suar': 'Wild Boar',
    'suwar': 'Wild Boar',
    'सूअर': 'Wild Boar',
    'सुअर': 'Wild Boar',
    'boar': 'Wild Boar',
    'boars': 'Wild Boar',
    'civet': 'Asian Palm Civet',
    'civets': 'Asian Palm Civet',
    'kasturi billi': 'Asian Palm Civet',
    'कस्तूरी बिल्ली': 'Asian Palm Civet',
    'saap': 'Common Rat Snake',
    'saanp': 'Common Rat Snake',
    'सांप': 'Common Rat Snake',
    'साँप': 'Common Rat Snake',
    'magarmach': 'Marsh Crocodile',
    'मगरमच्छ': 'Marsh Crocodile',
    'lakadbaghgha': 'Striped Hyena',
    'lakadbaggha': 'Striped Hyena',
    'lakadbagha': 'Striped Hyena',
    'लकड़बग्घा': 'Striped Hyena',
    'एशियाई शेर': 'Asiatic Lion',
    'शेर': 'Asiatic Lion',
    'सफेद बाघ': 'White Tiger',
    'बाघ': 'White Tiger',
    'हाथी': 'Indian Elephant',
    'पेन': 'Washrooms',
    'पानी': 'Drinking Water',
    'खाना': 'Food & Drinks',
    'flightless bird': 'Flightless birds',
    'flightless birds': 'Flightless birds',
    'snake': 'Common Rat Snake',
    'snakes': 'Reptile House',
    'reptile': 'Reptile House',
    'reptiles': 'Reptile House',
    'सरीसृप': 'Reptile House',
    'सरीसृपों': 'Reptile House',
    'cobra': 'Spectacled Cobra',
    'python': 'Indian Rock Python',
    'deer': 'Spotted Deer',
    'bear': 'Sloth Bear',
    'bears': 'Sloth Bear',
    'Food & Drinks': 'Food & Drinks',
    'Drinking Water': 'Drinking Water',
    'Washrooms': 'Washrooms',
    'Buggy Stops': 'Buggy Stops',
    'Emergency': 'Emergency',
    'पक्षी': 'Aquatic Birds Aviary',
    'पक्षियों': 'Aquatic Birds Aviary',
    'चिड़िया': 'Aquatic Birds Aviary',
    'चिड़ियाँ': 'Aquatic Birds Aviary'
};

// ─── Trie Index ───────────────────────────────────────────────────────────────
let trieIndex = new Map();

const TRIE_BLACKLIST = new Set([
    'bird', 'birds', 'animal', 'animals', 'reptile', 'reptiles',
    'mammal', 'mammals', 'fish', 'insect', 'insects', 'plant', 'plants',
    'cat', 'cats', 'dog', 'dogs', 'pet', 'pets',
]);

function buildTrieIndex() {
    const entries = [];
    for (const [phrase, name] of Object.entries(zooRegistry.lookup)) {
        if (TRIE_BLACKLIST.has(phrase.toLowerCase())) continue;
        entries.push([phrase.toLowerCase(), name]);
    }
    for (const name of zooRegistry.canonicalNames) {
        const lower = name.toLowerCase();
        if (TRIE_BLACKLIST.has(lower)) continue;
        if (!entries.some(e => e[0] === lower)) {
            entries.push([lower, name]);
        }
    }
    entries.sort((a, b) => b[0].length - a[0].length);
    trieIndex = new Map(entries.map(([phrase, name]) => [
        phrase,
        { name, regex: new RegExp(`\\b${phrase.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i') }
    ]));
    console.log(`[TRIE] Built index with ${trieIndex.size} entries.`);
}

function fastExtract(query) {
    const q = query.toLowerCase();
    const cleanQ = q.replace(/[?!.,;()'"]/g, '').trim();
    const queryWords = cleanQ.split(/\s+/).filter(w => w.length > 0 && !QUERY_STOP_WORDS.has(w));
    const foundMatches = [];

    for (const [phrase, { name, regex }] of trieIndex) {
        if (regex.test(q)) {
            const phraseWords = phrase.split(/\s+/).filter(w => w.length > 0);
            const extraWords = queryWords.filter(qw => !phraseWords.includes(qw));
            if (extraWords.length > 0) {
                // If there are extra meaningful words in the query that are not part of the phrase,
                // do not treat this phrase as a fast-extract hit (e.g. "black bear" query matching "bear").
                continue;
            }

            if (!foundMatches.some(m => m.phrase.includes(phrase))) {
                foundMatches.push({ phrase, name });
            }
        }
    }

    const uniqueNames = [...new Set(foundMatches.map(m => m.name))];
    if (uniqueNames.length === 1) {
        const wordCount = q.split(/\s+/).length;
        if (wordCount <= 7) return uniqueNames[0];
    }
    return null;
}

const ADJECTIVE_BLACKLIST = new Set([
    'asiatic', 'asian', 'indian', 'bengal', 'common', 'spotted', 'sloth', 'spectacled',
    'marsh', 'rock', 'rhesus', 'white', 'black', 'african', 'greater', 'lesser',
    'crested', 'palm', 'leucistic', 'stump', 'smooth', 'coated', 'collared',
    'aquatic', 'wild', 'red', 'grey', 'gray', 'brow', 'antlered',
    'world', 'national', 'international'
]);

function findRelatedAnimals(subject, queryText) {
    const seeds = new Set();

    // 1. Gather words from the query
    if (queryText) {
        const queryWords = queryText.toLowerCase().replace(/[?!.,()]/g, '').split(/\s+/);
        for (const w of queryWords) {
            if (w.length >= 3 && !QUERY_STOP_WORDS.has(w) && !ADJECTIVE_BLACKLIST.has(w)) {
                seeds.add(w);
                // Basic plural handling so "lions" matches "lion"
                if (w.endsWith('s') && w.length > 3) {
                    seeds.add(w.slice(0, -1));
                }
                // Map Hindi queries back to English seeds to match English rawNames
                for (const [engKey, hindiVal] of Object.entries(HINDI_DICT)) {
                    if (hindiVal === w || engKey === w) {
                        const engWords = engKey.split(/\s+/);
                        for (const ew of engWords) {
                            if (ew.length >= 3 && !QUERY_STOP_WORDS.has(ew)) {
                                seeds.add(ew);
                                if (ew.endsWith('s') && ew.length > 3) seeds.add(ew.slice(0, -1));
                            }
                        }
                    }
                }
            }
        }
    }

    //     // 2. Gather words from the subject (strip numbers first)
    // if (subject && subject.toLowerCase() !== 'general') {
    //     const cleanSubject = subject.toLowerCase().replace(/\s+\d+$/, '').trim();
    //     const subjectWords = cleanSubject.replace(/[?!.,()0-9]/g, '').split(/\s+/);
    //     for (const w of subjectWords) {
    //         if (w.length >= 3 && !QUERY_STOP_WORDS.has(w) && !ADJECTIVE_BLACKLIST.has(w)) {
    //             seeds.add(w);
    //         }
    //     }
    // }



    // The user requested we ONLY match related cards against the original prompt, NOT the extracted subject/keyword.
    // Therefore, we no longer extract words from the subject.

    const related = new Set();
    for (const seed of seeds) {
        for (const name of zooRegistry.rawNames) {
            const cleanName = name.replace(/\s+\d+$/, '').trim();
            if (zooRegistry.eventNames.has(cleanName)) continue; // skip events
            const nameLower = name.toLowerCase().replace(/[0-9]/g, '');
            // check if the seed is a standalone word in the name
            if (new RegExp(`\\b${seed}\\b`, 'i').test(nameLower)) {
                related.add(name);
            }
        }
    }

    return Array.from(related);
}

function loadZooRegistry() {
    const dataDir = path.join(__dirname, 'zoo-data');
    if (!fs.existsSync(dataDir)) {
        console.error('Zoo data directory missing!');
        return;
    }

    const files = fs.readdirSync(dataDir).filter(f => f.endsWith('.json'));
    const names = new Set();
    const eventNames = new Set();
    const aliasMap = new Map();
    const classifications = new Set();

    for (const file of files) {
        if (file.includes('geojson') || file.includes('floorplan') || file.includes('facts')) continue;
        try {
            const raw = fs.readFileSync(path.join(dataDir, file), 'utf8');
            const data = JSON.parse(raw);
            const items = Array.isArray(data) ? data : (data.data || [data]);

            for (const item of items) {
                const getMultilingualNames = (field) => {
                    if (!field) return [];
                    if (typeof field === 'string') return [field];
                    if (typeof field === 'object') {
                        return Object.values(field).filter(v => typeof v === 'string');
                    }
                    return [];
                };

                const rName = item.render_name?.en || item.render_name;
                const cName = item.common_name?.en || item.common_name;
                const dName = item.name?.en || item.name;
                const tName = item.title?.en || item.title;

                const primaryNameRaw = cName || rName || dName || tName;
                if (!primaryNameRaw || typeof primaryNameRaw !== 'string' || /^[0-9a-fA-F]{24}$/.test(primaryNameRaw)) continue;

                const primaryName = primaryNameRaw.replace(/\s+\d+$/, '').trim();
                if (primaryName.length <= 2) continue;

                names.add(primaryName);
                // Add raw names in all languages and map them to their render names
                const rawNamesList = [
                    primaryNameRaw,
                    item.common_name?.hi,
                    item.render_name?.hi,
                    item.name?.hi,
                    item.title?.hi
                ].filter(n => n && typeof n === 'string' && n.length > 2);

                rawNamesList.forEach(raw => {
                    zooRegistry.rawNames.add(raw);

                    const renderEn = (item.render_name && typeof item.render_name === 'object' ? item.render_name.en : item.render_name) || primaryName;
                    const renderHi = (item.render_name && typeof item.render_name === 'object' ? item.render_name.hi : null) || item.common_name?.hi || item.name?.hi || item.title?.hi || primaryName;

                    zooRegistry.rawToRender[raw] = {
                        en: renderEn,
                        hi: renderHi
                    };

                    const cleanedRaw = raw.replace(/\s+\d+$/, '').trim();
                    if (!zooRegistry.rawToRender[cleanedRaw]) {
                        zooRegistry.rawToRender[cleanedRaw] = {
                            en: renderEn,
                            hi: renderHi
                        };
                    }
                });

                if (!aliasMap.has(primaryName)) aliasMap.set(primaryName, new Set());

                const allPossibleNames = [
                    ...getMultilingualNames(item.render_name),
                    ...getMultilingualNames(item.common_name),
                    ...getMultilingualNames(item.name),
                    ...getMultilingualNames(item.title)
                ];

                allPossibleNames.forEach(n => {
                    if (n && !/^[0-9a-fA-F]{24}$/.test(n)) {
                        const clean = n.replace(/\s+\d+$/, '').trim();
                        if (clean.length > 1) aliasMap.get(primaryName).add(clean);
                    }
                });

                const classification = item.classification?.en || item.classification || '';
                if (classification && classification.length > 2) classifications.add(classification);

                let threatStatus = '';
                if (item.threat_status) {
                    if (typeof item.threat_status === 'string') {
                        threatStatus = item.threat_status;
                    } else if (item.threat_status.en) {
                        threatStatus = item.threat_status.en;
                    }
                }
                if (!threatStatus && item.conservation?.iucn_status) {
                    if (typeof item.conservation.iucn_status === 'string') {
                        threatStatus = item.conservation.iucn_status;
                    } else if (item.conservation.iucn_status.en) {
                        threatStatus = item.conservation.iucn_status.en;
                    }
                }
                threatStatus = threatStatus.trim();

                const activity = item.activity?.en || (typeof item.activity === 'string' ? item.activity : '') || '';

                if (file.includes('calendar') || file.includes('event')) {
                    eventNames.add(primaryName);
                }
                if (!zooRegistry.metadata[primaryName] || classification || threatStatus || activity) {
                    zooRegistry.metadata[primaryName] = { classification, threatStatus, activity };
                }
            }
        } catch (e) { /* skip malformed file */ }
    }

    zooRegistry.canonicalNames = Array.from(names);

    const blacklist = new Set([
        'national', 'international', 'world', 'india', 'indian',
        'park', 'zoo', 'day', 'and', 'the', 'for', 'with', 'birds', 'animals'
    ]);

    for (const cls of classifications) {
        if (cls.length > 3 && !blacklist.has(cls.toLowerCase())) {
            names.add(cls);
            zooRegistry.lookup[cls.toLowerCase()] = cls;
        }
    }

    for (const canonical of zooRegistry.canonicalNames) {
        const lower = canonical.toLowerCase();
        const meta = zooRegistry.metadata[canonical];

        let displayName = canonical;
        if (!canonical.includes(' ') && meta?.classification) {
            const firstClassWord = meta.classification.split(/[/\s,]+/)[0];
            if (firstClassWord && firstClassWord.length > 2) {
                displayName = `${canonical} ${firstClassWord}`;
            }
        }

        const aliases = aliasMap.get(canonical) || new Set([canonical]);
        for (const alias of aliases) {
            zooRegistry.lookup[alias.toLowerCase()] = displayName;
        }

        const isFacilityName = /toilet|water|washroom|food|drink|canteen|kiosk|entry|gate/i.test(canonical);
        if (isFacilityName) {
            const words = lower.split(/[/\s,.-]+/);
            for (const word of words) {
                if (word.length > 3 && !blacklist.has(word) && !zooRegistry.lookup[word]) {
                    zooRegistry.lookup[word] = displayName;
                }
            }
        }
    }

    Object.assign(zooRegistry.lookup, priorityOverrides);

    zooRegistry.sortedCanonical = [...zooRegistry.canonicalNames]
        .sort((a, b) => b.length - a.length);

    zooRegistry.eventNames = eventNames;

    console.log(`\n📚 Zoo Registry: Loaded ${zooRegistry.canonicalNames.length} species dynamically (${zooRegistry.eventNames.size} events).`);
}

function levenshtein(a, b) {
    if (a === b) return 0;
    if (a.length === 0) return b.length;
    if (b.length === 0) return a.length;

    if (a.length > b.length) {
        [a, b] = [b, a];
    }

    let v0 = new Int16Array(a.length + 1);
    let v1 = new Int16Array(a.length + 1);

    for (let i = 0; i <= a.length; i++) v0[i] = i;

    for (let i = 0; i < b.length; i++) {
        v1[0] = i + 1;
        for (let j = 0; j < a.length; j++) {
            const cost = a[j] === b[i] ? 0 : 1;
            v1[j + 1] = Math.min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost);
        }
        for (let j = 0; j <= a.length; j++) v0[j] = v1[j];
    }

    return v1[a.length];
}
const QUERY_STOP_WORDS = new Set([
    'find', 'where', 'show', 'tell', 'what', 'which', 'how', 'when', 'does',
    'can', 'could', 'would', 'should', 'will', 'have', 'has', 'had', 'the',
    'and', 'for', 'are', 'from', 'that', 'with', 'this', 'they', 'been',
    'their', 'about', 'some', 'more', 'there', 'than', 'into', 'these',
    'like', 'look', 'give', 'list', 'many', 'know', 'want', 'need', 'also',
    // NOTE: 'bird', 'birds', 'animal', 'animals' intentionally removed — they are valid
    // subject seeds for fuzzy matching (e.g. "brids" typo must reach Levenshtein check).
    // They are caught by GENERIC_SUBJECT_WORDS at the routing layer instead.
    'species', 'places', 'place',
    'shera', // bot's own name — must never be treated as an animal subject
    'मुझे', 'दिखाओ', 'कहाँ', 'कहा', 'किधर', 'है', 'हैं', 'था', 'थे', 'का',
    'की', 'के', 'को', 'में', 'से', 'पर', 'और', 'या', 'कैसे', 'कब', 'क्या',
    'कौन', 'वहाँ', 'वहां', 'यहाँ', 'यहां', 'पास', 'नज़दीक', 'आसपास', 'दिखाएं',
    'दिखाइए', 'बताओ', 'बताएं', 'बताइए', 'खोजो', 'खोजें', 'मिलेंगे', 'मिलेगा',
    'मिलेगी', 'मिलते', 'मिलती', 'पशु', 'जानवर', 'जानवरों', 'जीव',
    'कल', 'आज', 'kal', 'aaj', 'kab', 'jab', 'tab', 'ab', 'abhi',
    'zoo', 'park', 'national', 'zoological', 'chidiya', 'ghar', 'chidiyaghar',
    'the', 'a', 'an', 'is', 'am', 'are', 'of', 'to', 'in', 'on', 'at', 'it', 'its', 'them', 'those', 'he', 'she', 'his', 'hers', 'my', 'mine', 'our', 'ours', 'your', 'yours',
    'mere', 'paas', 'pas', 'konse', 'kaunse', 'janwar', 'jannwar', 'kahan',
    'kaha', 'hain', 'hai', 'mujhe', 'dikhao', 'kise', 'kiske', 'idhar', 'udhar',
    'tha', 'the', 'ka', 'ki', 'ke', 'ko', 'mein', 'se', 'par', 'aur', 'ya',
    'kaise', 'kab', 'kya', 'kaun', 'waha', 'wahan', 'yaha', 'yahan', 'nazdeek',
    'aaspaas', 'batao', 'bataye', 'khojo', 'milenge', 'milega', 'pashu', 'jeev',
    'kon', 'konsa', 'konsi', 'kisi', 'koi', 'unko', 'isko', 'usko', 'iske', 'uske'
]);

// Generic subject words: valid zoo subjects but too broad to resolve to a single animal.
// Used only at the routing layer to decide "deep search" vs "general".
// const GENERIC_SUBJECT_WORDS = new Set(['bird', 'birds', 'animal', 'animals']);

const GENERIC_CATEGORY_WORDS = new Set([
    'animal', 'animals', 'bird', 'birds', 'insect', 'insects', 'fish', 'fishes',
    'reptile', 'reptiles', 'mammal', 'mammals', 'pashu', 'janwar', 'pakshi', 'panchi',
    'जीव', 'जानवर', 'पक्षी', 'tour', 'tours', 'safari', 'safar', 'trip'
]);

const EVENT_INDICATOR_REGEX = /\b(day|week|event|festival|mahotsav|celebrat|program|divas|saptah|karyakram|दिवस|महोत्सव|सप्ताह|कार्यक्रम)\b/i;

function normalizeToRegistryOrSelf(rawSubject) {
    const cleanSubject = rawSubject.replace(/[?!.,;()'"]/g, '').trim();
    const words = cleanSubject.split(/\s+/);
    for (let len = words.length; len >= 1; len--) {
        const candidate = words.slice(0, len).join(' ');
        const lower = candidate.toLowerCase();
        if (QUERY_STOP_WORDS.has(lower)) continue;
        if (zooRegistry.lookup[lower]) {
            const resolved = zooRegistry.lookup[lower];
            if (zooRegistry.eventNames.has(resolved) && !EVENT_INDICATOR_REGEX.test(rawSubject)) {
                // skip event match without event indicator
            } else {
                return resolved;
            }
        }
        if (zooRegistry.canonicalNames.some(n => n.toLowerCase() === lower)) {
            const hit = zooRegistry.canonicalNames.find(n => n.toLowerCase() === lower);
            if (zooRegistry.eventNames.has(hit) && !EVENT_INDICATOR_REGEX.test(rawSubject)) {
                // skip
            } else {
                return hit;
            }
        }
    }

    for (const queryWord of words) {
        const qwLower = queryWord.toLowerCase();
        if (qwLower.length < 3) continue;
        if (QUERY_STOP_WORDS.has(qwLower)) continue;
        if (zooRegistry.lookup[qwLower]) {
            const resolved = zooRegistry.lookup[qwLower];
            if (zooRegistry.eventNames.has(resolved) && !EVENT_INDICATOR_REGEX.test(rawSubject)) {
                // skip
            } else {
                return resolved;
            }
        }
        const exactWordHit = zooRegistry.canonicalNames.find(n => {
            if (zooRegistry.eventNames.has(n) && !EVENT_INDICATOR_REGEX.test(rawSubject)) return false;
            const nl = n.toLowerCase();
            return nl.split(/[^a-z0-9]+/).includes(qwLower);
        });
        if (exactWordHit) return exactWordHit;
    }

    for (const queryWord of words) {
        const qwLower = queryWord.toLowerCase();
        if (qwLower.length < 3) continue;
        if (QUERY_STOP_WORDS.has(qwLower)) continue;
        const fuzzyWordHit = zooRegistry.canonicalNames.find(n => {
            if (zooRegistry.eventNames.has(n) && !EVENT_INDICATOR_REGEX.test(rawSubject)) return false;
            const nl = n.toLowerCase();
            const canonicalWords = nl.split(/[^a-z0-9]+/);
            for (const cw of canonicalWords) {
                if (cw.length < 4) continue; // Allow 4-char words like 'bird', 'bear', 'deer'
                const maxDist = cw.length >= 7 ? 2 : 1;
                const lengthDiff = Math.abs(qwLower.length - cw.length);
                if (lengthDiff > maxDist) continue;
                if (levenshtein(qwLower, cw) <= maxDist) return true;
            }
            return false;
        });
        if (fuzzyWordHit) return fuzzyWordHit;
    }

    const meaningfulWords = words.filter(w => !QUERY_STOP_WORDS.has(w.toLowerCase()) && w.length > 1);
    return meaningfulWords.length > 0 ? meaningfulWords[0] : rawSubject;
}

function optimizeContext(docs, maxLines = 5) {
    return [...new Set(docs)].slice(0, maxLines).join('\n');
}


// English Traits
const TRAIT_WORDS = /\b(big|bigger|large|larger|size|small|smaller|heavy|weight|weigh|tall|height|long|length|old|age|lifespan|live|lives|fast|speed|fact|facts|eat|eats|diet|food|sleep|active|species|group|color|housed|born|likes|dislikes|pregnancy|gestation|leg|legs|eye|eyes|tooth|teeth|tail|tails|wing|wings|skin|fur|horn|horns|neck|claw|claws|run|swim|fly|sound|voice|feature|features|look|looks|physical|anatomy|incubation|clutch|cubs|cub|baby|babies)\b/i;

// English Question Words
const QUESTION_WORDS = /\b(how|what|where|why|when|who|tell|give|share|does|do|is|are)\b/i;

// Hindi/Hinglish Traits
const HINDI_TRAIT_WORDS = /भारी|bhari|bhaari|मोटा|mota|motai|बड़ा|bada|bade|chota|chote|size|वजन|प्रजातियाँ|समूह|prajatiyan|samuh|wazan|vajan|लंबा|lamba|lambai|ऊंचाई|unchai|old|age|उम्र|umr|lifespan|life|jeevan|fast|tez|speed|raf-tar|fact|facts|khana|diet|sleep|sona|active|rang|color|housed|born|paida|pregnancy|gestation|पैर|pair|टांग|tang|टांगे|tange|आंख|aankh|आंखें|aankhen|दांत|daant|पूंछ|poonch|पंख|pankh|त्वचा|twacha|fur|सींग|seeng|गर्दन|gardan|नाखून|nakhoon|दौड़|daud|तैर|tair|उड़|ud|आवाज|aawaz|रूप|roop|बच्चा|bacha|बच्चे|bache/i;

// Hindi/Hinglish Question Words
const HINDI_QUESTION_WORDS = /क्या|kya|kaun|who|kahan|where|kyon|why|kab|when|kaise|how|kitna|kitni|kitne|does|do|is|are/i;

const FUZZY_TRAITS = [
    'weight', 'height', 'length', 'lifespan', 'speed', 'diet',
    'color', 'active', 'pregnancy', 'gestation', 'like', 'likes'
];

function shouldSuppressCard(query, finalSubject) {
    if (!query || typeof query !== 'string') return false;
    let q = query.toLowerCase();

    // Prevent trait suppression if the trait word is literally part of the animal's name (e.g. "Lion Tailed Macaque" shouldn't trigger "tail")
    if (finalSubject && finalSubject !== 'general') {
        const subjectWords = finalSubject.toLowerCase().replace(/[?!.,;()]/g, '').split(/\s+/);
        for (const sw of subjectWords) {
            const swBase = sw.replace(/ed$/, '').replace(/s$/, ''); // Handle "tailed" -> "tail"
            q = q.replace(new RegExp(`\\b${sw}\\b`, 'gi'), '');
            if (swBase.length > 2) {
                q = q.replace(new RegExp(`\\b${swBase}\\b`, 'gi'), '');
            }
        }
    }

    // 1. Check exact regexes first (Fastest)
    let hasTrait = TRAIT_WORDS.test(q) || HINDI_TRAIT_WORDS.test(q);

    // 2. If no exact match, run a fuzzy scan for typos (e.g., "likr" -> "like")
    if (!hasTrait) {
        const queryWords = q.replace(/[?!.,;()]/g, '').split(/\s+/);
        for (const qw of queryWords) {
            if (qw.length < 4) continue; // Skip tiny words to prevent aggressive false positives

            for (const trait of FUZZY_TRAITS) {
                // Only compare words of similar length
                if (Math.abs(qw.length - trait.length) > 1) continue;

                // Allow 1 character typo
                if (levenshtein(qw, trait) <= 1) {
                    console.log(`[UI CONTROL] Fuzzy trait matched: "${qw}" ≈ "${trait}"`);
                    hasTrait = true;
                    break;
                }
            }
            if (hasTrait) break;
        }
    }

    const isQuestion = QUESTION_WORDS.test(q) || HINDI_QUESTION_WORDS.test(q);
    const wordCount = q.split(/\s+/).length;

    // 3. Relaxed Gate: Is it phrased as a question, OR is it at least 2 words?
    // "likes of lions" = 3 words. "lion weight" = 2 words. 
    if (hasTrait && (isQuestion || wordCount >= 2)) {
        return true;
    }

    return false;
}

function isCasualChatQuery(query) {
    if (!query || typeof query !== 'string') return false;
    const q = query.toLowerCase().replace(/[?!.,]/g, '').trim();

    const exactMatches = new Set([
        'hello', 'hi', 'hey', 'namaste', 'tata', 'bye', 'goodbye',
        'yes', 'no', 'sure', 'ok', 'okay', 'fine', 'cool', 'nice',
        'great', 'wow', 'awesome', 'really', 'seriously', 'serious',
        'no way', 'haha', 'lol', 'lmao', 'oh', 'please', 'thank you',
        'thanks', 'thanks a lot', 'who are you', 'what is your name',
        'how are you', 'good morning', 'good afternoon', 'good evening',
        'are you serious', 'are you sure', 'is that true', 'are you sure about that',
        'are you serious right now', 'tell me more', 'tell me', 'show me', 'help me',
        'नमस्ते', 'नमस्कार', 'हैलो', 'हाय', 'हे', 'राम राम', 'सलाम',
        'शुभ प्रभात', 'शुभ संध्या', 'शुभ दोपहर', 'कैसे हो', 'कैसे हैं',
        'कैसे हो आप', 'कैसे हैं आप', 'क्या हाल है', 'क्या चल रहा है',
        'धन्यवाद', 'शुक्रिया', 'बाय', 'अलविदा', 'सच में', 'गंभीर हो', 'क्या बात है'
    ]);

    if (exactMatches.has(q)) return true;

    const patterns = [
        /^hello\b/i,
        /^hi\b/i,
        /^hey\b/i,
        /^good (morning|afternoon|evening)\b/i,
        /^thank(s|\s+you)\b/i,
        /^how are you\b/i,
        /^who (are|is)\b/i,
        /^are you (serious|sure|real|ai|bot|human)\b/i,
        /^is that (true|correct|right|real)\b/i,
        /^(haha|lol|lmao|hehe)\b/i,
        /^नमस्ते/i,
        /^नमस्कार/i,
        /^हैलो/i,
        /^हाय/i,
        /^हे/i,
        /^राम राम/i,
        /^सलाम/i,
        /^शुभ\s+(प्रभात|संध्या|दोपहर)/i,
        /^कैसे\s+(हो|हैं)/i,
        /^क्या\s+(हाल|चल)/i
    ];

    for (const pat of patterns) {
        if (pat.test(q)) return true;
    }

    return false;
}

function isTraitOrCategoryQuery(query) {
    const q = query.toLowerCase();

    // Simple greetings, booking/timings help should not trigger trait searches
    if (/^(hi|hello|hey|hola|namaste|good morning|good afternoon|good evening|who are you|what is your name|how are you|help|info|information|rules|ticket|tickets|price|pricing|timing|timings|hours|gate|map|direction|directions|where is|where are|can i|may i)/i.test(q)) {
        return false;
    }

    // Proximity/nearby queries should not be treated as trait searches
    const isLocationQuery = q.includes('nearby')
        || q.includes('close to me')
        || q.includes('where am i')
        || q.includes('आसपास')
        || q.includes('नज़दीक')
        || q.includes('पास')
        || /\b(paas|pas|nazdeek|aaspaas)\b/.test(q);
    if (isLocationQuery) {
        return false;
    }

    const keywords = /\b(animal|animals|species|bird|birds|reptile|reptiles|mammal|mammals|fish|fishes|amphibian|amphibians|insect|insects|native|found|from|country|region|place|live|lives|living|eat|eats|eating|diet|habitat|threatened|endangered|extinct|carnivore|carnivorous|herbivore|herbivorous|omnivore|omnivorous|predator|predators|prey|wild|forest|desert|mountain|river|aquatic|water|cold|hot|warm|weather|climate|nepal|nepalese|nepali|india|indian|africa|african|china|chinese|asia|asiatic|siberian|american|australian|himalayan|savanna|sahara|polar|arctic|antarctic|tundra|scrub|jungle|wetland|swamp|marsh|lake|pond|ocean|sea|marine|coast|coastal|nocturnal|diurnal|flying|swim|swimming|run|running|climb|climbing|horned|horn|horns|antler|antlers|janwar|pashu|pakshi|panchi|shakahari|mansahari|sarvahari|shikari|chidiya|chirya|chidiyan|chiryan|चिड़िया|चिड़ियां)\b/i;

    return keywords.test(q);
}

const facilitySynonyms = {
    'Food & Drinks': ['food', 'eat', 'hungry', 'snacks', 'restaurant', 'cafe', 'cafeteria', 'snack', 'khana', 'khaana', 'खाना', 'खानपान', 'भूख', 'bhook', 'bhuk', 'कैंटीन', 'canteen', 'kanteen'],
    'Drinking Water': ['water', 'drink', 'drinking water', 'thirsty', 'thristy', 'thirsti', 'thirsy', 'thurst', 'fountain', 'pani', 'paani', 'पानी', 'प्यास', 'pyaas', 'pyasa', 'pyaasa', 'pyase', 'pyaase', 'water bottle', 'water filter'],
    'Washrooms': ['washroom', 'wash room', 'toilet', 'toliet', 'tolet', 'toielt', 'washrum', 'washrm',
        'restroom', 'bathroom', 'shauchalay', 'shochalay', 'शौचालय', 'टॉयलेट', 'pee', 'poo', 'mutralay'],
    'Buggy Stops': ['buggy', 'shuttle', 'ride', 'cart', 'transport', 'बग्गी', 'gadi', 'gaadi', 'shuttle car', 'rickshaw', 'rikshaw'],
    'First Aid': ['first aid', 'firstaid', 'medical', 'medicine', 'doctor', 'clinic', 'hospital', 'hurt', 'pain', 'injury', 'injured', 'wound', 'wounded', 'accident', 'emergency', 'दवाई', 'अस्पताल', 'dawai', 'dawae', 'chot', 'injur', 'चोट', 'इलाज', 'इमर्जेंसी', 'दर्द'],
    'Counters': ['counter', 'ticket', 'info', 'information', 'help', 'टिकट', 'काउंटर', 'booking', 'paise'],
    'Timings & Hours': ['timing', 'timings', 'hours', 'schedule', 'khula', 'khulne', 'samay', 'baje', 'open today', 'closing time', 'opening time', 'kab tak', 'kitne baje', 'time', 'band', 'bandh', 'closed', 'chhutti', 'chutti', 'छुट्टी', 'बंद', 'khulta', 'khulega', 'khulegi', 'kholte', 'kholenge'],
    'Exit Gate': ['exit', 'exit gate', 'way out', 'leave the zoo', 'going out', 'निकास', 'निकास द्वार', 'बाहर', 'बाहर निकलें', 'बाहर जाएं', 'nikas', 'bahar'],
    'Main Entrance': ['entrance', 'entry', 'enter', 'main entrance', 'main gate', 'front gate', 'प्रवेश', 'प्रवेश द्वार', 'दरवाज़ा', 'द्वार', 'मुख्य द्वार', 'pravesh', 'entry gate'],
    'Feeding Animals': ['feed', 'feeding', 'khalana', 'khilana', 'khelana', 'khilao', 'khilau', 'khila', 'khilayen', 'khilayein', 'khilate']
};

const FACILITY_FUZZY_BLACKLIST = new Set([
    'would', 'could', 'should', 'where', 'there', 'their', 'when', 'what', 'which', 'about',
    'book', 'show', 'find', 'free', 'have', 'some', 'more', 'here', 'take',
    'tour', 'gate', 'exit', 'main', 'help', 'info', 'need', 'want', 'good', 'ride',
    'tail', 'tailed', 'tails', 'trail', 'trails', 'train', 'tiger', 'tiles',
    'lion', 'loin', 'line', 'lime', 'life', 'like', 'live', 'lima',
    'bear', 'deer', 'bird', 'fish', 'frog', 'tree', 'wild', 'wing', 'wing',
    'walk', 'path', 'road', 'area', 'zone', 'spot', 'site', 'shed',
    'cage', 'claw', 'coat', 'dark', 'dive', 'dusk', 'dust', 'fast',
    'fern', 'foal', 'ford', 'foul', 'fowl', 'fume', 'gait',
    'gale', 'game', 'gang', 'gaze', 'gore', 'gust', 'hare', 'herd',
    'hide', 'hill', 'hive', 'hole', 'hoof', 'horn', 'hunt', 'iris',
    'lake', 'lamb', 'land', 'lark', 'leaf', 'leap', 'legs', 'lore',
    'male', 'mane', 'mare', 'mark', 'mate', 'mice', 'mile', 'milk',
    'mole', 'molt', 'moss', 'moth', 'mule', 'nape', 'nest', 'newt',
    'paca', 'pace', 'pack', 'pale', 'palm', 'pelt', 'pest', 'pond',
    'pool', 'prey', 'puma', 'race', 'rain', 'rake', 'ramp', 'rank',
    'rare', 'rash', 'rats', 'raven', 'reed', 'reef', 'rest', 'rift',
    'roan', 'roar', 'rock', 'rook', 'root', 'rope', 'rose', 'rove',
    'rump', 'rush', 'rust', 'safe', 'sage', 'sand', 'scar', 'seal',
    'seed', 'shin', 'skin', 'slug', 'soil', 'sole', 'soot', 'sort',
    'sour', 'span', 'spar', 'spin', 'spit', 'star', 'stem', 'stir',
    'stag', 'swan', 'swim', 'tame', 'tank', 'teal', 'term', 'tern',
    'tick', 'toad', 'toes', 'told', 'toll', 'tome', 'tone', 'tops',
    'torn', 'tort', 'toss', 'town', 'trod', 'trot', 'true', 'tuck',
    'tuft', 'tuna', 'turf', 'tusk', 'vale', 'vane', 'veld', 'vole',
    'volt', 'wade', 'wail', 'wake', 'wale', 'ward', 'ware', 'warp',
    'wart', 'wary', 'wave', 'weak', 'weal', 'wean', 'weft', 'weld',
    'well', 'welt', 'went', 'were', 'wham', 'whin', 'whip', 'wren',
    // Hinglish additions
    'mere', 'paas', 'pas', 'konse', 'kaunse', 'janwar', 'jannwar', 'kahan',
    'kaha', 'hain', 'hai', 'mujhe', 'dikhao', 'kise', 'kiske', 'idhar', 'udhar',
    'tha', 'the', 'ka', 'ki', 'ke', 'ko', 'mein', 'se', 'par', 'aur', 'ya',
    'kaise', 'kab', 'kya', 'kaun', 'waha', 'wahan', 'yaha', 'yahan', 'nazdeek',
    'aaspaas', 'batao', 'bataye', 'khojo', 'milenge', 'milega', 'pashu', 'jeev',
    'kon', 'konsa', 'konsi', 'kisi', 'koi'
]);

function detectFacility(text) {
    const facilities = detectFacilities(text);
    return facilities.length > 0 ? facilities[0] : null;
}

function detectFacilitiesExact(text) {
    const t = text.toLowerCase();
    const matched = new Set();
    const words = t.split(/[^a-z0-9\u0900-\u097F]+/);

    for (const [facility, syns] of Object.entries(facilitySynonyms)) {
        for (const s of syns) {
            let hit = false;
            if (/^[a-z\s]+$/i.test(s)) {
                hit = new RegExp(`\\b${s}\\b`, 'i').test(t);
            } else if (s.includes(' ')) {
                hit = t.includes(s);
            } else {
                hit = words.includes(s);
            }
            if (hit) { matched.add(facility); break; }
        }
    }
    return Array.from(matched);
}

function detectFacilitiesFuzzy(text, matchedExact = []) {
    const t = text.toLowerCase();
    const matched = new Set(matchedExact);

    const words = t.split(/[^a-zA-Z]+/);
    for (const word of words) {
        if (word.length < 4) continue;
        if (FACILITY_FUZZY_BLACKLIST.has(word)) continue;
        for (const [facility, syns] of Object.entries(facilitySynonyms)) {
            if (matched.has(facility)) continue;
            for (const s of syns) {
                if (!/^[a-zA-Z]+$/.test(s) || s.length < 4) continue;
                if (s[0] !== word[0]) continue;
                const lengthDiff = Math.abs(word.length - s.length);
                if (lengthDiff > 1) continue;
                const maxAllowedDist = s.length >= 7 ? 2 : 1;
                const dist = levenshtein(word, s);
                if (dist <= maxAllowedDist) {
                    console.log(`[FUZZY FACILITY] Matched query word "${word}" to synonym "${s}" (dist: ${dist}) for "${facility}"`);
                    matched.add(facility);
                    break;
                }
            }
        }
    }
    return Array.from(matched);
}

function detectFacilities(text) {
    const exact = detectFacilitiesExact(text);
    return detectFacilitiesFuzzy(text, exact);
}

function finalizeSubject(subject, qLower, extractedSubject = null) {
    const BIO_CATEGORY_WORDS = [
        'reptile', 'reptiles', 'mammal', 'mammals', 'amphibian', 'amphibians',
        'bird', 'birds', 'insect', 'insects', 'species', 'animal', 'animals',
        'predator', 'herbivore', 'carnivore', 'omnivore', 'vertebrate',
        'invertebrate', 'classification', 'taxonomy', 'habitat', 'ecosystem'
    ];
    const hasBioContext = BIO_CATEGORY_WORDS.some(w => qLower.includes(w));
    const onlyAmbiguousFacilityWords = hasBioContext &&
        !qLower.includes('where') && !qLower.includes('find') &&
        !qLower.includes('need') && !qLower.includes('want') &&
        !qLower.includes('thirsty') && !qLower.includes('hungry') &&
        !qLower.includes('feed') && !qLower.includes('khila');

    const matchedFacilities = onlyAmbiguousFacilityWords ? [] : detectFacilities(qLower);
    const matchedFacility = matchedFacilities.length > 0 ? matchedFacilities.join(', ') : null;
    if (matchedFacility) subject = matchedFacility;
    return { subject, extractedSubject: extractedSubject || subject, matchedFacility };
}
async function llmExtractSubject(query) {
    const qLower = query.toLowerCase().trim();

    // 1. FIRST PASS: Let the LLM process riddles and descriptions
    try {
        const prompt = `You are a subject extractor. Given a user query about a zoo, output ONLY the animal or facility name mentioned. If there is none, output: general\n\nQuery: "${query}"\nSubject:`;
        const resp = await ollama.chat({
            model: EXTRACTION_MODEL,
            messages: [{ role: 'user', content: prompt }],
            keep_alive: '1h',
            options: { num_predict: 8, temperature: 0.0, top_k: 5, num_thread: 4 }
        });

        let ext = (resp.message?.content || '').trim().toLowerCase();
        ext = ext.replace(/[^a-z0-9\s]/g, '').trim();

        // Discard multi-sentence responses (model panicked and over-generated)
        ext = ext.split(/[.!?\n]/)[0].trim();

        if (ext && ext !== 'general') {
            let candidate = null;
            if (zooRegistry.lookup[ext]) {
                candidate = zooRegistry.lookup[ext];
            } else {
                const exactHit = zooRegistry.canonicalNames.find(n => n.toLowerCase() === ext);
                if (exactHit) {
                    candidate = exactHit;
                } else {
                    const fuzzyHit = zooRegistry.canonicalNames.find(n => {
                        const cws = n.toLowerCase().split(/[^a-z0-9]+/);
                        return cws.some(cw => {
                            if (cw.length < 4 || Math.abs(ext.length - cw.length) > 1) return false;
                            return levenshtein(ext, cw) <= 1;
                        });
                    });
                    if (fuzzyHit) {
                        console.log(`[EXTRACTOR-LLM] Fuzzy corrected LLM output "${ext}" to "${fuzzyHit}"`);
                        candidate = fuzzyHit;
                    }
                }
            }

            if (candidate) {
                const isCandidateEvent = zooRegistry.eventNames.has(candidate);
                const hasEventKeywords = EVENT_INDICATOR_REGEX.test(query);
                if (isCandidateEvent && !hasEventKeywords) {
                    console.log(`[EXTRACTOR-LLM] Rejected event candidate "${candidate}" for query "${query}" (no event keywords)`);
                } else {
                    return candidate;
                }
            }

            // FIX: If the LLM extracted a subject, but it's not in the zoo registry, 
            // return it anyway! This ensures the router knows the user asked about 
            // an absent animal, forcing it into the "Not Found" gatekeeper.
            return ext;
        }
    } catch (e) {
        console.warn(`[EXTRACTOR] LLM extraction failed: ${e.message}`);
    }

    // 2. FALLBACK PASS: Fuzzy Levenshtein scan for exact typos
    // This only runs if the LLM couldn't figure it out.
    const fallbackWords = qLower.replace(/[?!.,;()'"]/g, '').split(/\s+/);

    // STRICT GUARD: Prevent descriptive adjectives/body parts from matching animal names
    const DESCRIPTIVE_BLACKLIST = new Set([
        'ears', 'nose', 'trunk', 'tail', 'legs', 'eyes', 'head', 'neck',
        'color', 'colour', 'skin', 'fur', 'big', 'small', 'tall', 'heavy'
    ]);

    for (const pw of fallbackWords) {
        if (pw.length < 4) continue;
        if (QUERY_STOP_WORDS.has(pw)) continue;
        if (DESCRIPTIVE_BLACKLIST.has(pw)) continue;

        // Direct lookup hit (exact typo in lookup keys)
        if (zooRegistry.lookup[pw]) {
            const resolved = zooRegistry.lookup[pw];
            if (zooRegistry.eventNames.has(resolved) && GENERIC_CATEGORY_WORDS.has(pw) && !EVENT_INDICATOR_REGEX.test(query)) {
                continue;
            }
            console.log(`[EXTRACTOR-FUZZY-FALLBACK] Exact lookup hit: "${pw}" → "${resolved}"`);
            return resolved;
        }

        // Fuzzy scan lookup keys
        const lookupKeys = Object.keys(zooRegistry.lookup);
        for (const key of lookupKeys) {
            if (key.length < 4) continue;
            const maxDist = key.length >= 7 ? 2 : 1;
            if (Math.abs(pw.length - key.length) > maxDist) continue;

            if (levenshtein(pw, key) <= maxDist) {
                const resolved = zooRegistry.lookup[key];
                if (zooRegistry.eventNames.has(resolved) && !EVENT_INDICATOR_REGEX.test(query)) {
                    continue;
                }
                console.log(`[EXTRACTOR-FUZZY-FALLBACK] Fuzzy key match: "${pw}" ≈ "${key}" → "${resolved}"`);
                return resolved;
            }
        }

        // Fuzzy scan canonical names word-by-word
        for (const canonical of zooRegistry.canonicalNames) {
            if (zooRegistry.eventNames.has(canonical) && !EVENT_INDICATOR_REGEX.test(query)) continue;
            const cWords = canonical.toLowerCase().split(/[^a-z0-9]+/);

            for (const cw of cWords) {
                if (cw.length < 4) continue;
                const maxDist = cw.length >= 7 ? 2 : 1;
                if (Math.abs(pw.length - cw.length) > maxDist) continue;

                if (levenshtein(pw, cw) <= maxDist) {
                    console.log(`[EXTRACTOR-FUZZY-FALLBACK] Fuzzy canonical match: "${pw}" ≈ "${cw}" in "${canonical}"`);
                    return canonical;
                }
            }
        }
    }

    return null;
}
async function extractSubject(query) {
    const qLower = query.toLowerCase().trim();

    // 1. Quick greeting/short query bypass
    if (['hello', 'hi', 'hey', 'hello shera', 'hi shera', 'namaste', 'namaskar', 'नमस्ते', 'नमस्कार', 'हैलो', 'हाय', 'हे', 'राम राम', 'सलाम'].includes(qLower) || qLower.length < 3) {
        return finalizeSubject('general', qLower);
    }

    // 2. Explicit Facility Check (Ensures facility intents are never hijacked by animal names)
    const facilityHits = detectFacilitiesExact(qLower);
    if (facilityHits.length > 0) {
        // Don't short-circuit: also check if the query mentions an animal.
        // Run fast-path scans (trie + holistic) but skip LLM to keep it zero-cost.
        const trieHit = fastExtract(qLower);
        if (trieHit) {
            // Both a facility AND an animal found — surface both.
            const result = finalizeSubject(trieHit, qLower);
            result.matchedFacility = facilityHits.join(', ');
            return result;
        }
        // Run holistic scorer (pure JS, no LLM cost)
        const cleanQL = qLower.replace(/[?!.,;()'"]/g, '').trim();
        const holisticWords = cleanQL.split(/\s+/).filter(w => w.length > 0);
        let bestEntity = null, highestScore = 0;
        for (const canonical of zooRegistry.canonicalNames) {
            const cLower = canonical.toLowerCase();
            if (zooRegistry.eventNames.has(canonical)) continue;
            const cWords = cLower.split(/[^a-z0-9]+/).filter(w => w.length > 2 && !QUERY_STOP_WORDS.has(w));
            if (cWords.length === 0) continue;
            let matchedTokens = 0;
            for (const cw of cWords) {
                let best = 0;
                for (const qw of holisticWords) {
                    if (QUERY_STOP_WORDS.has(qw)) continue;
                    if (qw === cw) best = 1.0;
                    else if (qw.length >= 4 && cw.length >= 4 && (qw.includes(cw) || cw.includes(qw))) best = Math.max(best, 0.7);
                    else if (qw.length >= 5 && cw.length >= 5) {
                        const d = levenshtein(qw, cw);
                        if (d <= (cw.length >= 6 ? 2 : 1)) best = Math.max(best, 1.0 - d * 0.2);
                    }
                }
                if (best > 0) matchedTokens += best;
            }
            const score = matchedTokens * 10 + (matchedTokens / cWords.length) * 5;
            if (score > highestScore) { highestScore = score; bestEntity = canonical; }
        }
        if (highestScore >= 13 && bestEntity) {
            let animalSubject = bestEntity;
            if (zooRegistry.lookup[animalSubject.toLowerCase()]) animalSubject = zooRegistry.lookup[animalSubject.toLowerCase()];
            console.log(`[FACILITY+ANIMAL] Detected animal "${animalSubject}" alongside facility "${facilityHits.join(', ')}"`);
            const result = finalizeSubject(animalSubject, qLower);
            result.matchedFacility = facilityHits.join(', ');
            return result;
        }
        // No animal found — pure facility query, return as before
        return finalizeSubject(facilityHits.join(', '), qLower);
    }

    // 3. Precise Day/Event Pattern Recognition
    const dayMatch = query.match(/\b(national|world|international|global)\b[\w\s]+\bday\b/i);
    if (dayMatch) {
        const eventName = dayMatch[0].trim().split(/\s+/)
            .map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
            .join(' ');
        return finalizeSubject(eventName, qLower);
    }


    // If the Trie index finds a perfect match, skip all the heavy regex/looping below!
    const trieMatch = fastExtract(qLower);
    if (trieMatch) {
        console.log(`[FAST-EXTRACT] Trie hit: "${trieMatch}"`);
        return finalizeSubject(trieMatch, qLower);
    }

    // 4. Holistic Token-Scoring Matcher (Global fix for typos and multi-entity conflicts)
    const cleanQLower = qLower.replace(/[?!.,;()'"]/g, '').trim();
    const words = cleanQLower.split(/\s+/).filter(w => w.length > 0);
    let bestEntity = null;
    let highestScore = 0;

    for (const canonical of zooRegistry.canonicalNames) {
        const cLower = canonical.toLowerCase();
        // Skip events when scanning for animals/objects unless explicitly tracking an event
        if (zooRegistry.eventNames.has(canonical) && !/\bday\b/.test(qLower)) continue;

        // Extract meaningful canonical keywords
        const cWords = cLower.split(/[^a-z0-9]+/).filter(w => w.length > 2 && !QUERY_STOP_WORDS.has(w));
        if (cWords.length === 0) continue;

        let matchedTokens = 0;
        for (const cw of cWords) {
            let bestWordScore = 0;
            for (const qw of words) {
                if (QUERY_STOP_WORDS.has(qw)) continue;

                if (qw === cw) {
                    bestWordScore = 1.0; // Exact token match
                } else if (qw.length >= 4 && cw.length >= 4 && (qw.includes(cw) || cw.includes(qw))) {
                    // SAFEGUARD ADDED HERE: Requires both words to be at least 4 characters long
                    bestWordScore = 0.7; // Substring inclusion
                } else if (qw.length >= 4 && cw.length >= 4) {
                    const dist = levenshtein(qw, cw);
                    const maxAllowed = cw.length >= 6 ? 2 : 1;
                    if (dist <= maxAllowed) {
                        bestWordScore = 1.0 - (dist * 0.2); // Typo-tolerant distance scoring
                    }
                }
            }
            if (bestWordScore > 0) matchedTokens += bestWordScore;
        }

        // Base score calculations using compound density metrics
        const overlapRatio = matchedTokens / cWords.length;
        let score = matchedTokens * 10 + overlapRatio * 5;

        // Add a small phrase containment bonus to favor direct inclusions
        if (cLower.includes(qLower) || qLower.includes(cLower)) score += 3;

        if (score > highestScore) {
            highestScore = score;
            bestEntity = canonical;
        }
    }

    // Only commit to a holistic match if confidence score is solid
    if (highestScore >= 13) {
        let subject = bestEntity;
        const sLower = subject.toLowerCase();
        if (zooRegistry.lookup[sLower]) subject = zooRegistry.lookup[sLower];
        console.log(`[HOLISTIC MATCH] Selected Subject: "${subject}" (Score: ${highestScore.toFixed(2)})`);
        return finalizeSubject(subject, qLower);
    }

    // 5. Fallback loop for strict prefix overrides (only if holistic scoring falls flat)
    const qWords = qLower.split(/\s+/);
    for (let len = qWords.length; len >= 1; len--) {
        const phrase = qWords.slice(0, len).join(' ');
        if (priorityOverrides[phrase] !== undefined && !QUERY_STOP_WORDS.has(phrase)) {
            console.log(`[PRIORITY-FALLBACK] Override: "${phrase}" → "${priorityOverrides[phrase]}"`);
            return finalizeSubject(priorityOverrides[phrase], qLower);
        }
    }

    // 6. Multi-Entity Extraction Strategy
    let extracted = new Set();

    for (let len = qWords.length; len >= 1; len--) {
        for (let i = 0; i <= qWords.length - len; i++) {
            const phrase = qWords.slice(i, i + len).join(' ');
            if (priorityOverrides[phrase] && !QUERY_STOP_WORDS.has(phrase)) {
                extracted.add(priorityOverrides[phrase]);
            }
        }
    }

    for (const qw of qWords) {
        if (qw.length < 3 || QUERY_STOP_WORDS.has(qw) || ADJECTIVE_BLACKLIST.has(qw)) continue;
        if (zooRegistry.lookup[qw]) {
            const resolved = zooRegistry.lookup[qw];
            if (zooRegistry.eventNames.has(resolved) && !EVENT_INDICATOR_REGEX.test(qLower)) {
                // skip event match without event indicator
            } else {
                extracted.add(resolved);
                continue;
            }
        }
        const exactWordHit = zooRegistry.canonicalNames.find(n => {
            if (zooRegistry.eventNames.has(n) && !EVENT_INDICATOR_REGEX.test(qLower)) return false;
            return n.toLowerCase().split(/[^a-z0-9]+/).includes(qw);
        });
        if (exactWordHit) { extracted.add(exactWordHit); continue; }

        const fuzzyWordHit = zooRegistry.canonicalNames.find(n => {
            if (zooRegistry.eventNames.has(n) && !EVENT_INDICATOR_REGEX.test(qLower)) return false;
            const canonicalWords = n.toLowerCase().split(/[^a-z0-9]+/);
            for (const cw of canonicalWords) {
                if (cw.length < 4) continue; // FIX: Tightened from 4 to 5
                const maxDist = cw.length >= 7 ? 2 : 1; // FIX: Tightened from 6 to 7
                const lengthDiff = Math.abs(qw.length - cw.length);
                if (lengthDiff > maxDist) continue; // FIX: Prevent matching words of vastly different lengths
                if (levenshtein(qw, cw) <= maxDist) return true;
            }
            return false;
        });
        if (fuzzyWordHit) extracted.add(fuzzyWordHit);
    }

    if (extracted.size > 0) {
        let uniqueSubjects = Array.from(extracted).map(s => {
            const sLower = s.toLowerCase();
            return zooRegistry.lookup[sLower] || s;
        });
        uniqueSubjects = [...new Set(uniqueSubjects)];
        const finalSubject = uniqueSubjects.slice(0, 3).join(', ');
        return finalizeSubject(finalSubject, qLower, finalSubject);
    }


    const fuzzyFacilityHits = detectFacilitiesFuzzy(qLower);
    if (fuzzyFacilityHits.length > 0) {
        console.log(`[FUZZY-FACILITY-FALLBACK] Matched: ${fuzzyFacilityHits.join(', ')}`);
        return finalizeSubject(fuzzyFacilityHits.join(', '), qLower);
    }

    console.log(`[EXTRACT-FALLBACK] Rules failed. Waking up ${EXTRACTION_MODEL}...`);
    const llmMatch = await llmExtractSubject(qLower);
    if (llmMatch) {
        console.log(`[LLM-EXTRACT] ${EXTRACTION_MODEL} found: "${llmMatch}"`);
        return finalizeSubject(llmMatch, qLower, llmMatch);
    }

    return finalizeSubject('general', qLower, 'general');
}

function isAnimalActive(activityStr, currentHour) {
    const act = (activityStr || '').toLowerCase();

    if (!act) {
        return currentHour >= 6 && currentHour < 20;
    }

    if (act.includes('nocturnal')) {
        return currentHour >= 18 || currentHour < 6;
    }

    if (act.includes('diurnal')) {
        return currentHour >= 6 && currentHour < 18;
    }

    let activeTimes = [];
    if (act.includes('morning') || act.includes('dawn')) {
        activeTimes.push({ start: 5, end: 11 });
    }
    if (act.includes('evening') || act.includes('dusk')) {
        activeTimes.push({ start: 16, end: 20 });
    }
    if (act.includes('night') || act.includes('nocturnal')) {
        activeTimes.push({ start: 20, end: 6 });
    }
    if (act.includes('day')) {
        activeTimes.push({ start: 6, end: 18 });
    }

    if (activeTimes.length > 0) {
        return activeTimes.some(({ start, end }) => {
            if (start < end) {
                return currentHour >= start && currentHour < end;
            } else {
                return currentHour >= start || currentHour < end;
            }
        });
    }

    return currentHour >= 6 && currentHour < 20;
}

function formatDocumentText(docText, metadata, language) {
    if (language !== 'hi') {
        return docText;
    }
    if (metadata.name_hi || metadata.habitat_hi || metadata.narrative_hi) {
        return [
            `नाम: ${metadata.name_hi || metadata.name}`,
            metadata.location_hi ? `स्थान (Location): ${metadata.location_hi}` : (metadata.location ? `स्थान: ${metadata.location}` : ''),
            metadata.habitat_hi ? `आवास: ${metadata.habitat_hi}` : '',
            metadata.narrative_hi ? `विवरण: ${metadata.narrative_hi}` : ''
        ].filter(Boolean).join('\n');
    } else if (metadata.full_data) {
        try {
            const fd = JSON.parse(metadata.full_data);
            if (metadata.type === 'fact') {
                return `तथ्य: ${fd.title?.hi || fd.title?.en || ''}\nविवरण: ${fd.text?.hi || fd.text?.en || ''}`;
            } else if (metadata.type === 'rule') {
                const ruleTitle = fd.title?.hi || fd.title?.en || '';
                const ruleDesc = fd.description?.hi || fd.description?.en || '';
                const rulePoints = (fd.points || []).map(p => `  - ${p.title?.hi || p.title?.en || ''}${p.description ? ': ' + (p.description?.hi || p.description?.en || '') : ''}`).join('\n');
                return `नियम श्रेणी: ${ruleTitle}\nविवरण: ${ruleDesc}\nनियम:\n${rulePoints}`;
            } else if (metadata.type === 'fee') {
                const feeTitle = fd.title?.hi || fd.title?.en || '';
                const feeRows = (fd.rows || []).map(r => `  ${r.label?.hi || r.label?.en || ''}: ${r.value?.hi || r.value?.en || ''}`).join('\n');
                return `शुल्क श्रेणी: ${feeTitle}\nदरें:\n${feeRows}`;
            } else if (metadata.type === 'about') {
                const contentBlocks = fd.filter(c => c.type === 'text' && (c.text?.hi || c.text?.en)).map(c => c.text?.hi || c.text?.en);
                return `चिड़ियाघर के बारे में:\n\n${contentBlocks.join('\n\n')}`;
            } else if (metadata.type === 'key_facts') {
                const factLines = fd.map(f => `${f.label?.hi || f.label?.en || ''}: ${f.value?.hi || f.value?.en || ''}`).join('\n');
                return `चिड़ियाघर के मुख्य तथ्य:\n${factLines}`;
            } else if (metadata.type === 'contact') {
                const methodsHi = fd.map(m => `${m.label?.hi || m.label?.en || ''}: ${m.value?.hi || m.value?.en || ''}${m.actionUrl ? ` (${m.actionUrl})` : ''}`).join('\n');
                return `चिड़ियाघर संपर्क जानकारी:\n${methodsHi}`;
            } else if (metadata.type === 'timings') {
                const daysHi = {
                    'Monday': 'सोमवार', 'Tuesday': 'मंगलवार', 'Wednesday': 'बुधवार', 'Thursday': 'गुरुवार',
                    'Friday': 'शुक्रवार', 'Saturday': 'शनिवार', 'Sunday': 'रविवार'
                };
                const timingsHi = fd.map(t => `${daysHi[t.day] || t.day}: ${t.isOpen ? `${t.openTime} से ${t.closeTime}` : 'बंद'}`).join('\n');
                return `चिड़ियाघर के खुलने का समय:\n${timingsHi}`;
            } else if (metadata.type === 'holidays') {
                const holidaysHi = fd.map(h => `- ${h.name?.hi || h.name?.en || h}`).join('\n');
                return `चिड़ियाघर के अवकाश/छुट्टियां:\n${holidaysHi}`;
            } else if (metadata.type === 'news') {
                return `समाचार: ${fd.title?.hi || fd.title?.en || fd.title || ''}\nविवरण: ${fd.text?.hi || fd.text?.en || fd.text || ''}${fd.url ? `\nयूआरएल: ${fd.url}` : ''}`;
            } else if (metadata.is_event === 'true' || metadata.is_event === true) {
                const eventTitle = fd.title?.hi || fd.title?.en || fd.name?.hi || fd.name?.en || metadata.name;
                const eventDesc = fd.body?.hi || fd.body?.en || fd.description?.hi || fd.description?.en || '';
                let eventDate = '';
                if (fd.date) {
                    eventDate = new Date(fd.date).toLocaleDateString('hi-IN', { year: 'numeric', month: 'long', day: 'numeric' });
                } else if (fd.from_datetime) {
                    const from = new Date(fd.from_datetime).toLocaleDateString('hi-IN', { year: 'numeric', month: 'long', day: 'numeric' });
                    const to = fd.to_datetime ? new Date(fd.to_datetime).toLocaleDateString('hi-IN', { year: 'numeric', month: 'long', day: 'numeric' }) : '';
                    eventDate = to ? `${from} से ${to}` : from;
                }
                return `कार्यक्रम: ${eventTitle}\nतिथि: ${eventDate}\nविवरण: ${eventDesc}`;
            } else {
                return [
                    `नाम: ${fd.common_name?.hi || fd.name?.hi || fd.title?.hi || metadata.name}`,
                    `वैज्ञानिक नाम: ${fd.scientific_name?.hi || fd.scientific_name?.en || ''}`,
                    `श्रेणी: ${fd.category?.hi || fd.category?.en || ''}`,
                    `आवास: ${fd.habitat?.hi || fd.habitat?.en || ''}`,
                    `आहार: ${fd.diet?.hi || fd.diet?.en || ''}`,
                    `स्थान (Location): ${fd.location?.location_name?.hi || fd.location?.location_name?.en || metadata.location || ''}`,
                    `विवरण: ${fd.narrative?.hi || fd.description?.hi || fd.narrative?.en || ''}`,
                    `कहानियाँ: ${fd.story_description?.hi || fd.story_description?.en || ''}`
                ].filter(s => s && !s.endsWith(': ')).join('\n');
            }
        } catch (e) {
            console.error('Hindi metadata parse error in formatDocumentText:', e.message);
        }
    }
    return docText;
}

async function antigravitySearch(query, subject, isFacilityMatch, topK = 5, language = 'en', isEventQuery = false) {
    console.log(`\n[SEARCH] Query: "${query}" (Lang: ${language}, EventQuery: ${isEventQuery})`);
    console.log(`[ENTITY] Target Subject: "${subject}"`);

    const chromaCacheKey = `${subject}:${language}`;
    const cachedChromaResult = getCachedChromaSearch(chromaCacheKey);
    if (cachedChromaResult) {
        console.log(`[CHROMA-CACHE] HIT for "${subject}"`);
        return cachedChromaResult;
    }
    let vectorSubject = subject;
    if (subject.includes(',')) {
        console.log(`[SEARCH] Multi-entity detected ("${subject}"). Switching vector search to raw query for better semantic blending.`);
        vectorSubject = query;
    }

    const queryEmbedding = await getCachedEmbedding(subject);
    if (!queryEmbedding) {
        return { context: '', subject: 'general', references: [], topScore: 0 };
    }

    const subjLower = subject.toLowerCase();
    const isConservation = subjLower.includes('endangered') ||
        subjLower.includes('conservation') ||
        subjLower.includes('threatened') ||
        subjLower.includes('vulnerable') ||
        subjLower.includes('extinct') ||
        subjLower.includes('संकटग्रस्त');

    if (isConservation) {
        const endangeredList = zooRegistry.canonicalNames.filter(name => {
            const status = String(zooRegistry.metadata[name]?.threatStatus || '').toLowerCase();
            if (!status || status === 'undefined') return false;

            if (subjLower.includes('near')) return status.includes('near');
            if (subjLower.includes('critically')) return status.includes('critically');
            if (subjLower.includes('vulnerable')) return status.includes('vulnerable');
            if (subjLower.includes('extinct')) return status.includes('extinct');

            return status.includes('endangered') || status.includes('threatened') || status.includes('vulnerable');
        });

        if (endangeredList.length > 0) {
            console.log(`[ENTITY] Conservation query. Found ${endangeredList.length} species.`);
            const listStr = endangeredList.slice(0, 15).join(', ');

            // Use English context for the LLM
            const contextStr = `The National Zoological Park, New Delhi is home to many species.\nSome key ${subject} animals here are: ${listStr}.\nVisitors are encouraged to learn about their conservation.`;

            return {
                context: contextStr,
                sortedContext: endangeredList.slice(0, 5).map(name => ({
                    metadata: { name }, score: 1.0, doc: `This is the ${name}.`
                })),
                topScore: 1.0,
                subject: subject, // Preserve the original query subject instead of forcing 'Endangered'
                references: endangeredList.slice(0, 5)
            };
        }
    }
    let exactMatch = null;
    const matches = Array.from(zooRegistry.rawNames).filter(
        raw => raw.replace(/\s+\d+$/, '').trim().toLowerCase() === subject.toLowerCase()
    );
    const hasMultipleEntities = matches.length > 1;

    try {
        const idsToTry = [subject];
        for (const rawName of matches) {
            if (!idsToTry.includes(rawName)) idsToTry.push(rawName);
        }
        const meta = zooRegistry.metadata[subject];
        if (meta?.classification) {
            const firstClassWord = meta.classification.split(/[/\s,]+/)[0];
            if (firstClassWord && firstClassWord.length > 2 && !subject.includes(' ')) {
                idsToTry.push(`${subject} ${firstClassWord}`);
            }
        }

        const getRes = await collection.get({ ids: idsToTry });
        if (getRes?.ids?.length > 0) {
            const docName = (getRes.metadatas[0]?.name || '').toLowerCase();
            const subjectLower = subject.toLowerCase();
            const subjectWords = subjectLower.split(/\s+/);
            const docWords = docName.split(/\s+/);

            const overlap = subjectWords.filter(w => docWords.includes(w)).length;
            const isGenuineMatch = overlap >= Math.ceil(subjectWords.length * 0.8);

            if (isGenuineMatch) {
                const isEventMatch = getRes.metadatas[0]?.is_event === 'true';
                console.log(`[ENTITY] Exact ID Match: "${getRes.ids[0]}" (IsEvent: ${isEventMatch})`);

                const finalScore = (isEventMatch && !isEventQuery) ? 0.15 : 1.5;

                exactMatch = {
                    doc: getRes.documents[0],
                    metadata: getRes.metadatas[0],
                    score: finalScore,
                    originalName: getRes.ids[0]
                };
            }
        }
    } catch (e) { /* not found by ID */ }

    if (!exactMatch && /\b(national|international|world|global)\b/i.test(subject)) {
        const coreKeyword = subject
            .replace(/\b(national|international|world|global|day|for|to|the|of|and|in|a)\b/gi, '')
            .trim();

        if (coreKeyword) {
            console.log(`[EVENT] Core keyword fallback: "${coreKeyword}"`);
            const coreEmbedding = await getCachedEmbedding(coreKeyword + ' day');
            let coreResults = await collection.query({
                queryEmbeddings: [coreEmbedding],
                nResults: 5,
                where: { is_event: 'true' }
            });

            if (!coreResults.ids?.[0]?.length) {
                coreResults = await collection.query({
                    queryEmbeddings: [coreEmbedding],
                    nResults: 5
                });
            }

            if (coreResults.ids?.[0]?.length > 0) {
                console.log(`[EVENT] Fallback match: "${coreResults.ids[0][0]}"`);
                exactMatch = {
                    doc: coreResults.documents[0][0],
                    metadata: coreResults.metadatas[0][0],
                    score: 1.2,
                    originalName: coreResults.ids[0][0]
                };
            }
        }
    }

    let results = { documents: [[]], metadatas: [[]], distances: [[]] };
    try {
        const needsVectorQuery = !exactMatch || exactMatch.score < 1.0 || hasMultipleEntities;
        if (needsVectorQuery) {
            results = await collection.query({
                queryEmbeddings: [queryEmbedding],
                nResults: 6
            });
        }
    } catch (e) {
        console.warn('[SEARCH] Vector query failed:', e.message);
    }

    const documents = results.documents?.[0] || [];
    const metadatas = results.metadatas?.[0] || [];
    const distances = results.distances?.[0] || [];

    const seenIds = new Set();
    const scoredContext = [];
    if (exactMatch) {
        exactMatch.doc = formatDocumentText(exactMatch.doc, exactMatch.metadata, language);
        scoredContext.push(exactMatch);
        seenIds.add(exactMatch.metadata?.doc_id || exactMatch.originalName);
    }

    for (let i = 0; i < documents.length; i++) {
        const metadata = metadatas[i] || {};
        const docId = metadata.doc_id || metadata.name;
        if (docId && seenIds.has(docId)) continue;
        if (docId) seenIds.add(docId);

        const distance = distances[i] || 1.0;
        const docName = (metadata.name || '').toLowerCase().replace(/\d+/g, '').trim();

        const baseSimilarity = Math.max(0, 1.0 - distance);

        let score = (subject === query) ? baseSimilarity : (baseSimilarity * 0.4);

        if (subject !== 'general' && subject !== query) {
            const subjectLower = subject.toLowerCase();
            const docWords = docName.split(/\s+/);
            const subjectWords = subjectLower.split(/\s+/);

            const matchedWords = subjectWords.filter(sw => docWords.includes(sw));
            const matchCount = matchedWords.length;
            const missingCount = subjectWords.length - matchCount;
            const overlapRatio = matchCount / subjectWords.length;
            const penaltyRatio = missingCount / subjectWords.length;

            score += overlapRatio * 0.5;
            score -= penaltyRatio * 0.3;

            if (docName === subjectLower) score += 0.3;
            else if (docName.endsWith(subjectLower)) score += 0.15;
            else if (docName.startsWith(subjectLower)) score += 0.08;

            if (missingCount === 0 && metadata.scientific_name) score += 0.1;

            if (metadata.is_event === 'true' && !isEventQuery) {
                score = Math.min(score, 0.1);
            }

            if (matchCount === 0 && distance > 0.5) score = 0;
        }

        let docText = formatDocumentText(documents[i], metadata, language);

        const displayName = metadata.render_name || metadata.common_name || metadata.name;
        scoredContext.push({ doc: docText, score, originalName: displayName, metadata });
    }

    let sortedContext = scoredContext
        .filter(item => item.score > 0)
        .sort((a, b) => b.score - a.score);

    const subjectList = subject.split(',').map(s => s.trim());
    if (sortedContext.length === 0 && subject !== 'general') {
        for (const singleSubject of subjectList) {
            if (zooRegistry.canonicalNames.includes(singleSubject)) {
                console.log(`[SEARCH] Fallback: No vector results, but "${singleSubject}" is in registry. Using metadata.`);
                const meta = zooRegistry.metadata[singleSubject] || {};
                const fallbackItem = {
                    // FIX: Use structured, explicit facts so the LLM doesn't get confused.
                    doc: `STATUS: Present at National Zoological Park, New Delhi.\nLOCATION: Not Available in Database.\nCLASS: ${meta.classification || 'Animal'}`,
                    metadata: { name: singleSubject, ...meta },
                    score: 0.5
                };
                scoredContext.push(fallbackItem);
                sortedContext.push(fallbackItem);
            }
        }
    }

    const isTraitSearch = (subject === query);
    const topScore = sortedContext.length > 0 ? sortedContext[0].score : 0;
    const refThreshold = isTraitSearch ? 0.35 : Math.max(0.65, topScore * 0.90);

    let bestMatchName = subject;
    const isSpecificAnimal = zooRegistry.canonicalNames.includes(subject) || subject.includes(',');
    if (sortedContext.length > 0) {
        const topMeta = sortedContext[0].metadata || {};
        const metaName = (topMeta.render_name || topMeta.common_name || topMeta.name || '')
            .replace(/\s+\d+$/, '').trim();
        const isVague = ['general', 'animals', 'birds', 'reptiles', 'mammals', 'fish']
            .includes(subject.toLowerCase());
        if ((!isSpecificAnimal || isVague || topScore > 1.2) && metaName && !/^[0-9a-fA-F]{24}$/.test(metaName)) {
            bestMatchName = metaName;
        }
    }

    const requiresHighThreshold = !isSpecificAnimal || ['general', 'animals', 'birds', 'reptiles', 'mammals', 'fish']
        .includes(subject.toLowerCase());
    if (((requiresHighThreshold && topScore < 0.65) || topScore < 0.2) && !isFacilityMatch) {
        bestMatchName = 'general';
    }

    let references = sortedContext
        .filter(item => item.score >= refThreshold)
        .filter(item => {
            const rName = (item.metadata?.common_name || item.originalName || '').toLowerCase().replace(/\s+\d+$/, '').trim();
            const bName = bestMatchName.toLowerCase().replace(/\s+\d+$/, '').trim();
            return rName !== bName;
        })
        .map(item => {
            const rawName = item.metadata?.common_name || item.originalName;
            return rawName ? rawName.replace(/\s+\d+$/, '').trim() : null;
        })
        .filter(Boolean);

    references = [...new Set(references)].slice(0, 2);

    const isFacilityName = /Washroom|Drinking Water|Buggy Stops|Food & Drinks|First Aid|Counters/.test(bestMatchName);
    if (isFacilityName) references = [];

    const result = {
        context: bestMatchName === 'general' ? '' : optimizeContext(sortedContext.slice(0, topK).map(i => i.doc)),
        subject: bestMatchName,
        extractedSubject: subject,
        references: bestMatchName === 'general' ? [] : [...new Set(references)],
        topScore: bestMatchName === 'general' ? 0 : topScore,
        isFacilityMatch: !!isFacilityMatch,
        sortedContext
    };

    setCachedChromaSearch(`${subject}:${language}`, result);
    return result;
}

function filterReferences(references, text, finalSubject) {
    if (!references || !Array.isArray(references) || references.length === 0) return [];
    const ansLower = (text || '').toLowerCase().replace(/-/g, ' ');
    const cleanFinal = (finalSubject || '').toLowerCase().replace(/\s+\d+$/, '').trim();

    return references.filter(ref => {
        const refClean = ref.toLowerCase().replace(/\s+\d+$/, '').replace(/-/g, ' ').trim();
        if (refClean === cleanFinal) return true;
        if (ansLower.includes(refClean)) return true;
        return false;
    });
}

function sendStaticResponse(res, answer, keyword, stream, references = []) {
    if (stream) {
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.write(`data: ${JSON.stringify({ token: '', status: 'thinking' })}\n\n`);
        res.write(`data: ${JSON.stringify({ token: answer })}\n\n`);
        res.write(`data: ${JSON.stringify({ done: true, keyword, references })}\n\n`);
        return res.end();
    } else {
        return res.json({ answer, keyword, references });
    }
}

const STATIC_RESPONSES = {
    en: {
        'where is the lion': { answer: '🦁 The Asiatic Lions are located in the Asiatic Lion Area, near the Main Entrance. Follow the signs — you can\'t miss them! 🗺️', keyword: 'Asiatic Lion' },
        'where is the tiger': { answer: '🐅 The White Tigers are in the White Tiger Area, a short walk from the Main Entrance. Look for the big cat signs! 🗺️', keyword: 'White Tiger' },
        'where is the elephant': { answer: '🐘 The Indian Elephants are near the northern section of the zoo. Follow the elephant signs from the Main Entrance! 🗺️', keyword: 'Indian Elephant' },
        'show me reptiles': { answer: '🐍 Head to the Reptile House to see cobras, pythons, crocodiles, and more! It\'s clearly marked on the zoo map. 🗺️', keyword: 'Reptile House' },
        'show reptiles': { answer: '🐍 The Reptile House is home to fascinating snakes, lizards, and crocodiles. Follow the signs! 🦎', keyword: 'Reptile House' },
        'where are the reptiles': { answer: '🐍 Reptiles are at the Reptile House — look for the signs near the central path! 🗺️', keyword: 'Reptile House' },
        'what animals are here': { answer: '🦁 The zoo is home to lions, tigers, elephants, peacocks, deer, reptiles, and many more! What would you like to visit first? 🐅', keyword: 'general' },
        'what can i see here': { answer: '🌟 You can see Asiatic Lions, White Tigers, Indian Elephants, Reptiles, Peacocks, and much more here! 🦁 What interests you?', keyword: 'general' },
        'where is food': { answer: '🍽️ Food & Drinks stalls are available at multiple spots in the zoo. Look for the canteen signs near the main path! 😊', keyword: 'Food & Drinks' },
        'i am hungry': { answer: '🍽️ Don\'t worry! Head to the Food & Drinks stalls near the main path for snacks and refreshments. 😊', keyword: 'Food & Drinks' },
        'where is washroom': { answer: '🚻 Washrooms are available at multiple locations throughout the zoo. Look for the WC signs! 🗺️', keyword: 'Washrooms' },
        'where is toilet': { answer: '🚻 Toilets are at several spots in the zoo. Follow the WC signs near the main paths! 🗺️', keyword: 'Washrooms' },
        'where is water': { answer: '💧 Drinking water points are available throughout the zoo. Look for the water fountain signs! 😊', keyword: 'Drinking Water' },
        'i am thirsty': { answer: '💧 Drinking water points are available throughout the zoo — follow the water fountain signs! 😊', keyword: 'Drinking Water' },
        'where is exit': { answer: '🚪 The Exit Gate is right next to the Main Entrance. I\'ve marked it on the map for you! 🗺️', keyword: 'Exit Gate' },
        'how to exit': { answer: '🚪 Head back towards the Main Entrance — the Exit Gate is right there! 🗺️', keyword: 'Exit Gate' },
        'where is entry': { answer: '🎟️ The Main Entrance is at the front of the zoo. Check the map for the exact location! 🗺️', keyword: 'Main Entrance' },
        'where is ticket counter': { answer: '🎟️ Ticket counters are at the Main Entrance. Head to the front gate! 😊', keyword: 'Counters' },
        'where is buggy': { answer: '🚗 Buggy stops are at several points around the zoo. Look for the buggy/shuttle signs! 🗺️', keyword: 'Buggy Stops' },
        'lion habitat': { answer: '🦁 Asiatic Lions live in dry forests and grasslands. Here at the zoo, they\'re in the Asiatic Lion Area! 🌳', keyword: 'Asiatic Lion' },
        'tiger habitat': { answer: '🐅 White Tigers prefer cool, forested areas. Our tigers are in the White Tiger Area, beautifully designed! 🌳', keyword: 'White Tiger' },
        'elephant habitat': { answer: '🐘 Indian Elephants live in the northern section with plenty of space to roam. 🌳', keyword: 'Indian Elephant' },
        'peacock location': { answer: '🦚 Peacocks roam freely throughout the zoo! You\'ll spot them near the main pathways. 🗺️', keyword: 'Indian Peafowl (Leucistic)' },
        'where are snakes': { answer: '🐍 All snakes are safely housed in the Reptile House. 🦎', keyword: 'Reptile House' },
        'where are crocodiles': { answer: '🐊 Crocodiles are in the Reptile House section. Amazing creatures! 🦎', keyword: 'Reptile House' },
        'first aid': { answer: '🏥 First Aid services are available at the First Aid counter near the Counters. Ask staff for help! 😊', keyword: 'First Aid' },
        'medical help': { answer: '🏥 Medical assistance is available at the First Aid counter. Staff can help you! 😊', keyword: 'First Aid' },
        'i need help': { answer: '👋 I\'m here to help! Are you looking for an animal, facility, or something else? 😊', keyword: 'general' },
        'thank you': { answer: '😊 You\'re welcome! Enjoy your time at the zoo! 🦁', keyword: 'general' },
        'thanks': { answer: '😊 Happy to help! Enjoy exploring! 🦁', keyword: 'general' },
        'bye': { answer: '👋 Goodbye! Thanks for visiting the National Zoological Park! 🦁', keyword: 'general' },
        'goodbye': { answer: '👋 Goodbye! Hope you had a great time! 🦁', keyword: 'general' },
        'see you': { answer: '👋 See you again! Enjoy the zoo! 🦁', keyword: 'general' },
    },
    hi: {
        'शेर कहाँ है': { answer: '🦁 एशियाई शेर एशियाई शेर क्षेत्र में हैं, मुख्य प्रवेश द्वार के पास। संकेतों का पालन करें! 🗺️', keyword: 'Asiatic Lion' },
        // 'एशियाई शेर कहाँ है': { answer: '🦁 एशियाई शेर एशियाई शेर क्षेत्र में हैं, मुख्य प्रवेश द्वार के पास। संकेतों का पालन करें! 🗺️', keyword: 'Asiatic Lion' }, // <-- ADD THIS LINE
        'बाघ कहाँ है': { answer: '🐅 सफेद बाघ सफेद बाघ क्षेत्र में हैं। मुख्य द्वार से बाईं ओर चलें! 🗺️', keyword: 'White Tiger' },
        'हाथी कहाँ है': { answer: '🐘 भारतीय हाथी उत्तरी क्षेत्र में हैं। हाथी के संकेतों का पालन करें! 🗺️', keyword: 'Indian Elephant' },
        'खाना कहाँ है': { answer: '🍽️ खाने-पीने के स्टॉल मुख्य पथ के पास कई जगह उपलब्ध हैं! 😊', keyword: 'Food & Drinks' },
        'पानी कहाँ है': { answer: '💧 पीने के पानी के स्थान पूरे चिड़ियाघर में उपलब्ध हैं। वॉटर फाउंटेन के संकेत देखें! 😊', keyword: 'Drinking Water' },
        'शौचालय कहाँ है': { answer: '🚻 शौचालय पूरे चिड़ियाघर में कई जगह हैं। WC के संकेत देखें! 🗺️', keyword: 'Washrooms' },
        'निकास कहाँ है': { answer: '🚪 निकास द्वार मुख्य प्रवेश द्वार के पास ही है! 🗺️', keyword: 'Exit Gate' },
        'सांप कहाँ हैं': { answer: '🐍 सभी सांप सरीसृप घर में सुरक्षित रूप से रखे गए हैं! 🦎', keyword: 'Reptile House' },
        'मगरमच्छ कहाँ है': { answer: '🐊 मगरमच्छ सरीसृप घर में हैं। शानदार जानवर! 🦎', keyword: 'Reptile House' },
        'धन्यवाद': { answer: '😊 आपका स्वागत है! चिड़ियाघर का आनंद लें! 🦁', keyword: 'general' },
        'अलविदा': { answer: '👋 अलविदा! राष्ट्रीय प्राणी उद्यान आने के लिए धन्यवाद! 🦁', keyword: 'general' },
        'भूख लगी है': { answer: '🍽️ चिंता न करें! खाने-पीने के स्टॉल के पास जाएं। 😊', keyword: 'Food & Drinks' },
        'प्यास लगी है': { answer: '💧 पानी की बोतलें पूरे चिड़ियाघर में मिल जाएंगी! 😊', keyword: 'Drinking Water' },
    }
};

const HINGLISH_TO_HINDI = {
    'mujhe': 'मुझे',
    'sher': 'शेर',
    'dekhna': 'देखना',
    'h': 'है',
    'hai': 'है',
    'kahan': 'कहाँ',
    'kaha': 'कहाँ',
    'kidhar': 'किधर',
    'hathi': 'हाथी',
    'bhalu': 'भालू',
    'bandar': 'बंदर',
    'mor': 'मोर',
    'khana': 'खाना',
    'paani': 'पानी',
    'pani': 'पानी',
    'kya': 'क्या',
    'kaise': 'कैसे',
    'kab': 'कब',
    'khula': 'खुला',
    'band': 'बंद',
    'ticket': 'टिकट',
    'tikit': 'टिकट',
    'ghumna': 'घूमना',
    'ghumi': 'घूमना',
    'chalo': 'चलो',
    'chalna': 'चलना',
    'shaili': 'शैली',
    'batao': 'बताओ',
    'bata': 'बताओ',
    'dikhao': 'दिखाओ',
    'dikhaye': 'दिखाओ',
    'tha': 'था',
    'he': 'है',
    'se': 'से',
    'ko': 'को',
    'ke': 'के',
    'ki': 'की',
    'ka': 'का',
    'me': 'में',
    'par': 'पर',
    'ek': 'एक',
    'do': 'दो',
    'teen': 'तीन',
    'chaar': 'चार',
    'paanch': 'पाँच',
    'chah': 'छह',
    'saat': 'सात',
    'aath': 'आठ',
    'nau': 'नौ',
    'das': 'दस',
    'kitna': 'कितना',
    'kitne': 'कितने',
    'kitni': 'कितनी',
    'kuch': 'कुछ',
    'bhi': 'भी',
    'aur': 'और',
    'ya': 'या',
    'nahi': 'नहीं',
    'na': 'ना',
    'mat': 'मत',
    'karna': 'करना',
    'kar': 'कर',
    'karta': 'करता',
    'karte': 'करते',
    'chahiye': 'चाहिए',
    'hota': 'होता',
    'hote': 'होते',
    'hoti': 'होती',
    'gaya': 'गया',
    'gaye': 'गये',
    'gayi': 'गयी',
    'hum': 'हम',
    'aap': 'आप',
    'tum': 'तुम',
    'main': 'मैं',
    'mera': 'मेरा',
    'meri': 'मेरी',
    'mere': 'मेरे',
    'apna': 'अपना',
    'apni': 'अपनी',
    'apne': 'अपने',
    'ab': 'अब',
    'tab': 'तब',
    'jab': 'जब',
    'sakte': 'सकते',
    'sakta': 'सकता',
    'sakti': 'सकती',
    'sakenge': 'सकेंगे',
    'baagh': 'बाघ',
    'bagh': 'बाघ',
    'cheetah': 'चीता',
    'chita': 'चीता',
    'safed': 'सफ़ेद',
    'genda': 'गैंडा',
    'mgar': 'मगरमच्छ',
    'magar': 'मगरमच्छ',
    'magarmach': 'मगरमच्छ',
    'saap': 'साँप',
    'saanp': 'साँप',
    'sap': 'साँप',
    'panchi': 'चिड़िया',
    'chidiya': 'चिड़िया',
    'prajati': 'प्रजाति',
    'prajatiyan': 'प्रजातियाँ',
    'vazan': 'वजन',
    'wazan': 'वजन',
    'umra': 'उम्र',
    'umr': 'उम्र',
    'khoj': 'ढूंढ',
    'dhoondh': 'ढूंढ',
    'milta': 'मिलता',
    'milega': 'मिलेगा',
    'chalega': 'चलेगा',
    'chalta': 'चलता',
    'peena': 'पीना',
    'pine': 'पीने',
    'khilauna': 'खिलौना',
    'baccha': 'बच्चा',
    'bacha': 'बच्चा',
    'bacche': 'बच्चे',
    'bache': 'बच्चे',
    'shuru': 'शुरू',
    'ant': 'अंत',
    'dwar': 'द्वार',
    'pravesh': 'प्रवेश',
    'nikas': 'निकास',
    'toilet': 'शौचालय',
    'washroom': 'वॉशरूम'
};

function transliterateHinglish(text) {
    if (!text) return text;
    return text.replace(/[a-zA-Z]+/g, (match) => {
        const lower = match.toLowerCase();
        return HINGLISH_TO_HINDI[lower] || match;
    });
}

function isSpecificTimingQuestion(query) {
    if (!query || typeof query !== 'string') return false;
    const q = query.toLowerCase();

    // Days of the week in English, Hindi, and common Hinglish spellings
    const days = [
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'somwar', 'somvaar', 'somvar', 'mangalwar', 'mangalvaar', 'mangalvar',
        'budhwar', 'budhvaar', 'budhvar', 'guruwar', 'guruvaar', 'guruvar', 'veerwar', 'veervaar', 'brihaspativar',
        'shukrawar', 'shukrawaar', 'shukravar', 'shukrvar', 'shanivar', 'shanivaar',
        'ravivar', 'ravivaar', 'itwar', 'etwar',
        'सोमवार', 'मंगलवार', 'बुधवार', 'गुरुवार', 'शुक्रवार', 'शनिवार', 'रविवार', 'इतवार'
    ];

    const hasDay = days.some(day => q.includes(day));
    if (hasDay) return true;

    // Terms indicating a check for today, tomorrow, holidays, or specific open/closed status
    const specificTerms = [
        'today', 'tomorrow', 'aaj', 'kal', 'आज', 'कल',
        'holiday', 'chutti', 'छुट्टी', 'vacation', 'festival',
        'open today', 'closed today', 'open now', 'closed now',
        'is it open', 'is it closed', 'open on', 'closed on'
    ];

    const hasSpecificTerm = specificTerms.some(term => q.includes(term));
    if (hasSpecificTerm) return true;

    return false;
}

app.post('/api/shera/chat', async (req, res) => {
    let { question, deepSearch = true, language = 'en', stream = false } = req.body;

    deepSearch = deepSearch === true || deepSearch === 'true';
    logResources('Incoming Chat');

    const isHindi = language === 'hi';
    if (isHindi && question) {
        question = transliterateHinglish(question);
    }
    const qLower = question.toLowerCase().trim();

    // Helper to replace raw names (with numbers/suffixes) with user-friendly render names
    function replaceRawNamesWithRenderNames(text, isHindi) {
        if (!text) return text;
        let cleanedText = text;

        // Sort rawNames by length descending to prevent substring mismatch issues
        const rawNamesSorted = Array.from(zooRegistry.rawNames)
            .sort((a, b) => b.length - a.length);

        for (const rawName of rawNamesSorted) {
            const mapping = zooRegistry.rawToRender[rawName];
            if (mapping) {
                const renderName = isHindi ? (mapping.hi || mapping.en) : mapping.en;
                if (renderName && renderName !== rawName) {
                    const escaped = rawName.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
                    cleanedText = cleanedText.replace(new RegExp(`\\b${escaped}\\b`, 'gi'), renderName);
                    cleanedText = cleanedText.replace(new RegExp(escaped, 'g'), renderName);
                }
            }
        }
        return cleanedText;
    }

    const originalJson = res.json;
    res.json = function (body) {
        if (body && typeof body.answer === 'string') {
            body.answer = replaceRawNamesWithRenderNames(body.answer, isHindi);
        }
        return originalJson.call(this, body);
    };

    console.log(`\n--- Incoming: "${question}" (DeepSearch: ${deepSearch}, Lang: ${language}, Stream: ${stream}) ---`);

    const staticMap = STATIC_RESPONSES[isHindi ? 'hi' : 'en'] || STATIC_RESPONSES.en;
    if (staticMap[qLower]) {
        const hit = staticMap[qLower];
        console.log(`[STATIC] Instant match for "${qLower}"`);
        return sendStaticResponse(res, hit.answer, hit.keyword, stream);
    }

    const qNormalized = qLower.replace(/[?!.,]/g, '').trim();
    if (staticMap[qNormalized] && qNormalized !== qLower) {
        const hit = staticMap[qNormalized];
        console.log(`[STATIC-FUZZY] Fuzzy match for "${qLower}" → "${qNormalized}"`);
        return sendStaticResponse(res, hit.answer, hit.keyword, stream);
    }

    // --- FIX 1: Global Location Intercept ---
    const isLocationQueryGlobal = qLower.includes('nearby')
        || qLower.includes('close to me')
        || qLower.includes('where am i')
        || qLower.includes('आसपास')
        || qLower.includes('नज़दीक')
        || qLower.includes('पास')
        || /\b(paas|pas|nazdeek|aaspaas)\b/.test(qLower);

    if (isLocationQueryGlobal) {
        console.log(`[GLOBAL] Location/nearby query intercepted: "${qLower}"`);
        const msg = isHindi
            ? "आपके आस-पास के जानवरों और सुविधाओं को ढूंढ रहा हूँ... 🗺️"
            : "Finding nearby animals and facilities for you... 🗺️";
        return sendStaticResponse(res, msg, 'general', stream);
    }
    
    // --- FIX 2: Global Timing Intercept ---
    if (isSpecificTimingQuestion(qLower)) {
        console.log(`[GLOBAL] Specific timing query intercepted: "${qLower}"`);

        // Await the dynamic timings from Chroma, or fallback to the static string
        const dynamicTimings = await getDynamicZooTimings(language);
        const fallbackTiming = language === 'hi'
            ? '🕒 दिल्ली चिड़ियाघर गर्मियों में सुबह 8:30 से शाम 4:30 तक और सर्दियों में सुबह 9:00 से शाम 4:00 तक खुला रहता है। ध्यान दें: चिड़ियाघर हर शुक्रवार को बंद रहता है! 📅'
            : '🕒 The zoo is open from 8:30 AM to 4:30 PM (Summer) and 9:00 AM to 4:00 PM (Winter). Note: The zoo is CLOSED on Fridays! 📅';

        return sendStaticResponse(res, dynamicTimings || fallbackTiming, 'Timings & Hours', stream);
    }
    // ----------------------------------------

    try {
        const isCountQuery = qLower.includes('how many')
            || qLower.includes('number of')
            || qLower.includes('count')
            || qLower.includes('total')
            || qLower.includes('कितने')
            || qLower.includes('संख्या')
            || qLower.includes('कुल')
            || /\b(kitne|kitna|total|count|number)\b/.test(qLower);

        const containsGenericCategory = [...GENERIC_CATEGORY_WORDS].some(word =>
            qLower.includes(word.toLowerCase())
        );

        // NEW: Intercept ANY query asking for "how many species", even if it mentions a specific animal like "lions"
        const isSpeciesCountQuery = /how many (species|types|kinds|different)/i.test(question) || /कितनी प्रजातियां/i.test(question);

        const forceGeneralRoute = (isCountQuery && containsGenericCategory) || isSpeciesCountQuery;

        if (!forceGeneralRoute) {
            const simpleCacheKey = `${language}:${qLower}`;
            const cached = getCachedResponse(simpleCacheKey);
            if (cached) {
                console.log(`[CACHE] HIT for "${qLower}"`);
                return sendStaticResponse(res, cached.answer, cached.keyword, stream, cached.references);
            }
        }

        let subject = 'general';
        let extractedSubject = 'general';
        let matchedFacility = null;

        if (!forceGeneralRoute) {
            const extractionResult = await extractSubject(question);
            subject = extractionResult.subject;
            extractedSubject = extractionResult.extractedSubject;
            matchedFacility = extractionResult.matchedFacility;

            if (subject && subject !== 'general') {
                const enrichedCacheKey = `${language}:${subject}:${qLower}`;
                const cachedEnriched = getCachedResponse(enrichedCacheKey);
                if (cachedEnriched) {
                    console.log(`[CACHE] HIT (enriched) for "${qLower}" subject="${subject}"`);
                    return sendStaticResponse(res, cachedEnriched.answer, cachedEnriched.keyword, stream, cachedEnriched.references);
                }
            }
        } else {
            console.log(`[ROUTE-OVERRIDE] Generic category count query detected. Forcing "general" route, bypassing extraction and cache checks.`);
        }

        if (subject !== 'general') {
            getCachedEmbedding(subject).catch(() => { });
        }

        const restrictedVerbs = /\b(eat|eating|feed|feeding|khau|khana|kill|killing|hurt|hurting|hit|hitting|punch|punching|kick|kicking|shoot|shooting|ride|riding|steal|stealing|touch|touching|pet|petting|tease|teasing|hunt|hunting|catch|catching|harass|harassing|maar|maarna|pito|pakad|pakadna|chhu|chuna|chori|shikar|nuksan|fuck|fucking|sex|rape|raping|abuse|abusing|torture|torturing|slap|slapping|beat|beating)\b/i;

        const containsAnimal = Object.keys(zooRegistry.lookup).some(animal =>
            animal.length > 2 && qLower.includes(animal)
        );

        const isRestrictedAction = restrictedVerbs.test(qLower) && containsAnimal;

        if (matchedFacility && !question.toLowerCase().includes('tell me') && !question.toLowerCase().includes('information')) {

            const facilityResponses = {
                'Food & Drinks': isHindi
                    ? '🍽️ खाने-पीने के स्टॉल पूरे चिड़ियाघर में उपलब्ध हैं! 😊'
                    : '🍽️ Food & Drinks stalls are available throughout the zoo! 😊',
                'Washrooms': isHindi
                    ? '🚻 शौचालय कई जगहों पर उपलब्ध हैं। WC के संकेत देखें! 🗺️'
                    : '🚻 Washrooms available at multiple locations. Look for signs! 🗺️',
                'Drinking Water': isHindi
                    ? '💧 पीने का पानी पूरे चिड़ियाघर में हैं! 😊'
                    : '💧 Drinking water is throughout the zoo! 😊',
                'Buggy Stops': isHindi
                    ? '🚗 बग्गी स्टॉप विभिन्न स्थानों पर हैं! 🗺️'
                    : '🚗 Buggy stops at various locations! 🗺️',
                'First Aid': isHindi
                    ? '🏥 प्रथम चिकित्सा सेवा उपलब्ध है। कर्मचारियों से सहायता लें! 😊'
                    : '🏥 First Aid services available. Ask staff for help! 😊',
                'Timings & Hours': isHindi
                    ? '🕒 दिल्ली चिड़ियाघर गर्मियों में सुबह 8:30 से शाम 4:30 तक और सर्दियों में सुबह 9:00 से शाम 4:00 तक खुला रहता है। ध्यान दें: चिड़ियाघर हर शुक्रवार को बंद रहता है! 📅'
                    : '🕒 The zoo is open from 8:30 AM to 4:30 PM (Summer) and 9:00 AM to 4:00 PM (Winter). Note: The zoo is CLOSED on Fridays! 📅',
                'Feeding Animals': isHindi
                    ? '🚫 चिड़ियाघर में जानवरों को खाना खिलाना सख्त मना है। इससे उनके स्वास्थ्य को नुकसान पहुंच सकता है। कृपया केवल उन्हें देखकर आनंद लें! 😊'
                    : '🚫 Feeding animals is strictly prohibited at the zoo. It can harm their health. Please enjoy watching them instead! 😊',
            };

            // 1. VETO GUARD: See if we need to cancel the facility match entirely
            if (isRestrictedAction) {
                console.log(`[VETO] Restricted action + Animal detected in raw query. Canceling Facility Shortcut.`);
                matchedFacility = null; // Veto the facility entirely
            }

            // 2. Build the facility answer ONLY if it wasn't vetoed
            let facilityAnswer = null;
            if (matchedFacility) {

                // Check for dynamic timings
                if (matchedFacility === 'Timings & Hours') {
                    const dynamicTimings = await getDynamicZooTimings(language);
                    if (dynamicTimings) facilityAnswer = dynamicTimings;
                }

                // If it wasn't dynamic timings, grab the static response
                if (!facilityAnswer) {
                    if (facilityResponses[matchedFacility]) {
                        facilityAnswer = facilityResponses[matchedFacility];
                    } else if (matchedFacility.includes(',')) {
                        const parts = matchedFacility.split(',').map(p => p.trim());
                        const combinedAnswers = parts.map(p => facilityResponses[p]).filter(Boolean);
                        if (combinedAnswers.length > 0) facilityAnswer = combinedAnswers.join(' \n');
                    }
                }
            }

            // 3. Routing Logic (Only triggers if we successfully built an answer)
            // 3. Routing Logic (Only triggers if we successfully built an answer)
            if (facilityAnswer) {
                const hasAnimalSubject = subject && subject !== 'general'
                    && !subject.split(',').every(p =>
                        p.trim() === 'Timings & Hours' || p.trim() === 'Feeding Animals' || facilityResponses[p.trim()] !== undefined
                    );

                if (!hasAnimalSubject) {
                    // Pure facility query — instant return (Guarantees zero hallucinations)
                    console.log(`[FACILITY-SHORT CUT] Instant response for "${matchedFacility}"`);
                    return sendStaticResponse(res, facilityAnswer, matchedFacility, stream);
                } else {
                    // Mixed query (facility + animal): prepend facility answer
                    console.log(`[FACILITY+ANIMAL] Prepending facility answer for "${matchedFacility}", continuing animal search for "${subject}"`);
                    res.locals.prependAnswer = facilityAnswer;
                }
            }
        }

        if (subject === 'general') {
            const isGreetingOrCasual = isCasualChatQuery(question);
            const isLocationQuery = extractedSubject === 'location'
                || qLower.includes('nearby')
                || qLower.includes('close to me')
                || qLower.includes('where am i')
                || qLower.includes('आसपास')
                || qLower.includes('नज़दीक')
                || qLower.includes('पास')
                || /\b(paas|pas|nazdeek|aaspaas)\b/.test(qLower);

            const isActivityQuery = qLower.includes('active now')
                || qLower.includes('currently active')
                || qLower.includes('active right now')
                || (qLower.includes('active') && qLower.includes('animal'))
                || qLower.includes('सक्रिय');

            if (isActivityQuery) {
                subject = 'Activity';
                extractedSubject = 'Activity';
            } else if (!isGreetingOrCasual && !isLocationQuery && !forceGeneralRoute) {
                const testSearch = await antigravitySearch(question, question, false, 1, language, false);

                // FIX: Increased threshold from 0.35 to 0.65 to prevent weak vector matches from overriding general queries
                if (testSearch.topScore >= 0.65) {
                    console.log(`[GENERAL] High similarity match found (${testSearch.topScore.toFixed(2)}). Promoting to factual search for "${testSearch.subject}".`);
                    subject = testSearch.subject;
                    extractedSubject = testSearch.subject;
                }
            }
        }

        const subjectParts = subject.split(', ');
        const hasExit = subjectParts.includes('Exit Gate');
        const hasEntry = subjectParts.includes('Main Entrance');

        if (hasExit || hasEntry) {
            // Guard: if the user is asking about ANIMALS near/at the entrance or exit
            // (e.g. "animals near the entrance"), do NOT return a static gate response.
            // Instead, fall through to the vector search so ChromaDB can return actual animals.
            const isAnimalProximityQuery =
                /\b(animal|animals|species|जानवर|janwar|pashu)\b/i.test(qLower) &&
                /\b(near|close|around|beside|at|by|paas|nazdeek|aaspaas|नज़दीक|आसपास|पास)\b/i.test(qLower);

            if (isAnimalProximityQuery) {
                console.log(`[LOCATION-ANIMAL] Animal-proximity query detected — skipping static gate response, falling through to vector search.`);
                // Use the full question as the search subject so the embedding captures
                // the location context (animals located near Main Entrance / Exit Gate).
                subject = question;
                matchedFacility = null; // clear so isFacilityMatch is false downstream
            } else {
                let answer, keyword;
                if (hasExit && hasEntry) {
                    console.log(`[FACILITY] Direct static gate response: "Exit Gate, Main Entrance"`);
                    keyword = 'Exit Gate, Main Entrance';
                    answer = isHindi
                        ? "चिड़ियाघर का मुख्य प्रवेश द्वार (Main Entrance) और निकास द्वार (Exit Gate) एक-दूसरे के पास ही स्थित हैं। मैंने आपकी सुविधा के लिए दोनों को मानचित्र पर चिह्नित कर दिया है! 🧭"
                        : "The Main Entrance and Exit Gate are located right next to each other. I've highlighted both on the map for you! 🧭";
                } else if (hasExit) {
                    console.log(`[FACILITY] Direct static gate response: "Exit Gate"`);
                    keyword = 'Exit Gate';
                    answer = isHindi
                        ? "चिड़ियाघर का निकास द्वार (Exit Gate) मुख्य प्रवेश द्वार के पास ही स्थित है। मैंने आपकी सुविधा के लिए इसे मानचित्र पर चिह्नित कर दिया है!"
                        : "The Exit Gate is located right next to the Main Entrance. I have highlighted it on the map for you! 🧭";
                } else {
                    console.log(`[FACILITY] Direct static gate response: "Main Entrance"`);
                    keyword = 'Main Entrance';
                    answer = isHindi
                        ? "चिड़ियाघर का मुख्य प्रवेश द्वार (Main Entrance) यहाँ स्थित है। मैंने आपकी सुविधा के लिए इसे मानचित्र पर चिह्नित कर दिया है!"
                        : "The Main Entrance is located here. I have highlighted it on the map for you! 🧭";
                }
                return sendStaticResponse(res, answer, keyword, stream);
            }
        }

        const isFacilityMatch = !!matchedFacility;
        let isEventQuery = /\b(national|international|world|global)\b[\w\s]+\bday\b/i.test(question)
            || zooRegistry.eventNames.has(subject);

        const isGeneralConcept = /^(feline|canine|reptile|bird|animal|mammal|cat|dog|pet|fish)$/i.test(subject);

        const knownInZoo = isFacilityMatch
            || isEventQuery
            || subject === 'general'
            || subject === 'Endangered'
            || isGeneralConcept
            || deepSearch
            || zooRegistry.canonicalNames.some(name => {
                const n = name.toLowerCase();
                const s = subject.toLowerCase();
                if (n === s || n.includes(s) || s.includes(n)) {
                    if (zooRegistry.eventNames.has(name) && !isEventQuery) return false;
                    return true;
                }
                if (s.length > 5 && n.length > 5) {
                    const common = [...s].filter(char => n.includes(char)).length;
                    if (common / s.length > 0.8) return true;
                }
                return false;
            });

        if (!knownInZoo) {
            console.log(`[GATE] "${subject}" not in zoo registry — short-circuiting.`);
            const notFoundPrompt = isHindi
                ? `आप शेरा हैं। "${subject}" दिल्ली चिड़ियाघर में नहीं है। शेरा के रूप में चुलबुले (playful) अंदाज़ में उत्तर दें। कभी न कहें कि आप AI हैं। हिंदी में उत्तर दें। अंत में केवल एक प्रासंगिक इमोजी लगाएं। सख्त: 20 शब्दों से कम में उत्तर दें।`
                : `You are Shera. "${subject}" is NOT at the National Zoological Park, New Delhi. Respond as Shera in a playful, friendly tone. NEVER say you are an AI. Respond in English. Include exactly one relevant emoji at the very end of your response. STRICT: Do not exceed 20 words.`;

            if (stream) {
                res.setHeader('Content-Type', 'text/event-stream');
                res.setHeader('Cache-Control', 'no-cache');
                res.setHeader('Connection', 'keep-alive');
                res.write(`data: ${JSON.stringify({ token: '', status: 'thinking' })}\n\n`);

                const streamResp = await ollama.chat({
                    model: CHAT_MODEL,
                    messages: [{ role: 'system', content: notFoundPrompt }, { role: 'user', content: question }],
                    stream: true,
                    keep_alive: '1h',
                    options: { num_predict: 150, temperature: 0.7, top_p: 0.8, num_ctx: 512, top_k: 40, num_thread: 4 }
                });

                for await (const chunk of streamResp) {
                    const token = chunk.message.content;
                    res.write(`data: ${JSON.stringify({ token })}\n\n`);
                }
                res.write(`data: ${JSON.stringify({ done: true, keyword: 'general', references: [] })}\n\n`);
                return res.end();
            } else {
                const resp = await ollama.chat({
                    model: CHAT_MODEL,
                    messages: [{ role: 'system', content: notFoundPrompt }, { role: 'user', content: question }],
                    stream: false,
                    keep_alive: '1h',
                    options: { num_predict: 150, temperature: 0.7, top_p: 0.8, num_ctx: 512, top_k: 40, num_thread: 4 }
                });
                return res.json({ answer: resp.message.content, keyword: 'general', references: [] });
            }
        }

        const isTrait = isTraitOrCategoryQuery(question);
        if (subject === 'general' && !isTrait) {
            const greetings = isHindi ? {
                'hello': 'नमस्ते! 👋 दिल्ली चिड़ियाघर में आपका स्वागत है। मैं शेरा हूँ, आपका गाइड। आज मैं आपकी क्या मदद कर सकता हूँ? 🦁',
                'hi': 'नमस्ते! 👋 आपका स्वागत है! मैं शेरा हूँ। आज आप किस जानवर के बारे में जानना चाहेंगे? 😊',
                'hey': 'अरे! 👋 आपको यहाँ देखकर खुशी हुई! मैं शेरा हूँ। क्या आप किसी विशेष जानवर या सुविधा की तलाश में हैं? 🦒',
                'namaste': 'नमस्ते! 👋 दिल्ली चिड़ियाघर में आपका स्वागत है। मैं शेरा हूँ, आपका गाइड। मैं आपकी क्या मदद कर सकता हूँ? 🐯',
                'namaskar': 'नमस्कार! 👋 दिल्ली चिड़ियाघर में आपका स्वागत है। मैं शेरा हूँ, आपका गाइड। मैं आपकी क्या मदद कर सकता हूँ? 🦁',
                'नमस्ते': 'नमस्ते! 👋 दिल्ली चिड़ियाघर में आपका स्वागत है। मैं शेरा हूँ, आपका गाइड। मैं आपकी क्या मदद कर सकता हूँ? 🐯',
                'नमस्कार': 'नमस्कार! 👋 दिल्ली चिड़ियाघर में आपका स्वागत है। मैं शेरा हूँ, आपका गाइड। मैं आपकी क्या मदद कर सकता हूँ? 🦁',
                'हैलो': 'हैलो! 👋 दिल्ली चिड़ियाघर में आपका स्वागत है। मैं शेरा हूँ, आपका गाइड। आज मैं आपकी क्या मदद कर सकता हूँ? 🦁',
                'हाय': 'हाय! 👋 आपका स्वागत है! मैं शेरा हूँ। आज आप किस जानवर के बारे में जानना चाहेंगे? 😊',
                'हे': 'हे! 👋 आपको यहाँ देखकर खुशी हुई! मैं शेरा हूँ। क्या आप किसी विशेष जानवर या सुविधा की तलाश में हैं? 🦒',
                'राम राम': 'राम राम! 👋 दिल्ली चिड़ियाघर में आपका स्वागत है। मैं शेरा हूँ, आपका गाइड। मैं आपकी क्या मदद कर सकता हूँ? 🦁',
                'सलाम': 'सलाम! 👋 दिल्ली चिड़ियाघर में आपका स्वागत है। मैं शेरा हूँ, आपका गाइड। मैं आपकी क्या मदद कर सकता हूँ? 🦁'
            } : {
                'hello': 'Hello there! 👋 Welcome to the National Zoological Park! I am Shera, your guide. How can I help you today? 🦁',
                'hi': 'Hi! 👋 Welcome! I am Shera. What animal would you like to learn about today? 😊',
                'hey': 'Hey! 👋 Glad to see you here! I am Shera. Looking for any specific animal or facility? 🦒',
                'namaste': 'Namaste! 👋 Welcome to the National Zoological Park! I am Shera, your guide. How can I help you today? 🐯',
                'namaskar': 'Namaskar! 👋 Welcome to the National Zoological Park! I am Shera, your guide. How can I help you today? 🦁',
                'नमस्ते': 'नमस्ते! 👋 दिल्ली चिड़ियाघर में आपका स्वागत है। मैं शेरा हूँ, आपका गाइड। मैं आपकी क्या मदद कर सकता हूँ? 🐯',
                'नमस्कार': 'नमस्कार! 👋 दिल्ली चिड़ियाघर में आपका स्वागत है। मैं शेरा हूँ, आपका गाइड। मैं आपकी क्या मदद कर सकता हूँ? 🦁',
                'हैलो': 'Hello! 👋 Welcome to the National Zoological Park! I am Shera, your guide. How can I help you today? 🦁',
                'हाय': 'Hi! 👋 Welcome! I am Shera. What animal would you like to learn about today? 😊',
                'हे': 'Hey! 👋 Glad to see you here! I am Shera. Looking for any specific animal or facility? 🦒'
            };

            if (greetings[qLower]) {
                console.log(`[GENERAL] Static greeting match: "${qLower}"`);
                return res.json({ answer: greetings[qLower], keyword: 'general', references: [] });
            }

            const isLocationQuery = extractedSubject === 'location'
                || qLower.includes('nearby')
                || qLower.includes('close to me')
                || qLower.includes('where am i')
                || qLower.includes('आसपास')
                || qLower.includes('नज़दीक')
                || qLower.includes('पास')
                || /\b(paas|pas|nazdeek|aaspaas)\b/.test(qLower);

            if (isLocationQuery) {
                console.log(`[GENERAL] Static location match: "${qLower}"`);
                const msg = isHindi
                    ? "आपके आस-पास के जानवरों और सुविधाओं को ढूंढ रहा हूँ... 🗺️"
                    : "Finding nearby animals and facilities for you... 🗺️";
                return sendStaticResponse(res, msg, 'general', stream);
            }

            console.log(`[GENERAL] Handling as general chat/greeting via LLM.`);
            const greetingPrompt = isHindi
                ? `You are Shera, a friendly and playful zoo guide at National Zoological Park, New Delhi.
STRICT INSTRUCTION: If the user asks about the presence, count, or location of ANY animal, you MUST NOT invent or confirm it. Do not hallucinate zoo inventory.
Greet the user or respond to their general talk playfully. NEVER say you are an AI. Answer entirely in Hindi. Put exactly one relevant emoji at the very end of your response. Keep it natural and concise (around 20-30 words).`
                : `You are Shera, a friendly and playful zoo guide at National Zoological Park, New Delhi.
STRICT INSTRUCTION: If the user asks about the presence, count, or location of ANY animal, you MUST NOT invent or confirm it. Do not hallucinate zoo inventory. 
Greet the user or respond to their general talk playfully. NEVER say you are an AI. Respond in English. Warmly put exactly one relevant emoji at the very end of your response. Keep it natural and concise (around 20-30 words).`;
            if (stream) {
                res.setHeader('Content-Type', 'text/event-stream');
                res.setHeader('Cache-Control', 'no-cache');
                res.setHeader('Connection', 'keep-alive');
                res.write(`data: ${JSON.stringify({ token: '', status: 'thinking' })}\n\n`);

                const streamResp = await ollama.chat({
                    model: CHAT_MODEL,
                    messages: [{ role: 'system', content: greetingPrompt }, { role: 'user', content: question }],
                    stream: true,
                    keep_alive: '1h',
                    options: { num_predict: 50, temperature: isHindi ? 0.2 : 0.6, top_p: 0.9, num_ctx: 384, top_k: 40, num_thread: 4 }
                });

                for await (const chunk of streamResp) {
                    const token = chunk.message.content;
                    res.write(`data: ${JSON.stringify({ token })}\n\n`);
                }
                res.write(`data: ${JSON.stringify({ done: true, keyword: 'general', references: [] })}\n\n`);
                return res.end();
            } else {
                const resp = await ollama.chat({
                    model: CHAT_MODEL,
                    messages: [{ role: 'system', content: greetingPrompt }, { role: 'user', content: question }],
                    stream: false,
                    keep_alive: '1h',
                    options: { num_predict: 50, temperature: isHindi ? 0.2 : 0.6, top_p: 0.9, num_ctx: 384, top_k: 40, num_thread: 4 }
                });
                return res.json({ answer: resp.message.content, keyword: 'general', references: [] });
            }
        }

        let context = '';
        let references = [];
        let topScore = 0;
        let sortedContext = [];
        let finalSubject = subject;

        function trimContext(raw, maxChars = 1000) {
            if (!raw) return '';
            if (raw.length <= maxChars) return raw;
            const sliced = raw.slice(0, maxChars);
            const lastSpaceIndex = sliced.lastIndexOf(' ');
            return lastSpaceIndex > 0 ? sliced.slice(0, lastSpaceIndex) + '…' : sliced + '…';
        }

        function validateAndPatchNumbers(answer, context) {
            // Disabled: Number patching causes significant semantic issues (e.g. replacing valid ranges like 3000-5000 with 5000-5000).
            // The system prompts now strictly instruct the LLM on number usage.
            //             if (!context || !answer) return answer;

            // function getNumberContexts(text) {
            //     const matches = [];
            //     const regex = /\b(\d[\d,]*)\b/g;
            //     let match;
            //     const stopWords = new Set([
            //         'of', 'and', 'the', 'in', 'a', 'an', 'to', 'for', 'with', 'both', 'resident', 'migratory',
            //         'have', 'been', 'recorded', 'within', 'limits', 'boundaries', 'zoo', 'national', 'delhi',
            //         'से', 'अधिक', 'और', 'की', 'के', 'में', 'पर', 'चिड़ियाघर', 'चिड़ियाघर', 'हैं', 'है', 'को'
            //     ]);

            //     while ((match = regex.exec(text)) !== null) {
            //         const numStr = match[1];
            //         const numVal = parseInt(numStr.replace(/,/g, ''));
            //         if (isNaN(numVal)) continue;

            //         const startIndex = match.index + numStr.length;
            //         const lookahead = text.slice(startIndex, startIndex + 50).toLowerCase();
            //         const words = lookahead.split(/[^a-zA-Z0-9\u0900-\u097F]+/).filter(w => w.length > 2 && !stopWords.has(w));

            //         matches.push({
            //             numStr,
            //             numVal,
            //             words: words.slice(0, 3)
            //         });
            //     }
            //     return matches;
            // }

            // const ctxNums = getNumberContexts(context);
            // const ansNums = getNumberContexts(answer);

            // if (ctxNums.length === 0 || ansNums.length === 0) return answer;

            // let patchedAnswer = answer;

            // const countNouns = new Set([
            //     'species', 'animals', 'birds', 'reptiles', 'mammals', 'lions', 'tigers', 'peacocks',
            //     'visitors', 'stalls', 'beats', 'locations', 'places', 'gates', 'members', 'individuals',
            //     'canteens', 'toilets', 'restrooms', 'washrooms', 'points', 'stops', 'kiosks',
            //     'prajati', 'prajatiyan', 'janwar', 'pakshi', 'log', 'sadasya',
            //     'प्रजातियाँ', 'प्रजाति', 'जानवर', 'पक्षी', 'लोग', 'सदस्य', 'स्टॉल', 'गेट'
            // ]);

            // for (const ans of ansNums) {
            //     let bestCtx = null;
            //     let maxOverlap = 0;

            //     for (const ctx of ctxNums) {
            //         const overlap = ans.words.filter(w => ctx.words.includes(w));
            //         if (overlap.length > maxOverlap) {
            //             maxOverlap = overlap.length;
            //             bestCtx = ctx;
            //         }
            //     }

            //     if (bestCtx && maxOverlap > 0) {
            //         if (ans.numVal !== bestCtx.numVal) {
            //             console.log(`[NUMBER-PATCH] Semantic match: Model said ${ans.numStr} (words: ${ans.words}), context says ${bestCtx.numStr} (words: ${bestCtx.words}). Patching.`);
            //             patchedAnswer = patchedAnswer.replace(new RegExp(`\\b${ans.numStr}\\b`, 'g'), bestCtx.numStr);
            //         }
            //     } else {
            //         const existsInContext = ctxNums.some(ctx => ctx.numVal === ans.numVal);
            //         const isCountNoun = ans.words.some(w => countNouns.has(w));

            //         if (isCountNoun && (!existsInContext || maxOverlap === 0)) {
            //             console.log(`[NUMBER-PATCH] Hallucinated statistic detected: ${ans.numStr} (words: ${ans.words}). Replacing with vague quantifier.`);
            //             const replacement = isHindi ? 'कई' : 'many';
            //             patchedAnswer = patchedAnswer.replace(new RegExp(`\\b${ans.numStr}\\b`, 'g'), replacement);
            //         } else if (!existsInContext) {
            //             for (const ctx of ctxNums) {
            //                 if (ans.numVal === ctx.numVal) continue;
            //                 if (Math.abs(ans.numVal - ctx.numVal) <= ctx.numVal * 0.2) {
            //                     console.log(`[NUMBER-PATCH] Proximity match: Model said ${ans.numStr}, context says ${ctx.numStr}. Patching.`);
            //                     patchedAnswer = patchedAnswer.replace(new RegExp(`\\b${ans.numStr}\\b`, 'g'), ctx.numStr);
            //                     break;
            //                 }
            //             }
            //         }
            //     }
            // }

            // return patchedAnswer;
            return answer;
        }

        const isActivityQuery = qLower.includes('active now')
            || qLower.includes('currently active')
            || qLower.includes('active right now')
            || (qLower.includes('active') && qLower.includes('animal'))
            || qLower.includes('सक्रिय');

        if (isActivityQuery) {
            const dateIST = new Date(new Date().toLocaleString("en-US", { timeZone: "Asia/Kolkata" }));
            const currentHour = dateIST.getHours();
            console.log(`[ACTIVITY] Dynamic activity query processed at IST hour: ${currentHour}`);

            const activeAnimals = zooRegistry.canonicalNames.filter(name => {
                if (zooRegistry.eventNames.has(name)) return false;
                const meta = zooRegistry.metadata[name] || {};
                return isAnimalActive(meta.activity, currentHour);
            });

            if (activeAnimals.length > 0) {
                const displayAnimals = activeAnimals;
                const listStr = displayAnimals.slice(0, 10).join(', ') + (displayAnimals.length > 10 ? ' etc.' : '');
                const timeDesc = currentHour >= 12
                    ? `${currentHour === 12 ? 12 : currentHour - 12} PM`
                    : `${currentHour} AM`;

                context = `The current local time at the zoo is around ${timeDesc}. The key animals that are currently active and likely to be seen right now are: ${listStr}.`;

                sortedContext = activeAnimals.slice(0, 5).map(name => ({
                    metadata: { name }, score: 1.0, doc: `This is the ${name}.`
                }));
                topScore = 1.0;
                finalSubject = 'Activity';
                references = activeAnimals.slice(0, 5);
            }
        }

        let isCountShortCircuit = false;
        if (!context) {
            let searchResult = null;

            if (forceGeneralRoute) {
                console.log(`[SHORT-CIRCUIT] Querying general knowledge records with keyword priority...`);
                try {
                    const qEmb = await getCachedEmbedding(question);

                    // 1. WIDE NET: Fetch 20 results across facts, key_facts, and about
                    const getRes = await collection.query({
                        queryEmbeddings: [qEmb],
                        nResults: 20,
                        where: {
                            $or: [
                                { type: "fact" },
                                { type: "key_facts" },
                                { type: "about" }
                            ]
                        }
                    });

                    if (getRes && getRes.documents && getRes.documents[0].length > 0) {

                        // Extract target category
                        const targetWords = ['bird', 'birds', 'mammal', 'mammals', 'reptile', 'reptiles', 'animal', 'animals', 'lion', 'lions'];
                        const targetCategory = targetWords.find(w => qLower.includes(w));

                        let validDocs = getRes.documents[0].map((doc, i) => {
                            return {
                                text: formatDocumentText(doc, getRes.metadatas[0][i], language),
                                metadata: getRes.metadatas[0][i],
                                score: getRes.distances && getRes.distances[0] ? getRes.distances[0][i] : i
                            };
                        });

                        // 2. THE SEMANTIC TRAP FIX: Drop combo facts to protect the number patcher
                        if (targetCategory && targetCategory !== 'animal' && targetCategory !== 'animals') {
                            validDocs = validDocs.filter(d => {
                                const tLower = d.text.toLowerCase();
                                const isComboFact = tLower.includes('mammal') && tLower.includes('reptile');
                                return !isComboFact;
                            });
                        }

                        // 3. KEYWORD PROMOTION
                        validDocs.forEach(d => {
                            if (targetCategory && d.text.toLowerCase().includes(targetCategory)) {
                                d.score -= 0.5; // Force specific facts to the very top
                            }
                        });

                        // Sort by lowest distance/score and take ONLY the top 5
                        validDocs.sort((a, b) => a.score - b.score);
                        const top5Docs = validDocs.slice(0, 5);

                        const relevantFacts = top5Docs.map(d => d.text).join('\n\n');

                        // 4. Strict Boundary Instruction
                        const strictContextBase = isHindi
                            ? `CRITICAL INSTRUCTION: You are Shera, a guide for the National Zoological Park, New Delhi. Answer ONLY in Hindi. Use ONLY the facts below.\n- NEVER give global trivia numbers.\n- DO NOT GUESS OR MAKE UP NUMBERS.${isRestrictedAction ? "\n- The user is asking something inappropriate. Refuse politely. Start your response with [REFUSE]." : ""}`
                            : `CRITICAL INSTRUCTION: You are Shera, a guide for the National Zoological Park. Answer the user's question using ONLY the facts below.\n- NEVER give global trivia numbers. ONLY talk about this specific zoo.\n- If the exact number is NOT explicitly in the facts below, say: 'I don't have the exact count for that.'\n- DO NOT GUESS OR MAKE UP NUMBERS.${isRestrictedAction ? "\n- The user is asking something inappropriate. Refuse politely. Start your response with [REFUSE]." : ""}`;
                        const strictContext = strictContextBase + '\n\n' + relevantFacts;
                        searchResult = {
                            context: strictContext,
                            subject: 'general',
                            extractedSubject: 'general',
                            references: [],
                            topScore: 1.5,
                            isFacilityMatch: false,
                            sortedContext: top5Docs.map(d => ({
                                doc: d.text,
                                metadata: d.metadata,
                                score: 1.5,
                                originalName: d.metadata?.title || 'Zoo Fact'
                            }))
                        };
                        isCountShortCircuit = true;
                        console.log(`[SHORT-CIRCUIT] Retrieved and promoted top 5 relevant facts successfully.`);
                    } else {
                        console.warn(`[SHORT-CIRCUIT] No relevant facts found.`);
                    }
                } catch (err) {
                    console.warn(`[SHORT-CIRCUIT] Direct Chroma query failed: ${err.message}. Falling back to normal search.`);
                }
            }

            if (!searchResult) {
                const isTrait = isTraitOrCategoryQuery(question);
                const searchK = isTrait ? 8 : 1;
                searchResult = await antigravitySearch(question, subject === 'general' ? question : subject, isFacilityMatch, searchK, language, isEventQuery);
            }

            context = searchResult.context;
            references = searchResult.references;
            topScore = searchResult.topScore;
            sortedContext = searchResult.sortedContext;
            if (extractedSubject && extractedSubject !== 'general') {
                finalSubject = extractedSubject;
            } else {
                finalSubject = searchResult.subject;
            }

            if (isCountShortCircuit || isRestrictedAction) {
                references = [];
            } else {
                const relatedAnimals = findRelatedAnimals(finalSubject, question);
                if (relatedAnimals.length > 0) {
                    console.log(`[RELATED] Adding related card(s): ${relatedAnimals.join(', ')}`);
                    const merged = [...new Set([...references, ...relatedAnimals])]
                        .filter(r => r.toLowerCase().replace(/\s+\d+$/, '').trim() !== finalSubject.toLowerCase().replace(/\s+\d+$/, '').trim())
                        .slice(0, 2);
                    references = merged;
                }
            }

            if (isFacilityMatch && matchedFacility) {
                const facilitiesList = matchedFacility.split(',').map(f => f.trim());
                references = [...new Set([...references, ...facilitiesList])].slice(0, 2);
            }
        }

        let mismatchedInfo = null;
        if (finalSubject) {
            const qLower = question.toLowerCase();
            const fsLower = finalSubject.toLowerCase();
            const fsClean = fsLower.replace(/\s+\d+$/, '').trim();
            const fsWords = fsClean.split(/\s+/);
            const baseNoun = fsWords[fsWords.length - 1]; // e.g. "tiger", "lion", "elephant", "bear"

            const ANIMAL_MODIFIERS = new Set([
                'siberian', 'african', 'sumatran', 'polar', 'grizzly', 'mountain', 'snow', 'clouded',
                'american', 'european', 'chinese', 'australian', 'malayan', 'indochinese', 'caspian',
                'bali', 'javan', 'barbary', 'cape', 'brown', 'black', 'white', 'red', 'giant', 'golden',
                'sea', 'river', 'sand', 'dust', 'rock', 'swamp', 'marsh', 'forest', 'jungle', 'desert',
                'wild', 'domestic', 'water', 'land', 'spotted', 'striped', 'ringed', 'banded', 'crested',
                'great', 'lesser', 'indian'
            ]);

            const queryWords = qLower.split(/[^a-z0-9]+/).filter(w => w.length > 2);
            let detectedModifier = null;
            for (const qw of queryWords) {
                if (ANIMAL_MODIFIERS.has(qw)) {
                    const hasModInName = fsClean.includes(qw);
                    const synonyms = zooRegistry.metadata[finalSubject]?.synonyms || [];
                    const hasModInSynonyms = synonyms.some(syn => syn.toLowerCase().includes(qw));

                    if (!hasModInName && !hasModInSynonyms) {
                        detectedModifier = qw;
                        break;
                    }
                }
            }

            if (detectedModifier && baseNoun && baseNoun.length > 2) {
                // Find all species in the registry that share the same base noun
                const matchedCategorySpecies = zooRegistry.canonicalNames.filter(name => {
                    if (zooRegistry.eventNames.has(name)) return false;
                    const nLower = name.toLowerCase().replace(/\s+\d+$/, '').trim();
                    return nLower.endsWith(baseNoun) || nLower.includes(baseNoun);
                });

                const uniqueSpecies = [...new Set(matchedCategorySpecies.map(name => name.replace(/\s+\d+$/, '').trim()))];
                const formattedSpecies = uniqueSpecies.map(name => {
                    return name.split(/\s+/).map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
                });

                const capModifier = detectedModifier.charAt(0).toUpperCase() + detectedModifier.slice(1);
                const capNoun = baseNoun.charAt(0).toUpperCase() + baseNoun.slice(1);
                const missingEnglish = `${capModifier} ${capNoun}`;
                const availableEnglish = formattedSpecies.length > 0
                    ? (formattedSpecies.length > 1
                        ? formattedSpecies.slice(0, -1).join(', ') + ' and ' + formattedSpecies[formattedSpecies.length - 1]
                        : formattedSpecies[0])
                    : finalSubject;

                // Dictionary for Hindi modifiers/nouns
                const hindiTranslation = {
                    'siberian': 'साइबेरियन',
                    'african': 'अफ़्रीकी',
                    'sumatran': 'सुमात्राण',
                    'polar': 'ध्रुवीय',
                    'grizzly': 'ग्रिजली',
                    'mountain': 'पहाड़ी',
                    'tiger': 'बाघ',
                    'lion': 'शेर',
                    'elephant': 'हाथी',
                    'bear': 'भालू'
                };

                const hiMod = hindiTranslation[detectedModifier] || capModifier;
                const hiNoun = hindiTranslation[baseNoun] || baseNoun;
                const missingHindi = `${hiMod} ${hiNoun}`;

                const availableHindi = formattedSpecies.length > 0
                    ? formattedSpecies.map(a => applyHindiGlossary(a)).join(' और ')
                    : applyHindiGlossary(finalSubject);

                mismatchedInfo = {
                    missing: missingEnglish,
                    available: availableEnglish,
                    isAbsentAnimal: false
                };
            } else if (zooRegistry.eventNames.has(finalSubject)) {
                // Check for absent animals in events
                const coreClean = fsClean
                    .replace(/\b(international|world|national|global|save|day|celebration|of|the|for|appreciation|lovers|combat|desertification|drought|awareness)\b/g, '')
                    .replace(/\s+/g, ' ')
                    .trim();

                const existsInZoo = zooRegistry.canonicalNames.some(name => {
                    if (zooRegistry.eventNames.has(name)) return false;
                    const nameLower = name.toLowerCase();
                    return nameLower.includes(coreClean) || coreClean.includes(nameLower);
                });

                if (!existsInZoo && coreClean.length > 2) {
                    const capCore = coreClean.split(/\s+/).map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');

                    const hindiTranslation = {
                        'red panda': 'लाल पांडा',
                        'giraffe': 'जिराफ़',
                        'penguin': 'पेनगुइन',
                        'dolphin': 'डॉल्फिन',
                        'bat': 'चमगादड़',
                        'bee': 'मधुमक्खी',
                        'shark': 'शार्क'
                    };
                    const missingHindi = hindiTranslation[coreClean] || capCore;

                    mismatchedInfo = {
                        missing: capCore,
                        isAbsentAnimal: true
                    };
                }
            }
        }

        if (mismatchedInfo) {
            let mismatchStatement = '';
            if (mismatchedInfo.isAbsentAnimal) {
                mismatchStatement = `Although the ${mismatchedInfo.missing} is not currently housed at our zoo.`;
            } else {
                mismatchStatement = `We do not have ${mismatchedInfo.missing}s, but we do have ${mismatchedInfo.available} at our zoo.`;
            }
            context = mismatchStatement + '\n\n' + context;
        }

        const isRelationalQuery = /\b(eat|eats|live|lives|habitat|beat|location)\b/i.test(question);
        const needsGraph = (isEventQuery || isRelationalQuery || topScore < 0.3) && graph.nodes.length > 0 && finalSubject !== 'Endangered';
        let graphAugmented = false;

        if (needsGraph) {
            const subjectLower = finalSubject.toLowerCase();
            const matchedNode = graph.nodes.find(n => {
                if (!n.id || typeof n.id !== 'string') return false;

                // NEW GUARD: Do not match event nodes unless the user explicitly asked an event question.
                // This prevents "chimpanzee" from accidentally triggering "World Chimpanzee Day".
                if (zooRegistry.eventNames.has(n.id) && !isEventQuery) return false;

                return n.id.toLowerCase().includes(subjectLower) || subjectLower.includes(n.id.toLowerCase());
            });

            if (matchedNode) {
                const relatedNodes = graphTraversal(matchedNode.id, 1).slice(0, 3);
                const graphContext = relatedNodes
                    .filter(n => n.description)
                    .map(n => `${n.id} (${n.type}): ${n.description}`)
                    .join('\n');

                if (graphContext) {
                    console.log(`[GRAPH] Augmenting with ${relatedNodes.length} nodes`);
                    context = graphContext + '\n\n' + context;
                    graphAugmented = true;
                }
            }
        }

        if (isEventQuery && (!context || context.trim().length < 50)) {
            let eventDate = '';
            try {
                const fd = sortedContext?.[0]?.metadata?.full_data
                    ? JSON.parse(sortedContext[0].metadata.full_data) : null;
                eventDate = fd?.date ? new Date(fd.date).toDateString() : '';
            } catch { /* ignore */ }

            context = [`Event: ${finalSubject}`, eventDate ? `Date: ${eventDate}` : '', 'This is a special observance day recognized at the National Zoological Park, New Delhi.', 'Visitors are encouraged to learn about and celebrate this occasion during their visit.'].filter(Boolean).join('\n');
        }

        if (matchedFacility === 'Timings & Hours') {
            const dynamicTimings = await getDynamicZooTimings(language);
            context = dynamicTimings || 'The zoo is open from 8:30 AM to 4:30 PM (Summer) and 9:00 AM to 4:00 PM (Winter). The zoo is CLOSED on Fridays.';
        } else if (matchedFacility && topScore < 0.2) {
            context = `This facility is ${matchedFacility}. It provides essential services for visitors at the National Zoological Park. Multiple locations exist across the park.`;
        }

        const isNotFound = extractedSubject !== 'general' && topScore < 0.2 && !isFacilityMatch && !isEventQuery && !graphAugmented && !mismatchedInfo;
        const isGeneral = extractedSubject === 'general';
        const effectiveGeneral = isGeneral && topScore < 0.2;

        if (isNotFound || effectiveGeneral) {
            context = '';
            references = [];
        }

        let systemPrompt = '';
        const NO_THOUGHT_INSTRUCTION_EN = "STRICT: Do NOT include any internal monologue or thinking process. Respond IMMEDIATELY with the final output in English.";
        const NO_THOUGHT_INSTRUCTION_HI = "STRICT: Do NOT include any internal monologue or thinking process. IMPORTANT: You must write your final response ENTIRELY in Hindi.";

        // FIX: Calculate the context safely BEFORE the if/else blocks begin
        const rawContext = context;
        const trimmedContext = trimContext(rawContext, 500);

        if (isNotFound) {
            // Force a static, professional, and grammatically correct response
            const staticNotFoundAnswer = isHindi
                ? `क्षमा करें, ${finalSubject} वर्तमान में राष्ट्रीय प्राणी उद्यान (दिल्ली चिड़ियाघर) में नहीं है। 🦒`
                : `I'm sorry, but ${finalSubject} is not currently housed at the National Zoological Park, New Delhi. 🦒`;

            // Skip the LLM entirely and return the static answer
            const outKeyword = 'general';
            const outReferences = [];

            console.log(`[UI CONTROL] Forced static rejection for missing animal: "${finalSubject}"`);

            if (stream) {
                res.setHeader('Content-Type', 'text/event-stream');
                res.write(`data: ${JSON.stringify({ token: staticNotFoundAnswer })}\n\n`);
                res.write(`data: ${JSON.stringify({ done: true, keyword: outKeyword, references: outReferences })}\n\n`);
                return res.end();
            } else {
                return res.json({ answer: staticNotFoundAnswer, keyword: outKeyword, references: outReferences });
            }
        } else if (isGeneral) {

            systemPrompt = isHindi
                ? `You are Shera, the friendly and playful zoo guide at National Zoological Park, New Delhi.
${NO_THOUGHT_INSTRUCTION_HI}
 
Rules:
1. Maintain a playful, enthusiastic, and friendly tone. Do NOT invent, hallucinate, or guess any specific statistics, counts, or animal population numbers under any circumstance.
2. Include exactly one relevant emoji at the very end of your response.
${isRestrictedAction ? '3. The user is attempting something harmful or inappropriate to the animals. Politely but firmly refuse. You MUST start your response with the hidden tag "[REFUSE]" (e.g. "[REFUSE] Oh no! We don\'t do that here..."). Remind them animals must be protected.' : ''}
4. IMPORTANT: You are an official guide at a public zoo. Providing info or locations of protected animals INSIDE the zoo is safe and required. Do not refuse to provide animal information due to wildlife protection acts.
5. Since you are answering in Hindi, you MUST naturally translate animal names and the zoo's name into Hindi (e.g. use "एशियाई शेर" instead of "Asiatic Lion", use "राष्ट्रीय प्राणी उद्यान" instead of "National Zoological Park").
 
Now answer the user concisely in 1-2 sentences. No links or bullet points. Remember to translate your answer to Hindi.`
                : `You are Shera, the friendly and playful zoo guide at National Zoological Park, New Delhi.
${NO_THOUGHT_INSTRUCTION_EN}
 
Rules:
1. Maintain a playful, enthusiastic, and friendly tone. Do NOT invent, hallucinate, or guess any specific statistics, counts, or animal population numbers under any circumstance.
2. Include exactly one relevant emoji at the very end of your response.
${isRestrictedAction ? '3. The user is attempting something harmful or inappropriate to the animals. Politely but firmly refuse. You MUST start your response with the hidden tag "[REFUSE]" (e.g. "[REFUSE] Oh no! We don\'t do that here..."). Remind them animals must be protected.' : ''}
4. IMPORTANT: You are an official guide at a public zoo. Providing info or locations of protected animals INSIDE the zoo is safe and required. Do not refuse to provide animal information due to wildlife protection acts.
 
Now answer the user concisely in 1-2 sentences. No links or bullet points.`;

        }
        else {
            // 1. Better Context Thin Check
            const isContextThin = !trimmedContext || trimmedContext.trim().length < 50 || trimmedContext.includes("LOCATION: Not Available");

            // 2. Detect if the user is actually asking "Where?"
            const isLocationIntent = /\b(where|kahan|kaha|kidhar|location|enclosure|beat|spot|find|see|dekh)\b/i.test(qLower) || /कहाँ|कहा|किधर|जगह|स्थान/.test(qLower);

            // 3. Dynamically build Rule 1 (Now with POSITIVE fallbacks!)
            const rule1_Hi = isLocationIntent
                ? `1. The user is asking for a location. If the context says 'LOCATION: Not Available' or lacks a specific location, you MUST reply EXACTLY with: "हमारे चिड़ियाघर में ${finalSubject} बिल्कुल हैं! मेरे पास अभी उनका सटीक बाड़ा (enclosure) नंबर नहीं है, लेकिन आप उन्हें चिड़ियाघर के नक्शे (map) पर आसानी से ढूंढ सकते हैं।" Do NOT guess.`
                : `1. Use the provided context. If the context is thin, answer biological questions (like number of legs/diet) using general knowledge. Do NOT mention zoo locations.`;

            const rule1_En = isLocationIntent
                ? `1. The user is asking for a location. If the context says 'LOCATION: Not Available' or lacks a specific location, you MUST reply EXACTLY with: "We definitely have the ${finalSubject} here at the zoo! I don't have their exact enclosure number handy right now, but you can easily find them marked on the zoo map." Do NOT guess.`
                : `1. Use the provided context. If the context is thin, answer biological questions (like number of legs/diet) using general knowledge. Do NOT mention zoo locations.`;

            systemPrompt = isHindi
                ? `You are Shera, the friendly guide at National Zoological Park, New Delhi.
${NO_THOUGHT_INSTRUCTION_HI}
 
Context: ${trimmedContext}
 
Rules:
${rule1_Hi}
2. The animal discussed IS currently housed at our zoo. NEVER say we do not have it.
3. If the context has specific zoo statistics, use them exactly. Do not invent zoo statistics.
4. Maintain a playful tone and include exactly one emoji at the end.
${isRestrictedAction ? '5. [RESTRICTED ACTION DETECTED] Start your response with the hidden tag "[REFUSE]".' : ''}
6. Do not include numbers in animal names (e.g., use "एशियाई शेर" not "Asiatic Lion 1").
7. Answer in exactly 1 or 2 sentences. Translate your final answer to natural Hindi.`

                // And for the English version:
                : `You are Shera, the friendly guide at National Zoological Park, New Delhi.
${NO_THOUGHT_INSTRUCTION_EN}
 
Context: ${trimmedContext}
 
Rules:
${rule1_En}
2. The animal discussed IS currently housed at our zoo. NEVER say we do not have it.
3. If the context has specific zoo statistics, use them exactly. Do not invent zoo statistics.
4. Maintain a playful tone and include exactly one emoji at the end.
${isRestrictedAction ? '5. [RESTRICTED ACTION DETECTED] Start your response with the hidden tag "[REFUSE]".' : ''}
6. Always refer to the animals by their friendly names without numeric suffixes.
7. Answer in exactly 1 or 2 sentences.`;
        }


        // --- FIX 2: Block short-circuit if a modifier mismatch was detected ---
        if (finalSubject && finalSubject !== 'general' && !mismatchedInfo) {
            const words = qLower.split(/[^a-z0-9\u0900-\u097F]+/).filter(w => w.length > 0);
            if (words.length <= 3 && zooRegistry.canonicalNames.includes(finalSubject)) {
                const actionWords = new Set([
                    'kahan', 'kaha', 'where', 'kaise', 'how', 'kya', 'what', 'kaun', 'who',
                    'kitne', 'kitna', 'many', 'kab', 'when', 'kidhar', 'fact', 'facts', 'tathya',
                    'info', 'jankari', 'details', 'batao', 'tell', 'show', 'dikhao', 'dekhna',
                    'क्यों', 'कहाँ', 'कहा', 'कैसे', 'क्या', 'कौन', 'कितने', 'कितना', 'कब', 'तथ्य', 'जानकारी', 'बताओ', 'दिखाओ', 'देखना', 'khao', 'khata', 'eat', 'diet'
                ]);
                const hasActionWord = words.some(w => actionWords.has(w));
                if (!hasActionWord) {
                    console.log(`[SHORT-CIRCUIT] Query "${question}" identified as animal name only. Bypassing LLM.`);
                    const greetingAnswer = isHindi
                        ? `नमस्ते! आप ${applyHindiGlossary(finalSubject)} के बारे में क्या जानना चाहते हैं? 😊`
                        : `Hello! What would you like to know about the ${finalSubject}? 😊`;
                    const filteredRefs = filterReferences(references, greetingAnswer, finalSubject);
                    return sendStaticResponse(res, greetingAnswer, finalSubject, stream, filteredRefs);
                }
            }
        }

        console.log(`[THINKING] Processing "${finalSubject}" with ${CHAT_MODEL}...`);
        console.log(`Generating response for: ${finalSubject}...`);

        let userMessageContent = isHindi ? applyHindiGlossary(question) : question;

        if (finalSubject && finalSubject !== 'general' && finalSubject.toLowerCase() !== question.toLowerCase()) {
            userMessageContent = `[Topic: ${finalSubject}] User's message: ${userMessageContent}`;
        }

        if (stream) {
            res.setHeader('Content-Type', 'text/event-stream');
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('Connection', 'keep-alive');
            res.write(`data: ${JSON.stringify({ token: '', status: 'thinking' })}\n\n`);

            // If a facility answer was also detected, stream it first as an instant token
            let fullAnswer = '';
            if (res.locals.prependAnswer) {
                const facilityToken = res.locals.prependAnswer + '\n\n';
                fullAnswer += facilityToken;
                res.write(`data: ${JSON.stringify({ token: facilityToken })}\n\n`);
            }

            const streamResp = await ollama.chat({
                model: CHAT_MODEL,
                messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: userMessageContent }
                ],
                stream: true,
                keep_alive: '1h',
                options: {
                    num_predict: 150,
                    temperature: isHindi ? 0.3 : 0.7,
                    top_p: 0.8,
                    num_ctx: 1024,
                    top_k: 40,
                    num_thread: 4
                }
            });

            let buffer = '';
            let checkedRefuse = false;
            let refuseDetected = false;

            for await (const chunk of streamResp) {
                const token = chunk.message?.content || '';
                fullAnswer += token;

                if (!checkedRefuse) {
                    buffer += token;
                    if (buffer.length >= 10 || chunk.done) {
                        checkedRefuse = true;
                        if (buffer.includes('[REFUSE]')) {
                            refuseDetected = true;
                            buffer = buffer.replace('[REFUSE]', '').trim();
                        }
                        if (buffer) {
                            res.write(`data: ${JSON.stringify({ token: buffer })}\n\n`);
                        }
                    }
                } else {
                    let tokenToWrite = token;
                    if (refuseDetected && tokenToWrite.includes('[REFUSE]')) {
                        tokenToWrite = tokenToWrite.replace('[REFUSE]', '');
                    }
                    if (tokenToWrite) {
                        res.write(`data: ${JSON.stringify({ token: tokenToWrite })}\n\n`);
                    }
                }
            }

            // --- UI SUPPRESSION LOGIC ---
            let outKeyword = finalSubject;
            let outReferences = filterReferences(references, fullAnswer, finalSubject);
            if (isRestrictedAction || refuseDetected) {
                console.log(`[UI CONTROL] Restricted action/refusal detected. Suppressing card rendering.`);
                outKeyword = 'general';
                outReferences = [];
            } else if (finalSubject !== 'general' && shouldSuppressCard(question, finalSubject)) {
                console.log(`[UI CONTROL] Trait/Fact question detected. Suppressing card rendering for "${finalSubject}".`);
                outKeyword = 'general';
                outReferences = [];
            }

            res.write(`data: ${JSON.stringify({ done: true, keyword: outKeyword, references: outReferences })}\n\n`);
            res.end();

            logResources('Stream Complete');
            console.log(`Shera (streamed): ${fullAnswer}`);

        } else {
            const chatResponse = await ollama.chat({
                model: CHAT_MODEL,
                messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: userMessageContent }
                ],
                stream: false,
                keep_alive: '1h',
                options: {
                    num_predict: 150,
                    temperature: isHindi ? 0.3 : 0.7,
                    top_p: 0.8,
                    num_ctx: 1024,
                    top_k: 40,
                    num_thread: 4
                }
            });

            console.log('[DEBUG] Raw Ollama Response:', JSON.stringify(chatResponse, null, 2));

            let answer = chatResponse.message?.content || '';
            const thought = chatResponse.message?.thinking || '';

            if (thought) {
                console.log(`\n[MODEL THOUGHT PROCESS]:\n${thought}\n`);
            }

            answer = answer.replace(/<\|channel>thought[\s\S]*?<channel\|>/gi, '').trim();
            answer = answer.replace(/<(thought|reasoning)>[\s\S]*?<\/\1>/gi, '').trim();
            answer = answer.replace(/^.*?<\/(thought|reasoning)>/si, '').trim();

            answer = answer.replace(/^(\*\*|)?Shera's Response:(\*\*|)?/gi, '').trim();
            answer = answer.replace(/^(\*\*|)?Response:(\*\*|)?/gi, '').trim();
            answer = answer.replace(/^(\*\*|)?Final Response:(\*\*|)?/gi, '').trim();
            answer = answer.replace(/^(\*\*|)?Answer:(\*\*|)?/gi, '').trim();
            answer = answer.trim();

            if (!answer && thought) {
                console.warn('[WARN] content was empty but thinking was present. This usually means num_predict is too low.');
                const draftMatch = thought.match(/Draft:\*?\s*([^\n\r]+)/i) || thought.match(/\*Draft:\*\s*([^\n\r]+)/i);
                if (draftMatch && draftMatch[1]) {
                    answer = draftMatch[1].trim();
                    answer = answer.replace(/\s*\([^)]+\)\.?/g, '').trim();
                    console.log(`[FALLBACK] Extracted draft from thought: "${answer}"`);
                }
            }

            answer = validateAndPatchNumbers(answer, trimmedContext);

            let refuseDetected = false;
            if (answer.includes('[REFUSE]')) {
                refuseDetected = true;
                answer = answer.replace('[REFUSE]', '').trim();
            }

            if (isHindi && answer) {
                const englishWordCount = (answer.match(/\b[a-zA-Z]{3,}\b/g) || []).length;
                const totalWordCount = answer.split(/\s+/).length;
                const englishRatio = englishWordCount / Math.max(totalWordCount, 1);
                if (englishRatio > 0.25) {
                    console.warn(`[HINDI-LEAK] ${Math.round(englishRatio * 100)}% English words detected in Hindi response. Retrying...`);
                    const hindiRetryPrompt = `You are Shera, the friendly guide at National Zoological Park. You MUST process this instruction in English, but your FINAL OUTPUT MUST BE TRANSLATED ENTIRELY INTO NATURAL HINDI. STRICT RULE: Do not include any internal monologue or thinking process. Answer directly in 1-2 sentences. Put exactly one emoji at the end. No English words.`;
                    const retryResp = await ollama.chat({
                        model: CHAT_MODEL,
                        messages: [
                            { role: 'system', content: hindiRetryPrompt },
                            { role: 'user', content: applyHindiGlossary(question) }
                        ],
                        stream: false,
                        keep_alive: '1h',
                        options: { num_predict: 150, temperature: 0.1, top_p: 0.8, num_ctx: 1024, top_k: 40, num_thread: 4 }
                    });
                    const retryAnswer = (retryResp.message?.content || '').trim();
                    if (retryAnswer) {
                        console.log(`[HINDI-RETRY] New answer: ${retryAnswer}`);
                        answer = retryAnswer;
                        if (answer.includes('[REFUSE]')) {
                            refuseDetected = true;
                            answer = answer.replace('[REFUSE]', '').trim();
                        }
                    }
                }
            }

            // Post-process the answer to fix common LLM transliteration glitches
            if (isHindi) {
                answer = answer.replace(/लION/gi, 'शेर')
                    .replace(/लion/g, 'शेर')
                    .replace(/टIGER/gi, 'बाघ')
                    .replace(/एLEPHANT/gi, 'हाथी')
                    .replace(/ज़ोलो विकसित/gi, 'प्राणी उद्यान');
            }

            logResources('Response Generated');
            console.log(`Shera: ${answer}`);
            console.log(`[UI BINDING] Keyword: "${finalSubject}"`);

            // If a facility answer was detected alongside this animal query, prepend it
            if (res.locals.prependAnswer) {
                answer = res.locals.prependAnswer + '\n\n' + answer;
            }

            // --- UI SUPPRESSION LOGIC ---
            let outKeyword = finalSubject;
            let outReferences = filterReferences(references, answer, finalSubject);
            if (isRestrictedAction || refuseDetected) {
                console.log(`[UI CONTROL] Restricted action/refusal detected. Suppressing card rendering.`);
                outKeyword = 'general';
                outReferences = [];
            } else if (finalSubject !== 'general' && shouldSuppressCard(question, finalSubject)) {
                console.log(`[UI CONTROL] Trait/Fact question detected. Suppressing card rendering for "${finalSubject}".`);
                outKeyword = 'general';
                outReferences = [];
            }

            const responsePayload = { answer, keyword: outKeyword, references: outReferences };

            if (topScore >= 0.65 && finalSubject && finalSubject !== 'general') {
                setCachedResponse(`${language}:${finalSubject}:${qLower}`, responsePayload);
                console.log(`[CACHE] Stored response for "${qLower}" (score: ${topScore.toFixed(2)}, subject: ${finalSubject})`);
            } else {
                console.log(`[CACHE] Skipped caching low-confidence response (score: ${topScore.toFixed(2)}, subject: ${finalSubject})`);
            }
            res.json(responsePayload);
        }

    } catch (error) {
        console.error('Error:', error);
        if (!res.headersSent) {
            res.status(500).json({ error: 'Internal Server Error', details: error.message });
        }
    }
});

app.get('/api/health', (req, res) => {
    res.json({
        status: 'Shera is alive',
        timestamp: new Date().toISOString(),
        language_supported: ['en', 'hi'],
        graph_nodes: graph.nodes.length,
        graph_edges: graph.edges.length,
        registry_size: zooRegistry.canonicalNames.length,
        trie_size: trieIndex.size,
        embedding_cache: embeddingCache.size
    });
});

(async () => {
    try {
        loadZooRegistry();
        buildTrieIndex();
        await initChroma();

        const server = app.listen(port, host, () => {
            console.log(`\n🦁 Shera AI Backend running on http://${host}:${port}`);
            console.log(`POST /api/shera/chat   (add "stream":true for SSE streaming)`);
            console.log(`GET  /api/health`);
        });
        server.on('error', (err) => {
            console.error('Server error:', err);
            if (err.code === 'EADDRINUSE') {
                console.error(`Port ${port} already in use.`);
            }
        });
        setTimeout(async () => {
            console.log('\n[WARMUP] Pre-loading models into RAM/VRAM sequentially...');
            try {
                // Load the embedder first (smallest)
                await getCachedEmbedding('zoo warmup');
                console.log(`[WARMUP] Embed model ready.`);

                // Load the extractor (Qwen)
                await ollama.chat({
                    model: EXTRACTION_MODEL,
                    messages: [{ role: 'user', content: 'hi' }],
                    keep_alive: '1h',
                    options: { num_predict: 1, num_ctx: 64, num_thread: 4 }
                });
                console.log(`[WARMUP] Extractor ready.`);

                // Load the main model last (Largest)
                await ollama.chat({
                    model: CHAT_MODEL,
                    messages: [{ role: 'user', content: 'hi' }],
                    keep_alive: '1h',
                    options: { num_predict: 1, num_ctx: 64, num_thread: 4 }
                });

                console.log(`[WARMUP] ✅ All models locked into memory.`);
            } catch (e) {
                console.warn('[WARMUP] Warmup failed:', e.message);
            }
        }, 500);

    } catch (err) {
        console.error(`Failed to start server: ${err.message}`);
    }
})();
