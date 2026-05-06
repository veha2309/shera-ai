const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { Ollama } = require('ollama');
const { ChromaClient } = require('chromadb');
const { OllamaEmbeddingFunction } = require('@chroma-core/ollama');
require('dotenv').config();
const { GoogleGenerativeAI } = require("@google/generative-ai");

/**
 * Shera AI - Hybrid Chroma + GraphRAG Backend Server
 * 
 * GEMINI INFRASTRUCTURE:
 * Migrated to Gemini 1.5 Flash for high-speed, high-quality responses
 * on resource-constrained VM environments.
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
const CHAT_MODEL = 'gemini-1.5-flash';
const EXTRACTION_MODEL = 'gemini-1.5-flash';

function logResources(label) {
    const mem = process.memoryUsage();
    console.log(`[RESOURCES] ${label} - RSS: ${(mem.rss / 1024 / 1024).toFixed(2)}MB, Heap: ${(mem.heapUsed / 1024 / 1024).toFixed(2)}MB`);
}

// ─── Clients ──────────────────────────────────────────────────────────────────
const ollama = new Ollama();
const chroma = new ChromaClient({ host: 'localhost', port: 8000 });
const embedder = new OllamaEmbeddingFunction({
    url: 'http://127.0.0.1:11434',
    model: 'nomic-embed-text'
});

// ─── Gemini Client ──────────────────────────────────────────────────────────
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const geminiModel = genAI.getGenerativeModel({ 
    model: "gemini-1.5-flash",
    apiVersion:'v1',
    generationConfig: {
        maxOutputTokens: 100,
        temperature: 0.7,
    }
});

// ─── Embedding Cache ──────────────────────────────────────────────────────────
const embeddingCache = new Map();

// ─── LLM Response Cache ───────────────────────────────────────────────────────
const responseCache = new Map();
const RESPONSE_CACHE_TTL = 10 * 60 * 1000; // 10 minutes

function getCachedResponse(key) {
    const entry = responseCache.get(key);
    if (!entry) return null;
    if (Date.now() - entry.ts > RESPONSE_CACHE_TTL) {
        responseCache.delete(key);
        return null;
    }
    return entry.data;
}

function setCachedResponse(key, data) {
    responseCache.set(key, { data, ts: Date.now() });
}

async function getCachedEmbedding(text) {
    if (embeddingCache.has(text)) return embeddingCache.get(text);
    try {
        const resp = await ollama.embed({ model: EMBED_MODEL, input: text, keep_alive: '1h' });
        const embedding = resp.embeddings[0];
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

const adjacencyMap = {};
for (const edge of graph.edges) {
    if (!edge.source || !edge.target) continue;
    if (!adjacencyMap[edge.source]) adjacencyMap[edge.source] = [];
    if (!adjacencyMap[edge.target]) adjacencyMap[edge.target] = [];
    adjacencyMap[edge.source].push(edge);
    adjacencyMap[edge.target].push(edge);
}

// ─── OPTIMIZATION 2: Graph Traversal Cache ────────────────────────────────────
const graphTraversalCache = new Map();

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

        const node = graph.nodes.find(n => n.id && n.id === id);
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

const zooRegistry = {
    canonicalNames: [],
    lookup: {},
    metadata: {},
    sortedCanonical: [],
    eventNames: new Set()
};

// ─── Priority Overrides ───────────────────────────────────────────────────────
const priorityOverrides = {
    'peacock': 'Indian Peafowl (Leucistic)',
    'peacocks': 'Indian Peafowl (Leucistic)',
    'peafowl': 'Indian Peafowl (Leucistic)',
    'peafowls': 'Indian Peafowl (Leucistic)',
    'peahen': 'Indian Peafowl (Leucistic)',
    'peahens': 'Indian Peafowl (Leucistic)',
    'white peafowl': 'White Peafowl',
    'white peacock': 'White Peafowl',
    'lion tailed monkey': 'Lion Tailed Macaque',
    'lion tailed monkeys': 'Lion Tailed Macaque',
    'lion tailed macaque': 'Lion Tailed Macaque',
    'lion tailed macaques': 'Lion Tailed Macaque',
    'lion': 'Asiatic Lion',
    'lions': 'Asiatic Lion',
    'tiger': 'White Tiger',
    'tigers': 'White Tiger',
    'elephant': 'Indian Elephant',
    'elephants': 'Indian Elephant',
    'snakes': 'Reptile House',
    'snake': 'Common Rat Snake',
    'reptile': 'Reptile House',
    'reptiles': 'Reptile House',
    'deer': 'Spotted Deer',
    'monkey': 'Rhesus Macaque',
    'monkeys': 'Rhesus Macaque',
    'birds': 'Indian Peafowl (Leucistic)',
    'bear': 'Sloth Bear',
    'bears': 'Sloth Bear',
    'पेन': 'Washrooms',
    'पानी': 'Drinking Water',
    'खाना': 'Food & Drinks'
};

// ─── OPTIMIZATION 1: Trie Index ───────────────────────────────────────────────
let trieIndex = new Map();

function buildTrieIndex() {
    const entries = [];
    for (const [phrase, name] of Object.entries(zooRegistry.lookup)) {
        entries.push([phrase.toLowerCase(), name]);
    }
    for (const name of zooRegistry.canonicalNames) {
        const lower = name.toLowerCase();
        if (!entries.some(e => e[0] === lower)) {
            entries.push([lower, name]);
        }
    }
    entries.sort((a, b) => b[0].length - a[0].length);
    trieIndex = new Map(entries);
    console.log(`[TRIE] Built index with ${trieIndex.size} entries.`);
}

function fastExtract(query) {
    const q = query.toLowerCase();
    const foundMatches = [];

    for (const [phrase, name] of trieIndex) {
        const regex = new RegExp(`\\b${phrase.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i');
        if (regex.test(q)) {
            if (!foundMatches.some(m => m.phrase.includes(phrase))) {
                foundMatches.push({ phrase, name });
            }
        }
    }

    const uniqueNames = [...new Set(foundMatches.map(m => m.name))];
    if (uniqueNames.length === 1) {
        const wordCount = q.split(/\s+/).length;
        if (wordCount < 6) return uniqueNames[0];
    }
    return null;
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
                const rName = item.render_name?.en || item.render_name;
                const cName = item.common_name?.en || item.common_name;
                const dName = item.name?.en || item.name;
                const tName = item.title?.en || item.title;

                const primaryNameRaw = rName || cName || dName || tName;
                if (!primaryNameRaw || typeof primaryNameRaw !== 'string' || /^[0-9a-fA-F]{24}$/.test(primaryNameRaw)) continue;

                const primaryName = primaryNameRaw.replace(/\s+\d+$/, '').trim();
                if (primaryName.length <= 2) continue;

                names.add(primaryName);

                if (!aliasMap.has(primaryName)) aliasMap.set(primaryName, new Set());
                [rName, cName, dName, tName].forEach(n => {
                    if (n && typeof n === 'string' && !/^[0-9a-fA-F]{24}$/.test(n)) {
                        const clean = n.replace(/\s+\d+$/, '').trim();
                        if (clean.length > 2) aliasMap.get(primaryName).add(clean);
                    }
                });

                const classification = item.classification?.en || item.classification || '';
                if (classification && classification.length > 2) classifications.add(classification);
                const threatStatus = item.threat_status?.en || item.threat_status || '';

                if (file.includes('calendar') || file.includes('event')) {
                    eventNames.add(primaryName);
                }
                if (!zooRegistry.metadata[primaryName] || classification || threatStatus) {
                    zooRegistry.metadata[primaryName] = { classification, threatStatus };
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

function normalizeToRegistryOrSelf(rawSubject) {
    const words = rawSubject.trim().split(/\s+/);
    for (let len = words.length; len >= 1; len--) {
        const candidate = words.slice(0, len).join(' ');
        const lower = candidate.toLowerCase();
        if (zooRegistry.lookup[lower]) return zooRegistry.lookup[lower];
        if (zooRegistry.canonicalNames.some(n => n.toLowerCase() === lower)) {
            return zooRegistry.canonicalNames.find(n => n.toLowerCase() === lower);
        }
    }
    const sLower = rawSubject.toLowerCase();
    const fuzzyHit = zooRegistry.canonicalNames.find(n => {
        const nl = n.toLowerCase();
        return nl.includes(sLower) || sLower.includes(nl) || (sLower.length > 4 && nl.startsWith(sLower.slice(0, 5)));
    });
    if (fuzzyHit) return fuzzyHit;

    return words[0];
}

// ─── Context Helper ───────────────────────────────────────────────────────────
function optimizeContext(docs, maxLines = 5) {
    return [...new Set(docs)].slice(0, maxLines).join('\n');
}

// ─── Facility Synonym Map ─────────────────────────────────────────────────────
const facilitySynonyms = {
    'Food & Drinks': ['food', 'eat', 'hungry', 'snacks', 'restaurant', 'cafe', 'cafeteria', 'snack', 'khana', 'खाना', 'खानपान', 'भूख', 'कैंटीन', 'canteen'],
    'Drinking Water': ['water', 'drink', 'drinking water', 'thirsty', 'fountain', 'pani', 'पानी', 'प्यास'],
    'Washrooms': ['washroom', 'toilet', 'restroom', 'bathroom', 'shauchalay', 'शौचालय', 'टॉयलेट', 'pee', 'poo'],
    'Buggy Stops': ['buggy', 'shuttle', 'ride', 'cart', 'transport', 'बग्गी'],
    'First Aid': ['first aid', 'medical', 'medicine', 'doctor', 'clinic', 'hospital', 'दवाई', 'अस्पताल'],
    'Counters': ['counter', 'ticket', 'info', 'information', 'help', 'टिकट', 'काउंटर']
};

function detectFacility(text) {
    const t = text.toLowerCase();
    for (const [facility, syns] of Object.entries(facilitySynonyms)) {
        for (const s of syns) {
            const matched = /^[a-z\s]+$/i.test(s)
                ? new RegExp(`\\b${s}\\b`, 'i').test(t)
                : t.includes(s);
            if (matched) return facility;
        }
    }
    return null;
}

function finalizeSubject(subject, qLower, extractedSubject = null) {
    const matchedFacility = detectFacility(qLower) || detectFacility(subject);
    if (matchedFacility) subject = matchedFacility;
    return { subject, extractedSubject: extractedSubject || subject, matchedFacility };
}

async function extractSubject(query) {
    const qLower = query.toLowerCase().trim();

    if (['hello', 'hi', 'hey', 'नमस्ते', 'hello shera', 'hi shera'].includes(qLower) || qLower.length < 3) {
        return finalizeSubject('general', qLower);
    }

    const dayMatch = query.match(/\b(national|world|international|global)\b[\w\s]+\bday\b/i);
    if (dayMatch) {
        const eventName = dayMatch[0].trim().split(/\s+/)
            .map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
            .join(' ');
        console.log(`[EVENT] Day pattern: "${eventName}"`);
        return finalizeSubject(eventName, qLower);
    }

    if (qLower.includes('endangered') || qLower.includes('संकटग्रस्त')
        || qLower.includes('खतरे में') || qLower.includes('conservation')) {
        return finalizeSubject('Endangered', qLower);
    }

    const facilityHit = detectFacility(qLower);
    if (facilityHit) return finalizeSubject(facilityHit, qLower);

    const lowerQuery = query.toLowerCase();
    if (lowerQuery.includes('nearby') || lowerQuery.includes('close to me') || lowerQuery.includes('where am i')) {
        return { subject: 'general', extractedSubject: 'location', matchedFacility: null };
    }
    if (lowerQuery.includes('thank') || lowerQuery.includes('bye') || lowerQuery.includes('goodbye')) {
        return { subject: 'general', extractedSubject: 'greeting', matchedFacility: null };
    }

    const trieHit = fastExtract(query);
    let isEventQuery = /\b(national|international|world|global)\b[\w\s]+\bday\b/i.test(query);

    if (trieHit) {
        if (!zooRegistry.eventNames.has(trieHit) || isEventQuery) {
            console.log(`[TRIE] Fast match: "${trieHit}"`);
            return finalizeSubject(trieHit, qLower);
        }
    }

    // FAST FALLBACK: If query is short, don't waste LLM time on extraction
    const words = qLower.split(/\s+/);
    if (words.length <= 5 || qLower.length < 30) {
        console.log(`[EXTRACTION] Skipping LLM for short query: "${qLower}"`);
        return finalizeSubject(normalizeToRegistryOrSelf(qLower), qLower, qLower);
    }

    let subject = 'general';
    try {
        const prompt = `Extract the primary subject (Animal, Place, Facility, or Event) from the query.
Return ONLY the name in English. BE EXTREMELY BRIEF.
Mappings: "Sher/Shera"->"Asiatic Lion", "Bagh"->"White Tiger", "Hathi"->"Indian Elephant", "Pani"->"Drinking Water".

Query: ${query}`;

        const result = await geminiModel.generateContent(prompt);
        const raw = result.response.text().replace(/[^\w\s]/gi, '').trim();
        subject = raw ? normalizeToRegistryOrSelf(raw) : 'general';
    } catch (e) {
        console.error('[Gemini] Extraction error:', e.message);
    }

    const sLower = subject.toLowerCase();
    if (zooRegistry.lookup[sLower]) subject = zooRegistry.lookup[sLower];

    return finalizeSubject(subject, qLower, subject);
}

async function antigravitySearch(query, subject, isFacilityMatch, topK = 5, language = 'en', isEventQuery = false) {
    console.log(`\n[SEARCH] Query: "${query}" (Lang: ${language}, EventQuery: ${isEventQuery})`);
    console.log(`[ENTITY] Target Subject: "${subject}"`);

    const queryEmbedding = await getCachedEmbedding(subject);
    if (!queryEmbedding) {
        return { context: '', subject: 'general', references: [], topScore: 0 };
    }

    if (subject.toLowerCase() === 'endangered' || subject.toLowerCase() === 'conservation') {
        const endangeredList = zooRegistry.canonicalNames.filter(name => {
            const status = String(zooRegistry.metadata[name]?.threatStatus || '').toLowerCase();
            return status.includes('endangered') || status.includes('vulnerable') || status.includes('threatened');
        });

        if (endangeredList.length > 0) {
            console.log(`[ENTITY] Conservation query. Found ${endangeredList.length} species.`);
            const listStr = endangeredList.slice(0, 15).join(', ');
            return {
                context: `The National Zoological Park, New Delhi is home to many endangered and threatened species.\nSome key endangered/vulnerable animals here are: ${listStr}.\nVisitors are encouraged to learn about their conservation.`,
                sortedContext: endangeredList.slice(0, 5).map(name => ({
                    metadata: { name }, score: 1.0, doc: `This is the ${name}.`
                })),
                topScore: 1.0,
                subject,
                references: endangeredList.slice(0, 5)
            };
        }
    }

    let exactMatch = null;
    try {
        const getRes = await collection.get({ ids: [subject] });
        if (getRes?.ids?.length > 0) {
            const docName = (getRes.metadatas[0]?.name || '').toLowerCase();
            const subjectLower = subject.toLowerCase();
            const subjectWords = subjectLower.split(/\s+/);
            const docWords = docName.split(/\s+/);

            const overlap = subjectWords.filter(w => docWords.includes(w)).length;
            const isGenuineMatch = overlap >= Math.ceil(subjectWords.length * 0.8);

            if (isGenuineMatch) {
                const isEventMatch = getRes.metadatas[0]?.is_event === 'true';
                console.log(`[ENTITY] Exact ID Match: "${subject}" (IsEvent: ${isEventMatch})`);

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
        if (!exactMatch || exactMatch.score < 1.4) {
            results = await collection.query({
                queryEmbeddings: [queryEmbedding],
                nResults: 15
            });
        }
    } catch (e) {
        console.warn('[SEARCH] Vector query failed:', e.message);
    }

    const documents = results.documents?.[0] || [];
    const metadatas = results.metadatas?.[0] || [];
    const distances = results.distances?.[0] || [];

    const scoredContext = exactMatch ? [exactMatch] : [];

    for (let i = 0; i < documents.length; i++) {
        const metadata = metadatas[i] || {};
        const distance = distances[i] || 1.0;
        const docName = (metadata.name || '').toLowerCase().replace(/\d+/g, '').trim();

        const baseSimilarity = Math.max(0, 1.0 - distance);

        let score = baseSimilarity * 0.4;

        if (subject !== 'general') {
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

        let docText = documents[i];
        if (language === 'hi') {
            if (metadata.name_hi || metadata.habitat_hi || metadata.narrative_hi) {
                docText = [
                    `नाम: ${metadata.name_hi || metadata.name}`,
                    metadata.habitat_hi ? `आवास: ${metadata.habitat_hi}` : '',
                    metadata.narrative_hi ? `विवरण: ${metadata.narrative_hi}` : ''
                ].filter(Boolean).join('\n');
            } else if (metadata.full_data) {
                try {
                    const fd = JSON.parse(metadata.full_data);
                    docText = [
                        `नाम: ${fd.common_name?.hi || fd.name?.hi || fd.title?.hi || metadata.name}`,
                        `वैज्ञानिक नाम: ${fd.scientific_name?.hi || fd.scientific_name?.en || ''}`,
                        `श्रेणी: ${fd.category?.hi || fd.category?.en || ''}`,
                        `आवास: ${fd.habitat?.hi || fd.habitat?.en || ''}`,
                        `आहार: ${fd.diet?.hi || fd.diet?.en || ''}`,
                        `स्थान: ${fd.location?.location_name?.hi || fd.location?.location_name?.en || ''}`,
                        `विवरण: ${fd.narrative?.hi || fd.description?.hi || fd.narrative?.en || ''}`,
                        `कहानियाँ: ${fd.story_description?.hi || fd.story_description?.en || ''}`
                    ].filter(s => s && !s.endsWith(': ')).join('\n');
                } catch (e) {
                    console.error('Hindi metadata parse error:', e.message);
                }
            }
        }

        const displayName = metadata.render_name || metadata.common_name || metadata.name;
        scoredContext.push({ doc: docText, score, originalName: displayName, metadata });
    }

    const sortedContext = scoredContext
        .filter(item => item.score > 0)
        .sort((a, b) => b.score - a.score);

    if (scoredContext.length === 0 && subject !== 'general' && zooRegistry.canonicalNames.includes(subject)) {
        console.log(`[SEARCH] Fallback: No vector results, but "${subject}" is in registry. Using metadata.`);
        const meta = zooRegistry.metadata[subject] || {};
        scoredContext.push({
            doc: `The ${subject} is one of the species at National Zoological Park, New Delhi. ${meta.classification ? `It is classified as ${meta.classification}.` : ''}`,
            metadata: { name: subject, ...meta },
            score: 0.5
        });
    }

    const topScore = sortedContext.length > 0 ? sortedContext[0].score : 0;
    const refThreshold = Math.max(0.4, topScore * 0.85);

    let bestMatchName = subject;
    if (sortedContext.length > 0) {
        const topMeta = sortedContext[0].metadata || {};
        const metaName = (topMeta.render_name || topMeta.common_name || topMeta.name || '')
            .replace(/\s+\d+$/, '').trim();
        const isVague = ['general', 'animals', 'birds', 'reptiles', 'mammals', 'fish']
            .includes(subject.toLowerCase());
        if ((isVague || topScore > 1.2) && metaName && !/^[0-9a-fA-F]{24}$/.test(metaName)) {
            bestMatchName = metaName;
        }
    }

    if (topScore < 0.2 && !isFacilityMatch) bestMatchName = 'general';

    let references = sortedContext
        .filter(item => item.score >= refThreshold)
        .map(item => item.originalName?.replace(/\s+\d+$/, '').trim())
        .filter(Boolean);

    const isFacilityName = /Washroom|Drinking Water|Buggy Stops|Food & Drinks|First Aid|Counters/.test(bestMatchName);
    if (isFacilityName) references = [];

    return {
        context: optimizeContext(sortedContext.slice(0, topK).map(i => i.doc)),
        subject: bestMatchName,
        extractedSubject: subject,
        references: [...new Set(references)],
        topScore,
        isFacilityMatch: !!isFacilityMatch,
        sortedContext
    };
}

app.post('/api/shera/chat', async (req, res) => {
    let { question, deepSearch = true, language = 'en', stream = false } = req.body;

    deepSearch = deepSearch === true || deepSearch === 'true';
    logResources('Incoming Chat');

    const isHindi = language === 'hi';
    const qLower = question.toLowerCase().trim();
    console.log(`\n--- Incoming: "${question}" (DeepSearch: ${deepSearch}, Lang: ${language}, Stream: ${stream}) ---`);

    try {
        // ─── Response Cache Check ───────────────────────────────────────────────────
        const cacheKey = `${language}:${qLower}`;
        if (!stream) {
            const cached = getCachedResponse(cacheKey);
            if (cached) {
                console.log(`[CACHE] HIT for "${qLower}"`);
                return res.json(cached);
            }
        }

        const { subject, extractedSubject, matchedFacility } = await extractSubject(question);

        const isFacilityMatch = !!matchedFacility;
        let isEventQuery = /\b(national|international|world|global)\b[\w\s]+\bday\b/i.test(question);

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
                ? `आप शेरा हैं। "${subject}" दिल्ली चिड़ियाघर में नहीं है। शेरा के रूप में उत्तर दें। कभी न कहें कि आप AI हैं। हिंदी में उत्तर दें। सख्त: 20 शब्दों से कम में उत्तर दें।`
                : `You are Shera. "${subject}" is NOT at the National Zoological Park, New Delhi. Respond as Shera. NEVER say you are an AI. Respond in English. STRICT: Do not exceed 20 words.`;

            if (stream) {
                res.setHeader('Content-Type', 'text/event-stream');
                res.setHeader('Cache-Control', 'no-cache');
                res.setHeader('Connection', 'keep-alive');
                res.write(`data: ${JSON.stringify({ token: '', status: 'thinking' })}\n\n`);

                const result = await geminiModel.generateContentStream([notFoundPrompt, question]);
                for await (const chunk of result.stream) {
                    const token = chunk.text();
                    res.write(`data: ${JSON.stringify({ token })}\n\n`);
                }
                res.write(`data: ${JSON.stringify({ done: true, keyword: 'general', references: [] })}\n\n`);
                return res.end();
            } else {
                const result = await geminiModel.generateContent([notFoundPrompt, question]);
                const answer = result.response.text();
                return res.json({ answer, keyword: 'general', references: [] });
            }
        }

        if (subject === 'general') {
            const greetings = {
                'hello': 'Hello there! 👋 Welcome to the National Zoological Park! I am Shera, your guide. How can I help you today? 🦁',
                'hi': 'Hi! 👋 Welcome! I am Shera. What animal would you like to learn about today? 😊',
                'hey': 'Hey! 👋 Glad to see you here! I am Shera. Looking for any specific animal or facility? 🦒',
                'नमस्ते': 'नमस्ते! 👋 दिल्ली चिड़ियाघर में आपका स्वागत है। मैं शेरा हूँ, आपका गाइड। मैं आपकी क्या मदद कर सकता हूँ? 🐯'
            };

            if (greetings[qLower]) {
                console.log(`[GENERAL] Static greeting match: "${qLower}"`);
                return res.json({ answer: greetings[qLower], keyword: 'general', references: [] });
            }

            console.log(`[GENERAL] Handling as general chat/greeting via LLM.`);
            const greetingPrompt = isHindi
                ? 'आप शेरा हैं, एक मिलनसार गाइड। उपयोगकर्ता का स्वागत करें। कभी न कहें कि आप AI हैं। हिंदी में उत्तर दें। सख्त: 20 शब्दों से कम में उत्तर दें।'
                : 'You are Shera, a friendly zoo guide. Greet the user or respond to their general talk. NEVER say you are an AI. Respond in English. STRICT: Do not exceed 20 words.';

            if (stream) {
                res.setHeader('Content-Type', 'text/event-stream');
                res.setHeader('Cache-Control', 'no-cache');
                res.setHeader('Connection', 'keep-alive');
                res.write(`data: ${JSON.stringify({ token: '', status: 'thinking' })}\n\n`);

                const result = await geminiModel.generateContentStream([greetingPrompt, question]);
                for await (const chunk of result.stream) {
                    const token = chunk.text();
                    res.write(`data: ${JSON.stringify({ token })}\n\n`);
                }
                res.write(`data: ${JSON.stringify({ done: true, keyword: 'general', references: [] })}\n\n`);
                return res.end();
            } else {
                const result = await geminiModel.generateContent([greetingPrompt, question]);
                const answer = result.response.text();
                return res.json({ answer, keyword: 'general', references: [] });
            }
        }

        let { context, references, topScore, sortedContext, subject: resolvedSubject } =
            await antigravitySearch(question, subject, isFacilityMatch, 1, language, isEventQuery);

        const finalSubject = resolvedSubject;

        const isRelationalQuery = /\b(eat|eats|live|lives|endangered|habitat|beat|location)\b/i.test(question);
        const needsGraph = (isEventQuery || isRelationalQuery || topScore < 0.3) && graph.nodes.length > 0;
        let graphAugmented = false;

        if (needsGraph) {
            const subjectLower = finalSubject.toLowerCase();
            const matchedNode = graph.nodes.find(n =>
                n.id && typeof n.id === 'string' &&
                (n.id.toLowerCase().includes(subjectLower) || subjectLower.includes(n.id.toLowerCase()))
            );

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

            context = isHindi
                ? [`कार्यक्रम: ${finalSubject}`, eventDate ? `तारीख: ${eventDate}` : '', 'यह नेशनल जूलॉजिकल पार्क, नई दिल्ली में मान्यता प्राप्त एक विशेष दिन है।', 'आगंतुकों को इस अवसर पर जागरूक होने और इसे मनाने के लिए प्रोत्साहित किया जाता है।'].filter(Boolean).join('\n')
                : [`Event: ${finalSubject}`, eventDate ? `Date: ${eventDate}` : '', 'This is a special observance day recognized at the National Zoological Park, New Delhi.', 'Visitors are encouraged to learn about and celebrate this occasion during their visit.'].filter(Boolean).join('\n');
        }

        if (matchedFacility && topScore < 0.2) {
            context = isHindi
                ? `यह सुविधा ${matchedFacility} है। यह नेशनल जूलॉजिकल पार्क में आगंतुकों के लिए आवश्यक सेवाएं प्रदान करती है।`
                : `This facility is ${matchedFacility}. It provides essential services for visitors at the National Zoological Park. Multiple locations exist across the park.`;
        }

        const isNotFound = extractedSubject !== 'general' && topScore < 0.2 && !isFacilityMatch && !isEventQuery && !graphAugmented;
        const isGeneral = extractedSubject === 'general';
        const effectiveGeneral = isGeneral && topScore < 0.2;

        if (isNotFound || effectiveGeneral) {
            context = '';
            references = [];
        }

        let systemPrompt = '';
        const NO_THOUGHT_INSTRUCTION = "STRICT: Do NOT include any internal monologue or thinking process. Respond IMMEDIATELY with the final output.";

        if (isNotFound) {
            systemPrompt = isHindi
                ? `आप शेरा (Shera) हैं, दिल्ली चिड़ियाघर के मित्रवत गाइड।
${NO_THOUGHT_INSTRUCTION}

नियम:
1. हमेशा शेरा के रूप में उत्तर दें, और कभी न कहें कि आप AI हैं।
2. उपयोगकर्ता ने "${extractedSubject}" के बारे में पूछा है, लेकिन यह दिल्ली चिड़ियाघर में नहीं है।
3. स्पष्ट रूप से बताएं कि यह यहाँ नहीं है।
4. अपना उत्तर अधिकतम 20 शब्दों में दें। कोई बुलेट या शीर्षक नहीं।`

                : `You are Shera, the friendly guide of National Zoological Park, New Delhi.
${NO_THOUGHT_INSTRUCTION}

Rules:
1. Always stay in character as Shera and never mention AI.
2. The user asked about "${extractedSubject}", but it is NOT at the National Zoological Park, New Delhi.
3. Politely explain it is not here.
4. STRICT: Do not exceed 20 words. No bullet points or headings.`;

        } else if (isGeneral) {

            systemPrompt = isHindi
                ? `आप शेरा (Shera) हैं, दिल्ली चिड़ियाघर के गाइड।
${NO_THOUGHT_INSTRUCTION}

नियम:
1. स्वागतपूर्ण और मित्रवत रहें। कभी न कहें कि आप AI हैं।
2. उपयोगकर्ता के प्रश्नों का उत्तर अधिकतम 20 शब्दों में दें।
3. कोई बुलेट या शीर्षक नहीं।`

                : `You are Shera, the friendly zoo guide at National Zoological Park, New Delhi.
${NO_THOUGHT_INSTRUCTION}

Rules:
1. Be friendly and welcoming. Never mention AI.
2. Respond to the user naturally.
3. STRICT: Do not exceed 20 words. No bullet points or headings.`;

        } else {

            systemPrompt = isHindi
                ? `आप शेरा (Shera) हैं, राष्ट्रीय प्राणी उद्यान, नई दिल्ली के शेर गाइड।
${NO_THOUGHT_INSTRUCTION}

संदर्भ:
${context}

सख्त नियम:
1. हमेशा शेरा के रूप में उत्तर दें। कभी न कहें कि आप AI हैं।
2. उत्तर स्पष्ट, संक्षिप्त और तथ्यात्मक रखें।
3. अधिकतम 20 शब्दों में अपना उत्तर दें। कोई बुलेट या शीर्षक नहीं।
4. कभी भी नाम न बदलें। संदर्भ में दिए गए सटीक नाम का ही उपयोग करें।`

                : `You are Shera, the Lion Guide at National Zoological Park, New Delhi.
${NO_THOUGHT_INSTRUCTION}

Context:
${context}

STRICT RULES:
1. Always stay in character as Shera. Never mention AI.
2. Keep answers clear, concise, and factual.
3. STRICT: Do not exceed 20 words. No bullet points, headings, or emojis.
4. NEVER improvise or change names. Use EXACT names from context.`;
        }

        console.log(`[THINKING] Processing "${finalSubject}" with Gemini...`);
        console.log(`Generating response for: ${finalSubject}...`);

        if (stream) {
            res.setHeader('Content-Type', 'text/event-stream');
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('Connection', 'keep-alive');
            res.write(`data: ${JSON.stringify({ token: '', status: 'thinking' })}\n\n`);

            let fullAnswer = '';
            const result = await geminiModel.generateContentStream([systemPrompt, `User Question: ${question}\n\nShera's Response:`]);
            for await (const chunk of result.stream) {
                const token = chunk.text();
                fullAnswer += token;
                res.write(`data: ${JSON.stringify({ token })}\n\n`);
            }

            res.write(`data: ${JSON.stringify({ done: true, keyword: finalSubject, references })}\n\n`);
            res.end();

            logResources('Stream Complete');
            console.log(`Shera (streamed): ${fullAnswer}`);

        } else {
            const result = await geminiModel.generateContent([systemPrompt, `User Question: ${question}\n\nShera's Response:`]);
            let answer = result.response.text();

            // ── Clean Answer
            answer = answer.replace(/^(\*\*|)?Shera's Response:(\*\*|)?/gi, '').trim();
            answer = answer.replace(/^(\*\*|)?Response:(\*\*|)?/gi, '').trim();
            answer = answer.replace(/^(\*\*|)?Final Response:(\*\*|)?/gi, '').trim();
            answer = answer.replace(/^(\*\*|)?Answer:(\*\*|)?/gi, '').trim();
            answer = answer.trim();

            logResources('Response Generated');
            console.log(`Shera: ${answer}`);
            console.log(`[UI BINDING] Keyword: "${finalSubject}"`);

            const responsePayload = { answer, keyword: finalSubject, references };
            setCachedResponse(`${language}:${qLower}`, responsePayload);
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

    } catch (err) {
        console.error(`Failed to start server: ${err.message}`);
    }
})();