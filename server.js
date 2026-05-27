const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { Ollama } = require('ollama');
const { ChromaClient } = require('chromadb');
const { OllamaEmbeddingFunction } = require('@chroma-core/ollama');

/**
 * Shera AI - Hybrid Chroma + GraphRAG Backend Server
 * 
 * ULTRA-LIGHTWEIGHT CONFIGURATION:
 * Optimizations applied for Qwen 2 (0.5b):
 *  1. Tiny 350MB footprint for ultra-low latency on weak VMs.
 *  2. Extreme sampling speed (Temp 0.7, Top P 0.8).
 *  3. LLM-Bypass Extraction: Skips LLM for short/simple subject queries.
 *  4. Graph traversal result cache.
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
const CHAT_MODEL = 'llama3.2:1b';
const EXTRACTION_MODEL = 'llama3.2:1b';

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
    'shera': 'Asiatic Lion',
    'bagh': 'White Tiger',
    'hathi': 'Indian Elephant',
    'bhalu': 'Sloth Bear',
    'भालू': 'Sloth Bear',
    'saap': 'Common Rat Snake',
    'saanp': 'Common Rat Snake',
    'सांप': 'Common Rat Snake',
    'साँप': 'Common Rat Snake',
    'magarmach': 'Marsh Crocodile',
    'मगरमच्छ': 'Marsh Crocodile',
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
    'cobra': 'Spectacled Cobra',
    'python': 'Indian Rock Python',
    'deer': 'Spotted Deer',
    'bear': 'Sloth Bear',
    'bears': 'Sloth Bear',
    'Food & Drinks': 'Food & Drinks',
    'Drinking Water': 'Drinking Water',
    'Washrooms': 'Washrooms',
    'Buggy Stops': 'Buggy Stops',
    'Emergency': 'Emergency'
};

// ─── OPTIMIZATION 1: Trie Index ───────────────────────────────────────────────
let trieIndex = new Map();

// Generic category words that should NEVER be used as trie lookup keys.
// These are too broad and cause false single-match fast-extractions.
const TRIE_BLACKLIST = new Set([
    'bird', 'birds', 'animal', 'animals', 'reptile', 'reptiles',
    'mammal', 'mammals', 'fish', 'insect', 'insects', 'plant', 'plants',
    'cat', 'cats', 'dog', 'dogs', 'pet', 'pets',
]);

function buildTrieIndex() {
    const entries = [];
    for (const [phrase, name] of Object.entries(zooRegistry.lookup)) {
        if (TRIE_BLACKLIST.has(phrase.toLowerCase())) continue; // skip generic words
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
        if (wordCount <= 7) return uniqueNames[0]; // raised from < 6 to handle 6-7 word queries
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
    const matrix = [];
    for (let i = 0; i <= b.length; i++) matrix[i] = [i];
    for (let j = 0; j <= a.length; j++) matrix[0][j] = j;
    for (let i = 1; i <= b.length; i++) {
        for (let j = 1; j <= a.length; j++) {
            if (b.charAt(i - 1) === a.charAt(j - 1)) {
                matrix[i][j] = matrix[i - 1][j - 1];
            } else {
                matrix[i][j] = Math.min(
                    matrix[i - 1][j - 1] + 1, // substitution
                    matrix[i][j - 1] + 1,     // insertion
                    matrix[i - 1][j] + 1      // deletion
                );
            }
        }
    }
    return matrix[b.length][a.length];
}

function normalizeToRegistryOrSelf(rawSubject) {
    const words = rawSubject.trim().split(/\s+/);
    // 1. Direct candidate matching
    for (let len = words.length; len >= 1; len--) {
        const candidate = words.slice(0, len).join(' ');
        const lower = candidate.toLowerCase();
        if (zooRegistry.lookup[lower]) return zooRegistry.lookup[lower];
        if (zooRegistry.canonicalNames.some(n => n.toLowerCase() === lower)) {
            return zooRegistry.canonicalNames.find(n => n.toLowerCase() === lower);
        }
    }

    // 2. Exact word-level matching for any word in the query
    for (const queryWord of words) {
        const qwLower = queryWord.toLowerCase();
        if (qwLower.length < 3) continue;
        if (zooRegistry.lookup[qwLower]) return zooRegistry.lookup[qwLower];
        const exactWordHit = zooRegistry.canonicalNames.find(n => {
            const nl = n.toLowerCase();
            return nl.split(/[^a-z0-9]+/).includes(qwLower);
        });
        if (exactWordHit) return exactWordHit;
    }

    // 3. Fuzzy word-level matching
    for (const queryWord of words) {
        const qwLower = queryWord.toLowerCase();
        if (qwLower.length < 3) continue;
        const fuzzyWordHit = zooRegistry.canonicalNames.find(n => {
            const nl = n.toLowerCase();
            const canonicalWords = nl.split(/[^a-z0-9]+/);
            for (const cw of canonicalWords) {
                if (cw.length < 3) continue;
                const dist = levenshtein(qwLower, cw);
                const maxDist = cw.length >= 6 ? 2 : 1;
                if (dist <= maxDist) return true;
            }
            return false;
        });
        if (fuzzyWordHit) return fuzzyWordHit;
    }

    return words[0];
}

// ─── Context Helper ───────────────────────────────────────────────────────────
function optimizeContext(docs, maxLines = 5) {
    return [...new Set(docs)].slice(0, maxLines).join('\n');
}

// ─── Facility Synonym Map ─────────────────────────────────────────────────────
const facilitySynonyms = {
    'Food & Drinks': ['food', 'eat', 'hungry', 'snacks', 'restaurant', 'cafe', 'cafeteria', 'snack', 'khana', 'khaana', 'खाना', 'खानपान', 'भूख', 'bhook', 'bhuk', 'कैंटीन', 'canteen', 'kanteen'],
    'Drinking Water': ['water', 'drink', 'drinking water', 'thirsty', 'thristy', 'thirsti', 'thirsy', 'thurst', 'fountain', 'pani', 'paani', 'पानी', 'प्यास', 'pyaas', 'pyasa', 'pyaasa', 'pyase', 'pyaase', 'water bottle', 'water filter'],
    'Washrooms': ['washroom', 'wash room', 'toilet', 'restroom', 'bathroom', 'shauchalay', 'shochalay', 'शौचालय', 'टॉयलेट', 'pee', 'poo', 'mutralay'],
    'Buggy Stops': ['buggy', 'shuttle', 'ride', 'cart', 'transport', 'बग्गी', 'gadi', 'gaadi', 'shuttle car', 'rickshaw', 'rikshaw'],
    'First Aid': ['first aid', 'medical', 'medicine', 'doctor', 'clinic', 'hospital', 'दवाई', 'अस्पताल', 'dawai', 'dawae', 'chot', 'injur'],
    'Counters': ['counter', 'ticket', 'info', 'information', 'help', 'टिकट', 'काउंटर', 'booking', 'paise'],
    'Exit Gate': ['exit', 'exit gate', 'way out', 'leave the zoo', 'going out', 'निकास', 'निकास द्वार', 'बाहर', 'बाहर निकलें', 'बाहर जाएं', 'nikas', 'bahar'],
    'Main Entrance': ['entrance', 'entry', 'enter', 'main entrance', 'main gate', 'front gate', 'प्रवेश', 'प्रवेश द्वार', 'दरवाज़ा', 'द्वार', 'मुख्य द्वार', 'pravesh', 'entry gate']
};

// Returns first matching facility (for single-match use)
function detectFacility(text) {
    const facilities = detectFacilities(text);
    return facilities.length > 0 ? facilities[0] : null;
}

// Returns ALL matching facilities in the text (for multi-match use)
function detectFacilities(text) {
    const t = text.toLowerCase();
    const matched = new Set();

    // 1. Direct / exact match
    for (const [facility, syns] of Object.entries(facilitySynonyms)) {
        for (const s of syns) {
            const hit = /^[a-z\s]+$/i.test(s)
                ? new RegExp(`\\b${s}\\b`, 'i').test(t)
                : t.includes(s);
            if (hit) { matched.add(facility); break; }
        }
    }

    // 2. Levenshtein typo-tolerant word checking
    const words = t.split(/[^a-zA-Z]+/);
    for (const word of words) {
        if (word.length < 4) continue;
        for (const [facility, syns] of Object.entries(facilitySynonyms)) {
            if (matched.has(facility)) continue; // already matched
            for (const s of syns) {
                if (!/^[a-zA-Z]+$/.test(s) || s.length < 4) continue;
                const dist = levenshtein(word, s);
                const maxAllowedDist = s.length >= 6 ? 2 : 1;
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

function finalizeSubject(subject, qLower, extractedSubject = null) {
    const matchedFacilities = detectFacilities(qLower);
    const matchedFacility = matchedFacilities.length > 0 ? matchedFacilities.join(', ') : null;
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

    // Bare "{topic} day" pattern — resolve to matching event in registry
    const bareDayMatch = qLower.match(/\b([a-z][a-z\s]{1,30})\s+day\b/);
    if (bareDayMatch) {
        const dayTopic = bareDayMatch[1].trim();
        const topicWords = dayTopic.split(/\s+/);
        const eventHit = [...zooRegistry.eventNames].find(name => {
            const nl = name.toLowerCase();
            // Match if any meaningful topic word appears in the event name
            return topicWords.some(w => w.length > 3 && nl.includes(w));
        });
        if (eventHit) {
            console.log(`[EVENT] Bare day match: "${dayTopic}" → "${eventHit}"`);
            return { subject: eventHit, extractedSubject: eventHit, matchedFacility: null, isEventOverride: true };
        }
    }

    if (qLower.includes('endangered') || qLower.includes('संकटग्रस्त')
        || qLower.includes('खतरे में') || qLower.includes('conservation')) {
        return finalizeSubject('Endangered', qLower);
    }

    const facilityHits = detectFacilities(qLower);
    if (facilityHits.length > 0) return finalizeSubject(facilityHits.join(', '), qLower);

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

    // FAST FALLBACK: If query is very short (≤4 words), skip LLM extraction
    // NOTE: facility/gate checks already ran above, so this only catches truly short unknown queries
    const words = qLower.split(/\s+/);
    if (words.length <= 4) {
        console.log(`[EXTRACTION] Skipping LLM for short query: "${qLower}"`);
        return finalizeSubject(normalizeToRegistryOrSelf(qLower), qLower, qLower);
    }

    let subject = 'general';
    try {
        const extractionResp = await ollama.chat({
            model: EXTRACTION_MODEL,
            messages: [
                {
                    role: 'system',
                    content: `Extract the primary subject (Animal, Place, Facility, or Event) from the query.
Return ONLY the name in English. BE EXTREMELY BRIEF.
Mappings: "Sher/Shera"->"Asiatic Lion", "Bagh"->"White Tiger", "Hathi"->"Indian Elephant", "Pani"->"Drinking Water".`
                },
                { role: 'user', content: query }
            ],
            keep_alive: '1h',
            options: { num_predict: 32, temperature: 0, num_ctx: 1024 }
        });
        // Strip gemma4 <think>...</think> blocks and clean up
        let rawContent = extractionResp.message.content || '';
        rawContent = rawContent.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
        rawContent = rawContent.replace(/<\|channel\|>thought[\s\S]*?<channel\|>/gi, '').trim();
        rawContent = rawContent.replace(/<(thought|reasoning)>[\s\S]*?<\/\1>/gi, '').trim();
        // Keep only the first line (gemma4 sometimes adds explanations after)
        rawContent = rawContent.split('\n')[0].trim();
        const raw = rawContent.replace(/[^\w\s\(\)\-\.&]/gi, '').trim();
        console.log(`[EXTRACTION] LLM raw: "${rawContent}" → normalized: "${raw}"`);
        subject = raw ? normalizeToRegistryOrSelf(raw) : 'general';
    } catch (e) {
        console.error('[LLM] Extraction error:', e.message);
    }

    const sLower = subject.toLowerCase();
    if (zooRegistry.lookup[sLower]) subject = zooRegistry.lookup[sLower];

    return finalizeSubject(subject, qLower, subject);
}

function isAnimalActive(activityStr, currentHour) {
    const act = (activityStr || '').toLowerCase();

    // Default to Diurnal (active during day: 6 AM to 8 PM) if not specified
    if (!act) {
        return currentHour >= 6 && currentHour < 20;
    }

    if (act.includes('nocturnal')) {
        return currentHour >= 18 || currentHour < 6; // Active from 6 PM to 6 AM
    }

    if (act.includes('diurnal')) {
        return currentHour >= 6 && currentHour < 18; // Active from 6 AM to 6 PM
    }

    // Crepuscular / specific times (e.g. morning, evening, night)
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
                // cross midnight
                return currentHour >= start || currentHour < end;
            }
        });
    }

    // Default fallback
    return currentHour >= 6 && currentHour < 20;
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
            return status.includes('endangered') || status.includes('threatened');
        });

        if (endangeredList.length > 0) {
            console.log(`[ENTITY] Conservation query. Found ${endangeredList.length} species.`);
            const listStr = endangeredList.slice(0, 15).join(', ');
            return {
                context: `The National Zoological Park, New Delhi is home to many endangered and threatened species.\nSome key endangered animals here are: ${listStr}.\nVisitors are encouraged to learn about their conservation.`,
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
        const idsToTry = [subject];
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

    let sortedContext = scoredContext
        .filter(item => item.score > 0)
        .sort((a, b) => b.score - a.score);

    if (sortedContext.length === 0 && subject !== 'general' && zooRegistry.canonicalNames.includes(subject)) {
        console.log(`[SEARCH] Fallback: No vector results, but "${subject}" is in registry. Using metadata.`);
        const meta = zooRegistry.metadata[subject] || {};
        const fallbackItem = {
            doc: `The ${subject} is one of the species at National Zoological Park, New Delhi. ${meta.classification ? `It is classified as ${meta.classification}.` : ''}`,
            metadata: { name: subject, ...meta },
            score: 0.5
        };
        scoredContext.push(fallbackItem);
        sortedContext.push(fallbackItem);
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

function sendStaticResponse(res, answer, keyword, stream) {
    if (stream) {
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.write(`data: ${JSON.stringify({ token: '', status: 'thinking' })}\n\n`);
        res.write(`data: ${JSON.stringify({ token: answer })}\n\n`);
        res.write(`data: ${JSON.stringify({ done: true, keyword, references: [] })}\n\n`);
        return res.end();
    } else {
        return res.json({ answer, keyword, references: [] });
    }
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

        let { subject, extractedSubject, matchedFacility } = await extractSubject(question);

        if (subject === 'general') {
            const isGreeting = /^(hello|hi|hey|नमस्ते|tata|bye|goodbye|thank|thanks)$/i.test(qLower);
            const isLocationQuery = extractedSubject === 'location'
                || qLower.includes('nearby')
                || qLower.includes('close to me')
                || qLower.includes('where am i')
                || qLower.includes('आसपास')
                || qLower.includes('नज़दीक')
                || qLower.includes('पास');

            const isActivityQuery = qLower.includes('active now')
                || qLower.includes('currently active')
                || qLower.includes('active right now')
                || (qLower.includes('active') && qLower.includes('animal'))
                || qLower.includes('सक्रिय');

            if (isActivityQuery) {
                subject = 'Activity';
                extractedSubject = 'Activity';
            } else if (!isGreeting && !isLocationQuery) {
                console.log(`[GENERAL] Factual check for unrecognized query: "${question}"`);
                const testSearch = await antigravitySearch(question, question, false, 1, language, false);
                if (testSearch.topScore >= 0.4) {
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
                ? `आप शेरा हैं। "${subject}" दिल्ली चिड़ियाघर में नहीं है। शेरा के रूप में उत्तर दें। कभी न कहें कि आप AI हैं। हिंदी में उत्तर दें। सख्त: 20 शब्दों से कम में उत्तर दें।`
                : `You are Shera. "${subject}" is NOT at the National Zoological Park, New Delhi. Respond as Shera. NEVER say you are an AI. Respond in English. STRICT: Do not exceed 20 words.`;

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
                    options: { num_predict: 350, temperature: 0.7, top_p: 0.8, num_ctx: 1024 }
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
                    options: { num_predict: 350, temperature: 0.7, top_p: 0.8, num_ctx: 1024 }
                });
                return res.json({ answer: resp.message.content, keyword: 'general', references: [] });
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

            const isLocationQuery = extractedSubject === 'location'
                || qLower.includes('nearby')
                || qLower.includes('close to me')
                || qLower.includes('where am i')
                || qLower.includes('आसपास')
                || qLower.includes('नज़दीक')
                || qLower.includes('पास');

            if (isLocationQuery) {
                console.log(`[GENERAL] Static location match: "${qLower}"`);
                const msg = isHindi
                    ? "आपके आस-पास के जानवरों और सुविधाओं को ढूंढ रहा हूँ... 🗺️"
                    : "Finding nearby animals and facilities for you... 🗺️";
                return sendStaticResponse(res, msg, 'general', stream);
            }

            console.log(`[GENERAL] Handling as general chat/greeting via LLM.`);
            const greetingPrompt = isHindi
                ? 'आप शेरा हैं, एक मिलनसार चिड़ियाघर गाइड। उपयोगकर्ता का स्वागत करें या उनकी सामान्य बातचीत का उत्तर दें। कभी न कहें कि आप AI हैं। हिंदी में उत्तर दें। उत्तर में प्यारे और प्रासंगिक इमोजीस (जैसे 🦁, 👋, ✨) का प्रयोग करें। उत्तर को प्राकृतिक और संक्षिप्त रखें (लगभग 20-30 शब्द)।'
                : 'You are Shera, a friendly zoo guide. Greet the user or respond to their general talk. NEVER say you are an AI. Respond in English. Warmly include friendly emojis (e.g. 🦁, 👋, ✨). Keep it natural and concise (around 20-30 words).';

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
                    options: { num_predict: 350, temperature: 0.9, top_p: 0.9, num_ctx: 2048 }
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
                    options: { num_predict: 350, temperature: 0.9, top_p: 0.9, num_ctx: 1024 }
                });
                return res.json({ answer: resp.message.content, keyword: 'general', references: [] });
            }
        }

        let context = '';
        let references = [];
        let topScore = 0;
        let sortedContext = [];
        let finalSubject = subject;

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
                const listStr = activeAnimals.slice(0, 10).join(', ') + ', etc.';
                const timeDesc = currentHour >= 12
                    ? `${currentHour === 12 ? 12 : currentHour - 12} PM`
                    : `${currentHour} AM`;

                context = isHindi
                    ? `चिड़ियाघर में अभी भारतीय समयानुसार लगभग ${timeDesc} बज रहे हैं। इस समय दिल्ली चिड़ियाघर में सक्रिय और देखने योग्य मुख्य जानवर निम्नलिखित हैं: ${activeAnimals.slice(0, 10).join(', ')} आदि।`
                    : `The current local time at the zoo is around ${timeDesc}. The key animals that are currently active and likely to be seen right now are: ${listStr}.`;

                sortedContext = activeAnimals.slice(0, 5).map(name => ({
                    metadata: { name }, score: 1.0, doc: `This is the ${name}.`
                }));
                topScore = 1.0;
                finalSubject = 'Activity';
                references = activeAnimals.slice(0, 5);
            }
        }

        if (!context) {
            const searchResult = await antigravitySearch(question, subject, isFacilityMatch, 1, language, isEventQuery);
            context = searchResult.context;
            references = searchResult.references;
            topScore = searchResult.topScore;
            sortedContext = searchResult.sortedContext;
            finalSubject = searchResult.subject;
        }

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
        const NO_THOUGHT_INSTRUCTION_EN = "STRICT: Do NOT include any internal monologue or thinking process. Respond IMMEDIATELY with the final output in English.";
        const NO_THOUGHT_INSTRUCTION_HI = "सख्त निर्देश: कोई भी आंतरिक सोच या विचार प्रक्रिया (thinking process) शामिल न करें। सीधे केवल अंतिम उत्तर ही हिंदी में लिखें।";

        if (isNotFound) {
            systemPrompt = isHindi
                ? `आप शेरा (Shera) हैं, दिल्ली चिड़ियाघर के मित्रवत गाइड।
${NO_THOUGHT_INSTRUCTION_HI}

नियम:
1. हमेशा शेरा के रूप में उत्तर दें, और कभी न कहें कि आप AI हैं।
2. उपयोगकर्ता ने "${extractedSubject}" के बारे में पूछा है, लेकिन यह दिल्ली चिड़ियाघर में नहीं है।
3. स्पष्ट और मित्रवत रूप से बताएं कि यह यहाँ नहीं है।
4. अपने उत्तर में संबंधित इमोजीस (जैसे 🦁, 🗺️) शामिल करें।
5. उत्तर छोटा और सरल रखें (लगभग 20-30 शब्द)। कोई बुलेट या शीर्षक नहीं।`

                : `You are Shera, the friendly guide of National Zoological Park, New Delhi.
${NO_THOUGHT_INSTRUCTION_EN}

Rules:
1. Always stay in character as Shera and never mention AI.
2. The user asked about "${extractedSubject}", but it is NOT at the National Zoological Park, New Delhi.
3. Politely and warmly explain it is not here.
4. Use relevant and friendly emojis in your response (e.g. 🦁, 🗺️).
5. Keep your response concise but natural (around 20-30 words). No bullet points or headings.`;

        } else if (isGeneral) {

            systemPrompt = isHindi
                ? `आप शेरा (Shera) हैं, दिल्ली चिड़ियाघर के गाइड।
${NO_THOUGHT_INSTRUCTION_HI}

नियम:
1. स्वागतपूर्ण, मित्रवत और खुशमिजाज रहें। कभी न कहें कि आप AI हैं।
2. अपने उत्तर में प्यारे और प्रासंगिक इमोजीस (जैसे 🦁, 👋, ✨) शामिल करें।
3. उत्तर लगभग 20-30 शब्दों का रखें। कोई बुलेट या शीर्षक नहीं।`

                : `You are Shera, the friendly zoo guide at National Zoological Park, New Delhi.
${NO_THOUGHT_INSTRUCTION_EN}

Rules:
1. Be friendly, welcoming, and cheerful. Never mention AI.
2. Respond to the user naturally and include friendly emojis (e.g. 🦁, 👋, ✨).
3. Keep it warm and concise (around 20-30 words). No bullet points or headings.`;

        } else {

            systemPrompt = isHindi
                ? `आप शेरा (Shera) हैं, राष्ट्रीय प्राणी उद्यान, नई दिल्ली के शेर गाइड।
${NO_THOUGHT_INSTRUCTION_HI}

संदर्भ:
${context}

सख्त नियम:
1. हमेशा शेरा के रूप में उत्तर दें। कभी न कहें कि आप AI हैं।
2. उत्तर स्पष्ट, संक्षिप्त और तथ्यात्मक रखें।
3. अपने उत्तर में हमेशा जानवरों या स्थान से जुड़े उपयुक्त इमोजीस (जैसे 🦁, 🐅, 🌳) का प्रयोग करें।
4. उत्तर को प्राकृतिक और मित्रवत रखें (लगभग 30-40 शब्द)। कोई बुलेट या शीर्षक नहीं।
5. कभी भी नाम न बदलें। संदर्भ में दिए गए सटीक नाम का ही उपयोग करें।`

                : `You are Shera, the Lion Guide at National Zoological Park, New Delhi.
${NO_THOUGHT_INSTRUCTION_EN}

Context:
${context}

STRICT RULES:
1. Always stay in character as Shera. Never mention AI.
2. Keep answers clear, concise, and factual.
3. Warmly include relevant emojis in your response (e.g. 🦁, 🐅, 🌳).
4. Keep the response natural and friendly (around 30-40 words). No bullet points or headings.
5. NEVER improvise or change names. Use EXACT names from context.`;
        }

        console.log(`[THINKING] Processing "${finalSubject}" with ${CHAT_MODEL}...`);
        console.log(`Generating response for: ${finalSubject}...`);

        let userMessageContent = question;
        if (finalSubject && finalSubject !== 'general' && finalSubject.toLowerCase() !== question.toLowerCase()) {
            userMessageContent = `About "${finalSubject}": ${question}`;
        }

        if (stream) {
            res.setHeader('Content-Type', 'text/event-stream');
            res.setHeader('Cache-Control', 'no-cache');
            res.setHeader('Connection', 'keep-alive');
            res.write(`data: ${JSON.stringify({ token: '', status: 'thinking' })}\n\n`);

            let fullAnswer = '';
            const streamResp = await ollama.chat({
                model: CHAT_MODEL,
                messages: [
                    { role: 'system', content: systemPrompt },
                    { role: 'user', content: userMessageContent }
                ],
                stream: true,
                keep_alive: '1h',
                options: { num_predict: 650, temperature: 0.7, top_p: 0.8, num_ctx: 2048 }
            });

            for await (const chunk of streamResp) {
                const token = chunk.message?.content || '';
                fullAnswer += token;
                res.write(`data: ${JSON.stringify({ token })}\n\n`);
            }

            res.write(`data: ${JSON.stringify({ done: true, keyword: finalSubject, references })}\n\n`);
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
                options: { num_predict: 650, temperature: 0.7, top_p: 0.8, num_ctx: 1024 }
            });

            console.log('[DEBUG] Raw Ollama Response:', JSON.stringify(chatResponse, null, 2));

            let answer = chatResponse.message?.content || '';
            const thought = chatResponse.message?.thinking || '';

            if (thought) {
                console.log(`\n[MODEL THOUGHT PROCESS]:\n${thought}\n`);
            }

            // ── Clean Answer: Stripping Gemma 4 specific tags and fallbacks
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
                // Fallback to extract a draft from thought process if present
                const draftMatch = thought.match(/Draft:\*?\s*([^\n\r]+)/i) || thought.match(/\*Draft:\*\s*([^\n\r]+)/i);
                if (draftMatch && draftMatch[1]) {
                    answer = draftMatch[1].trim();
                    // Strip off English translation helpers (e.g. "(Asiatic lions...)")
                    answer = answer.replace(/\s*\([^)]+\)\.?/g, '').trim();
                    console.log(`[FALLBACK] Extracted draft from thought: "${answer}"`);
                }
            }

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