const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { Ollama } = require('ollama');
const { ChromaClient } = require('chromadb');
const { OllamaEmbeddingFunction } = require('@chroma-core/ollama');

/**
 * Shera AI - Hybrid Chroma + GraphRAG Backend Server
 * Semantic retrieval via ChromaDB
 * Relationship reasoning via Antigravity Graph
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

// Models
const EMBED_MODEL = 'nomic-embed-text';
const CHAT_MODEL = 'gemma2:2b';

function logResources(label) {
    const mem = process.memoryUsage();
    console.log(`[RESOURCES] ${label} - RSS: ${(mem.rss / 1024 / 1024).toFixed(2)}MB, Heap: ${(mem.heapUsed / 1024 / 1024).toFixed(2)}MB`);
}

// Paths
const GRAPH_PATH = path.join(__dirname, 'antigravity_store', 'graph.json');

// Clients
const ollama = new Ollama();
const chroma = new ChromaClient({ path: "http://localhost:8001" });
const embedder = new OllamaEmbeddingFunction({
    url: "http://127.0.0.1:11434",
    model: "nomic-embed-text"
});

let collection;

// Middleware
app.use(cors());
app.use(express.json());

// Graph Store
// ---------------------------
// Initialize Graph (DISABLED - Vector Mode)
// ---------------------------
// Replace the current empty graph initialization with:
function loadAllGraphs() {
    const storeDir = path.join(__dirname, 'antigravity_store');
    if (!fs.existsSync(storeDir)) {
        console.warn('No antigravity_store found. Running in vector-only mode.');
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

// Build adjacency map
const adjacencyMap = {};
for (const edge of graph.edges) {
    if (!adjacencyMap[edge.source]) adjacencyMap[edge.source] = [];
    if (!adjacencyMap[edge.target]) adjacencyMap[edge.target] = [];
    adjacencyMap[edge.source].push(edge);
    adjacencyMap[edge.target].push(edge);
}

function graphTraversal(startNodeId, maxHops = 2) {
    const visited = new Set();
    const results = [];
    const queue = [{ id: startNodeId, hop: 0 }];

    while (queue.length > 0) {
        const { id, hop } = queue.shift();
        if (visited.has(id) || hop > maxHops) continue;
        visited.add(id);

        const node = graph.nodes.find(n => n.id === id);
        if (node) results.push(node);

        const edges = adjacencyMap[id] || [];
        for (const edge of edges) {
            const nextId = edge.source === id ? edge.target : edge.source;
            if (!visited.has(nextId)) {
                queue.push({ id: nextId, hop: hop + 1 });
            }
        }
    }

    return results;
}

// ---------------------------
// Initialize Chroma
// ---------------------------
async function initChroma() {
    collection = await chroma.getOrCreateCollection({
        name: 'zoo_collection',
        embeddingFunction: embedder
    });

    console.log(`Connected to ChromaDB collection: zoo_collection`);
}

// ---------------------------
// Dynamic Zoo Registry
// ---------------------------
const zooRegistry = {
    canonicalNames: [],
    lookup: {},
    metadata: {}
};

function loadZooRegistry() {
    const dataDir = path.join(__dirname, 'zoo-data');
    if (!fs.existsSync(dataDir)) {
        console.error("Zoo data directory missing!");
        return;
    }

    const files = fs.readdirSync(dataDir).filter(f => f.endsWith('.json'));
    const names = new Set();

    for (const file of files) {
        if (file.includes('geojson') || file.includes('floorplan')) continue;
        try {
            const raw = fs.readFileSync(path.join(dataDir, file), 'utf8');
            const data = JSON.parse(raw);
            const items = Array.isArray(data) ? data : (data.data || [data]);

            for (const item of items) {
                const name = item.common_name?.en || item.render_name?.en || item.name?.en || item.name || item.title?.en || item.title;
                const classification = item.classification?.en || item.classification || "";
                const threatStatus = item.threat_status?.en || item.threat_status || "";

                if (name && typeof name === 'string' && !/^[0-9a-fA-F]{24}$/.test(name)) {
                    const cleanName = name.replace(/\s+\d+$/, '').trim();
                    if (cleanName.length > 2) {
                        names.add(cleanName);
                        // Store metadata if not already present or if new one has classification
                        if (!zooRegistry.metadata[cleanName] || classification || threatStatus) {
                            zooRegistry.metadata[cleanName] = {
                                classification,
                                threatStatus
                            };
                        }
                    }
                }
            }
        } catch (e) { }
    }

    zooRegistry.canonicalNames = Array.from(names);

    // Build fuzzy lookup
    const blacklist = new Set(['national', 'international', 'world', 'india', 'indian', 'park', 'zoo', 'day', 'and', 'the', 'for', 'with', 'birds', 'animals']);

    for (const canonical of zooRegistry.canonicalNames) {
        const lower = canonical.toLowerCase();
        const meta = zooRegistry.metadata[canonical];

        // Ensure name is at least two words if possible for better search accuracy
        let displayName = canonical;
        if (!canonical.includes(' ') && meta?.classification) {
            const firstClassWord = meta.classification.split(/[/\s,]+/)[0];
            if (firstClassWord && firstClassWord.length > 2) {
                displayName = `${canonical} ${firstClassWord}`;
            }
        }

        zooRegistry.lookup[lower] = displayName;

        // Map individual words if they are significant and NOT blacklisted
        const words = lower.split(/[/\s,.-]+/);
        for (const word of words) {
            if (word.length > 3 && !blacklist.has(word) && !zooRegistry.lookup[word]) {
                zooRegistry.lookup[word] = displayName;
            }
        }
    }

    // Hardcoded priorities for highly ambiguous one-word terms
    const priorityOverrides = {
        "lion": "Asiatic Lion",
        "lions": "Asiatic Lion",
        "tiger": "White Tiger",
        "tigers": "White Tiger",
        "elephant": "Indian Elephant",
        "elephants": "Indian Elephant",
        "monkey": "Bonnet Macaque",
        "monkeys": "Bonnet Macaque",
        "snake": "Indian Rock Python",
        "snakes": "Indian Rock Python"
    };
    Object.assign(zooRegistry.lookup, priorityOverrides);

    console.log(`\n📚 Zoo Registry: Loaded ${zooRegistry.canonicalNames.length} species dynamically.`);
}

// ---------------------------
// Helper: Optimize Context
// ---------------------------
function optimizeContext(paths, maxLines = 30) {
    return [...new Set(paths)]
        .slice(0, maxLines)
        .join('\n');
}

// ---------------------------
// Hybrid Search Engine (Scored Identity Reranking)
// ---------------------------
// ---------------------------
// Helper: Extract Subject & Facilities
// ---------------------------
async function extractSubject(query) {
    const qLower = query.toLowerCase();
    const dayPattern = /\b(national|world|international|global)\b[\w\s]+\bday\b/i;
    const dayMatch = query.match(dayPattern);
    if (dayMatch) {
        const eventName = dayMatch[0]
            .trim()
            .split(/\s+/)
            .map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
            .join(' ');
        console.log(`[EVENT] Day pattern detected: "${eventName}"`);
        return { subject: eventName, extractedSubject: eventName, matchedFacility: null };
    }

    // ---------------------------
    // High-Priority Conservation Keywords
    // ---------------------------
    if (qLower.includes("endangered") || qLower.includes("संकटग्रस्त") || qLower.includes("खतरे में") || qLower.includes("conservation")) {
        return { subject: "Endangered", extractedSubject: "Endangered", matchedFacility: null };
    }

    const extractionResp = await ollama.chat({
        model: CHAT_MODEL,
        messages: [{
            role: 'system',
            content: `Extract the most specific subject (Animal, Place, Facility, or Event) from the query. 
            Return ONLY the name in English. BE SPECIFIC.
            If the query is ONLY a greeting or casual talk, return "general".
            
            CRITICAL MAPPINGS:
            - "Sher", "Shera", "एशियाई शेर", "शेर" -> "Asiatic Lion"
            - "Bagh", "बाघ", "Tiger" -> "White Tiger"
            - "Hathi", "हाथी", "Elephant" -> "Indian Elephant"
            - "Monkey", "Monkeys", "बंदर", "कपि" -> "Bonnet Macaque"
            
            Always prefer specific names.
            No extra text.`
        }, {
            role: 'user',
            content: query
        }]
    });

    const extractedSubject = extractionResp.message.content.replace(/[^\w\s]/gi, '').trim();
    let subject = extractedSubject;

    // ---------------------------
    // Accuracy Shield: Dynamic Registry Lookup
    // ---------------------------
    const qWords = qLower.split(/\s+/);

    // 1. Try longest word match from registry (more specific than LLM sometimes)
    const sortedWords = qWords.filter(w => w.length > 3).sort((a, b) => b.length - a.length);
    for (const word of sortedWords) {
        if (zooRegistry.lookup[word]) {
            subject = zooRegistry.lookup[word];
            break;
        }
    }

    // 2. Canonicalize whatever we have (Registry or LLM)
    const sLower = subject.toLowerCase();
    if (zooRegistry.lookup[sLower]) {
        subject = zooRegistry.lookup[sLower];
    }

    // ---------------------------
    // Facility Mapping
    // ---------------------------
    const facilitySynonyms = {
        'Food & Drinks': ['food', 'eat', 'hungry', 'restaurant', 'cafe', 'cafeteria', 'snack', 'khana', 'खाना', 'खानपान', 'कैंटीन', 'canteen'],
        'Drinking Water': ['water', 'drink', 'drinking water', 'thirsty', 'fountain', 'pani', 'पानी', 'प्यास'],
        'Washrooms': ['washroom', 'toilet', 'restroom', 'bathroom', 'shauchalay', 'शौचालय', 'टॉयलेट'],
        'Buggy Stops': ['buggy', 'shuttle', 'ride', 'cart', 'transport', 'बग्गी'],
        'First Aid': ['first aid', 'medical', 'medicine', 'doctor', 'clinic', 'hospital', 'दवाई', 'अस्पताल'],
        'Counters': ['counter', 'ticket', 'info', 'information', 'help', 'टिकट', 'काउंटर']
    };

    let matchedFacility = null;
    for (const [facility, syns] of Object.entries(facilitySynonyms)) {
        if (syns.some(s => {
            if (/^[a-z\s]+$/i.test(s)) {
                const regex = new RegExp(`\\b${s}\\b`, 'i');
                return regex.test(qLower) || regex.test(sLower);
            }
            return qLower.includes(s) || sLower.includes(s);
        })) {
            matchedFacility = facility;
            break;
        }
    }

    if (matchedFacility) {
        subject = matchedFacility;
    }

    return { subject, extractedSubject, matchedFacility };
}

// ---------------------------
// Hybrid Search Engine (Scored Identity Reranking)
// ---------------------------
async function antigravitySearch(query, subject, isFacilityMatch, topK = 5, language = 'en') {
    console.log(`\n[SEARCH] Query: "${query}" (Lang: ${language})`);
    console.log(`[ENTITY] Target Subject: "${subject}"`);

    // Use the extracted English subject for the primary vector search
    // This is much more reliable than the raw query, especially for non-English queries
    const embedResp = await ollama.embed({
        model: EMBED_MODEL,
        input: subject
    });

    const queryEmbedding = embedResp.embeddings[0];

    // ---------------------------
    // Special Case: Endangered/Conservation Queries
    // ---------------------------
    if (subject.toLowerCase() === "endangered" || subject.toLowerCase() === "conservation") {
        const endangeredList = zooRegistry.canonicalNames.filter(name => {
            const rawStatus = zooRegistry.metadata[name]?.threatStatus || "";
            const status = String(rawStatus).toLowerCase();
            return status.includes('endangered') || status.includes('vulnerable') || status.includes('threatened');
        });

        if (endangeredList.length > 0) {
            console.log(`[ENTITY] Conservation Query detected. Found ${endangeredList.length} species.`);
            const listStr = endangeredList.slice(0, 15).join(", ");
            const customContext = `
The National Zoological Park, New Delhi is home to many endangered and threatened species.
Some of the most important endangered/vulnerable animals you can see here are: ${listStr}.
Visitors are encouraged to learn about their conservation and visit them at their enclosures.
            `.trim();

            return {
                context: customContext,
                sortedContext: endangeredList.slice(0, 5).map(name => ({
                    metadata: { name },
                    score: 5000,
                    doc: `This is the ${name}.`
                })),
                topScore: 5000
            };
        }
    }

    // 1. Try Exact ID Match first (Super reliable for known subjects)
    let exactMatch = null;
    try {
        const getRes = await collection.get({ ids: [subject] });
        if (getRes && getRes.ids && getRes.ids.length > 0) {
            console.log(`[ENTITY] Exact ID Match Found: "${subject}"`);
            exactMatch = {
                doc: getRes.documents[0],
                metadata: getRes.metadatas[0],
                score: 10000,
                originalName: getRes.ids[0]
            };
        }
    } catch (e) {
        // Not found by ID, continue to vector search
    }
    // After the exact ID match block, add:
    // --- Calendar Event Fallback: try core keyword if title variant didn't match ---
    if (!exactMatch && subject.match(/\b(national|international|world|global)\b/i)) {
        // Extract core subject (e.g. "National Sloth Day" -> "Sloth")
        const coreKeyword = subject
            .replace(/\b(national|international|world|global|day|for|to|the|of|and|in|a)\b/gi, '')
            .trim();

        if (coreKeyword) {
            console.log(`[EVENT] Trying core keyword fallback: "${coreKeyword}"`);
            // Search by event_keyword in metadata via vector search on the core word
            const coreEmbedResp = await ollama.embed({ model: EMBED_MODEL, input: coreKeyword + " day" });
            const coreResults = await collection.query({
                queryEmbeddings: [coreEmbedResp.embeddings[0]],
                nResults: 10,
                where: { is_calendar_event: "true" }  // Only search calendar events
            });

            if (coreResults.ids?.[0]?.length > 0) {
                // Pick the best event result and treat it as an exact match
                console.log(`[EVENT] Fallback match: "${coreResults.ids[0][0]}"`);
                exactMatch = {
                    doc: coreResults.documents[0][0],
                    metadata: coreResults.metadatas[0][0],
                    score: 8000,
                    originalName: coreResults.ids[0][0]
                };
            }
        }
    }

    const results = await collection.query({
        queryEmbeddings: [queryEmbedding],
        nResults: 150
    });

    const documents = results.documents?.[0] || [];
    const metadatas = results.metadatas?.[0] || [];
    const distances = results.distances?.[0] || [];

    const scoredContext = [];

    // Add exact match to context if found
    if (exactMatch) {
        scoredContext.push(exactMatch);
    }

    for (let i = 0; i < documents.length; i++) {
        const metadata = metadatas[i] || {};
        const distance = distances[i] || 1.0;
        const docName = (metadata.name || "").toLowerCase();

        // Base score: Convert distance (0 to 2+) to score (up to 1000)
        // 0.0 distance -> 1000 score
        // 1.0 distance -> 0 score
        let score = Math.max(0, (1.0 - distance) * 1000);

        if (i < 3) {
            console.log(`  - Match [${i}]: "${metadata.name}" | Distance: ${distance.toFixed(4)} | Base Score: ${score.toFixed(2)}`);
        }

        if (subject !== "general") {
            const cleanDocName = docName.replace(/\d+/g, '').trim().toLowerCase();
            const subjectLower = subject.toLowerCase();
            const docWords = cleanDocName.split(/\s+/);
            const subjectWords = subjectLower.split(/\s+/);

            // 1. Direct String Overlap
            const directMatch = cleanDocName.includes(subjectLower) || subjectLower.includes(cleanDocName);

            // 2. Word-based Overlap
            const matchedWords = subjectWords.filter(sw => docWords.includes(sw));
            const matchCount = matchedWords.length;
            const missingCount = subjectWords.filter(sw => !docWords.includes(sw)).length;

            if (i < 3) {
                console.log(`  - Match [${i}] Words: doc=[${docWords}], subject=[${subjectWords}], matched=[${matchedWords}], count=${matchCount}`);
            }

            if (directMatch || matchCount > 0) {
                if (cleanDocName === subjectLower) {
                    score += 4000; // Perfect exact match
                } else if (cleanDocName.endsWith(subjectLower)) {
                    score += 2500; // Strong suffix match (e.g. "Asiatic Lion" for "Lion")
                } else if (cleanDocName.startsWith(subjectLower)) {
                    score += 1500; // Strong prefix match
                } else {
                    score += (matchCount * 800); // Proportional word match
                    if (directMatch) score += 500;
                }

                // Penalize missing subject words heavily (Crucial for distinguishing specific species)
                score -= (missingCount * 2500);

                // Specificity Bonus: Only if we match ALL words from the subject
                if (missingCount === 0) {
                    const subjectWordCount = subjectWords.length;
                    const wordCount = docWords.length;
                    const extraWordsCount = wordCount - subjectWordCount;

                    score += (wordCount * 500);
                    score -= (extraWordsCount * 1000); // Penalty for extra words (prefer exact/shorter matches)
                    score += (cleanDocName.length * 10);
                }

                // Penalty for extra noise in doc name not present in subject or query
                const extraWords = docWords.filter(w => !subjectWords.includes(w) && !query.toLowerCase().includes(w));
                score -= (extraWords.length * 500); // Heavier penalty for noise

                // Biology Priority: If this is an animal, give it a significant edge
                if (metadata.full_data && metadata.full_data.includes("scientific_name")) {
                    score += 2000;
                }
            } else {
                // If NO word overlap, we need a decent semantic match
                if (distance > 0.75) {
                    score = 0; // Reject weak semantic-only matches
                } else if (distance > 0.6) {
                    score -= 300; // Moderate penalty for semantic match without words
                }
            }
        }

        // --- Bilingual Context Support ---
        let docText = documents[i];
        if (language === 'hi' && metadata.full_data) {
            try {
                const fullData = JSON.parse(metadata.full_data);
                docText = `
नाम: ${fullData.common_name?.hi || fullData.name?.hi || fullData.title?.hi || fullData.common_name?.en || metadata.name}
वैज्ञानिक नाम: ${fullData.scientific_name?.hi || fullData.scientific_name?.en || ""}
श्रेणी: ${fullData.category?.hi || fullData.category?.en || ""}
आवास: ${fullData.habitat?.hi || fullData.habitat?.en || ""}
आहार: ${fullData.diet?.hi || fullData.diet?.en || ""}
स्थान: ${fullData.location?.location_name?.hi || fullData.location_name?.hi || fullData.location?.location_name?.en || ""}
विवरण: ${fullData.narrative?.hi || fullData.description?.hi || fullData.text?.hi || fullData.narrative?.en || ""}
कहानियाँ: ${fullData.story_description?.hi || fullData.story_description?.en || ""}
                `.trim();
            } catch (e) {
                console.error("Hindi metadata parse error:", e);
            }
        }

        scoredContext.push({ doc: docText, score, originalName: metadata.name, metadata });
    }

    const sortedContext = scoredContext
        .filter(item => item.score > 0)
        .sort((a, b) => b.score - a.score);

    const resultDocs = sortedContext.slice(0, topK).map(item => item.doc);

    // Determine references based on score confidence
    const topScore = sortedContext.length > 0 ? sortedContext[0].score : 0;
    const refThreshold = Math.max(800, topScore * 0.85); // Increased minimum threshold to 800

    // Determine the best display name/keyword
    // If the extracted subject is specific and we have a high-confidence match, 
    // we prefer the subject as the binding keyword for better UI accuracy.
    let bestMatchName = subject;

    if (sortedContext.length > 0) {
        const topItem = sortedContext[0];
        const metaName = topItem.originalName.replace(/\s+\d+$/, '').trim();

        // If the subject is "general" or too vague, use the metaName
        const isVague = ["general", "animals", "birds", "reptiles", "mammals", "fish"].includes(subject.toLowerCase());

        if (isVague || topScore > 5000) {
            // If it's a very strong match for a specific doc, or subject is vague, use doc name
            if (!/^[0-9a-fA-F]{24}$/.test(metaName)) {
                bestMatchName = metaName;
            }
        }
    }

    // If the top match is very weak, we shouldn't bind to it as a specific animal
    if (topScore < 250 && !isFacilityMatch) {
        bestMatchName = "general";
    }

    if (sortedContext.length > 0) {
        console.log(`[SEARCH] Top Match: "${sortedContext[0].originalName}" | Score: ${sortedContext[0].score.toFixed(2)}`);
    }

    let references = sortedContext
        .filter(item => item.score >= refThreshold)
        .map(item => item.originalName.replace(/\s+\d+$/, '').trim());

    // If it's a facility match, we should be very strict about animal references
    // usually facilities shouldn't show animal cards unless explicitly asked
    const isFacility = /Washroom|Drinking Water|Buggy Stops|Food & Drinks|First Aid|Counters/.test(bestMatchName);
    if (isFacility) {
        references = []; // Suppress animal cards for facility queries
    }

    return {
        context: optimizeContext(resultDocs.slice(0, topK)),
        subject: bestMatchName,
        extractedSubject: subject,
        references: [...new Set(references)],
        topScore: topScore,
        isFacilityMatch: !!isFacilityMatch,
        sortedContext: sortedContext  // ✅ add this
    };
}

// ---------------------------
// Main Chat Endpoint
// ---------------------------
app.post('/api/shera/chat', async (req, res) => {
    const { question, deepSearch, language = 'en' } = req.body;

    logResources('Incoming Chat');

    const isHindi = language === 'hi';
    console.log(`\n--- Incoming: "${question}" (DeepSearch: ${deepSearch}, Lang: ${language}) ---`);

    try {
        // 1. Resolve entity first
        let { subject, extractedSubject, matchedFacility } = await extractSubject(question);

        // 2. Search for context
        const isFacilityMatch = !!matchedFacility;
        let { context, references, topScore, sortedContext } = await antigravitySearch(question, subject, isFacilityMatch, deepSearch ? 12 : 5, language);
        // --- Graph Augmentation ---
        // Use graph when: event query, low vector score, or relational query
        const isEventQuery = /\b(national|international|world|global)\b[\w\s]+\bday\b/i.test(question);
        const isRelationalQuery = /\b(eat|eats|live|lives|endangered|habitat|beat|location)\b/i.test(question);
        const needsGraph = (isEventQuery || isRelationalQuery || topScore < 500) && graph.nodes.length > 0;

        if (needsGraph) {
            // Find the best matching graph node for the subject
            const subjectLower = subject.toLowerCase();
            const matchedNode = graph.nodes.find(n =>
                n.id.toLowerCase().includes(subjectLower) ||
                subjectLower.includes(n.id.toLowerCase())
            );

            if (matchedNode) {
                const relatedNodes = graphTraversal(matchedNode.id, 2);
                const graphContext = relatedNodes
                    .filter(n => n.description)
                    .map(n => `${n.id} (${n.type}): ${n.description}`)
                    .join('\n');

                if (graphContext) {
                    console.log(`[GRAPH] Augmenting context with ${relatedNodes.length} graph nodes`);
                    context = graphContext + '\n\n' + context; // Graph context takes priority
                }
            }
        }
        // --- Synthetic Event Context for Calendar Events ---
        if (isEventQuery && (!context || context.trim().length < 50)) {
            const eventDate = sortedContext?.[0]?.metadata?.full_data
                ? (() => {
                    try {
                        const fd = JSON.parse(sortedContext[0].metadata.full_data);
                        return fd.date ? new Date(fd.date).toDateString() : '';
                    } catch { return ''; }
                })()
                : '';

            context = [
                `Event: ${subject}`,
                eventDate ? `Date: ${eventDate}` : '',
                `This is a special observance day recognized at the National Zoological Park, New Delhi.`,
                `Visitors are encouraged to learn about and celebrate this occasion during their visit.`
            ].filter(Boolean).join('\n');

            if (isHindi) {
                context = [
                    `कार्यक्रम: ${subject}`,
                    eventDate ? `तारीख: ${eventDate}` : '',
                    `यह नेशनल जूलॉजिकल पार्क, नई दिल्ली में मान्यता प्राप्त एक विशेष दिन है।`,
                    `आगंतुकों को इस अवसर पर जागरूक होने और इसे मनाने के लिए प्रोत्साहित किया जाता है।`
                ].filter(Boolean).join('\n');
            }
        }
        // 3. Inject facility context if missing but facility matched
        if (matchedFacility && topScore < 200) {
            context = `This facility is ${matchedFacility}. It provides essential services for visitors at the National Zoological Park. You can find multiple locations of this facility across the park.`;
            if (isHindi) {
                context = `यह सुविधा ${matchedFacility} है। यह नेशनल जूलॉजिकल पार्क में आगंतुकों के लिए आवश्यक सेवाएं प्रदान करती है। आप पार्क में इस सुविधा के कई स्थान पा सकते हैं।`;
            }
        }

        // A topScore below 250 after distance-based calculation means no relevant content was found
        // However, if it's a facility match (via keywords), we trust it.
        let graphAugmented = false;

        if (needsGraph) {
            const subjectLower = subject.toLowerCase();
            const matchedNode = graph.nodes.find(n =>
                n.id.toLowerCase().includes(subjectLower) ||
                subjectLower.includes(n.id.toLowerCase())
            );

            if (matchedNode) {
                const relatedNodes = graphTraversal(matchedNode.id, 2);
                const graphContext = relatedNodes
                    .filter(n => n.description)
                    .map(n => `${n.id} (${n.type}): ${n.description}`)
                    .join('\n');

                if (graphContext) {
                    console.log(`[GRAPH] Augmenting context with ${relatedNodes.length} graph nodes`);
                    context = graphContext + '\n\n' + context;
                    graphAugmented = true;  // ✅ flag it
                }
            }
        }
        const isNotFound = extractedSubject !== "general"
            && topScore < 250
            && !isFacilityMatch
            && !isEventQuery      // events always have an answer
            && !graphAugmented;   // graph found context

        const isGeneral = extractedSubject === "general";

        // IMPORTANT: If we found documents (topScore > 250), it's NOT a general greeting anymore
        // even if the extractor thought it was. This handles category searches better.
        const effectiveGeneral = isGeneral && topScore < 250;

        if (isNotFound || effectiveGeneral) {
            context = ""; // Clear context to prevent hallucination from noisy matches
            references = []; // Clear references to prevent showing unrelated animal cards
        }

        // 2. Build augmented prompt
        let systemPrompt = "";

        if (isNotFound) {
            systemPrompt = isHindi ? `
आप शेरा (Shera) हैं।
महत्वपूर्ण: उपयोगकर्ता ने "${extractedSubject}" के बारे में पूछा है, लेकिन यह दिल्ली चिड़ियाघर (National Zoological Park, New Delhi) में नहीं है।
आपको केवल यह बताना है कि यह यहाँ नहीं है। कोई अन्य जानकारी या रोचक तथ्य न दें।
उत्तर केवल हिंदी (Hindi) में दें।
            ` : `
You are Shera.
IMPORTANT: The user asked about "${extractedSubject}", but it is NOT at the National Zoological Park, New Delhi.
You must ONLY state that it is not currently at this zoo. DO NOT provide any other info or fun facts.
Respond ONLY in English.
            `;
        } else if (isGeneral) {
            systemPrompt = isHindi ? `
आप शेरा (Shera) हैं, एक मिलनसार गाइड।
उपयोगकर्ता का स्वागत करें या उनकी सामान्य बातचीत का उत्तर दें।
उत्तर केवल हिंदी (Hindi) में दें।
            ` : `
You are Shera, a friendly zoo guide.
Greet the user or respond to their general chat.
Respond ONLY in English.
            `;
        } else {
            systemPrompt = isHindi ? `
आप शेरा (Shera) हैं, एक "Lion Guide"।
महत्वपूर्ण: उपयोगकर्ता किसी भी भाषा में पूछे, आपको उत्तर केवल हिंदी (Hindi) में ही देना है।

सीमा: कुल अधिकतम 25 शब्द।
शैली:
- इस प्रारूप का उपयोग करें: "**विवरण**: ...", "**स्थान**: ...", "**रोचक तथ्य**: ..."
- "विवरण", "स्थान", और "रोचक तथ्य" लेबल्स को हमेशा डबल एस्टेरिस्क (**) में रखें।
- किसी भी प्रतीक (*, -, आदि) का उपयोग न करें।
- स्थान (स्थान) के बारे में हमेशा हिंदी में ही जानकारी दें।
- यदि सटीक स्थान उपलब्ध नहीं है, तो कहें: "नीचे दिए गए नेविगेट बटन पर क्लिक करके वहां पहुंचें।" (अंग्रेजी शब्दों का प्रयोग न करें)।
- कभी भी "N/A", "Location", "Label", या "Data Missing" का उपयोग न करें।

निर्देश:
- केवल (Context) में दी गई जानकारी का उपयोग करें।
- जवाब एकदम छोटा, पेशेवर और मददगार होना चाहिए।

संदर्भ (Context):
${context}

याद रखें: केवल हिंदी (Hindi) में ही उत्तर देना है।
            ` : `
You are Shera, the Lion Guide.
IMPORTANT: Respond ONLY in English.

Strict Limit: Max 25 words total.
Style:
- Use this exact format: "**Overview**: ...", "**Location**: ...", "**Fun Fact**: ..."
- DO NOT use bullet points, asterisks (*), or dashes (-).
- For Location, always say: "Click on the bottom 'Navigate' button to find your way there."
- NEVER output "N/A", "Not available", "Label", or placeholders.

Instructions:
- Use ONLY the provided context.
- Keep the response ultra-short, professional, and helpful.

Context:
${context}

Remember: Respond ONLY in English.
            `;
        }

        console.log(`Generating response...`);

        const chatResponse = await ollama.chat({
            model: CHAT_MODEL,
            messages: [
                { role: 'system', content: systemPrompt },
                { role: 'user', content: question }
            ],
            stream: false
        });

        const answer = chatResponse.message.content;

        logResources('Response Generated');

        console.log(`Shera: ${answer}`);
        console.log(`[UI BINDING] Keyword: "${subject}"`);

        res.json({
            answer,
            keyword: subject,
            references: references
        });

    } catch (error) {
        console.error('Error:', error);
        res.status(500).json({
            error: 'Internal Server Error',
            details: error.message
        });
    }
});

// ---------------------------
// Health Check
// ---------------------------
app.get('/api/health', (req, res) => {
    res.json({
        status: 'Shera is alive',
        timestamp: new Date().toISOString(),
        language_supported: ['en', 'hi']
    });
});

// ---------------------------
// Start Server
// ---------------------------
(async () => {
    try {
        loadZooRegistry();
        await initChroma();

        const server = app.listen(port, host, () => {
            console.log(`\n🦁 Shera AI Backend (Vector Mode) running on http://${host}:${port}`);
            console.log(`POST /api/shera/chat`);
            console.log(`GET /api/health`);
        });

        server.on('error', (err) => {
            console.error('Server error:', err);
            if (err.code === 'EADDRINUSE') {
                console.error(`Port ${port} is already in use. Please kill the existing process.`);
            }
        });

    } catch (err) {
        console.error(`Failed to start server: ${err.message}`);
    }
})();