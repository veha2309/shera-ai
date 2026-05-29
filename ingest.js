const fs = require('fs');
const path = require('path');
const { ChromaClient } = require('chromadb');
const { OllamaEmbeddingFunction } = require('@chroma-core/ollama');
const { Ollama } = require('ollama');

/**
 * Shera AI - Enhanced Chroma Ingestion Script
 * Ingests ALL files in zoo-data with dedicated handlers per file type.
 */

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

/** Extract English string from { en, hi } objects or plain strings */
function en(val) {
    if (!val) return '';
    if (typeof val === 'object') return val.en || '';
    return String(val);
}

/** Join a list of non-empty strings with a separator */
function join(parts, sep = '\n') {
    return parts.filter(Boolean).join(sep);
}

// ─────────────────────────────────────────────
// Ollama embed wrapper
// ─────────────────────────────────────────────
async function embed(ollama, model, text) {
    const res = await ollama.embed({ model, input: text, keep_alive: '1h' });
    return res.embeddings[0];
}

// ─────────────────────────────────────────────
// Upsert helper
// ─────────────────────────────────────────────
async function store(collection, ollama, model, { id, label, text, metadata }) {
    if (!text || !text.trim()) return;
    const embedding = await embed(ollama, model, text);
    await collection.upsert({
        ids: [id],
        embeddings: [embedding],
        documents: [text],
        metadatas: [{ ...metadata, doc_id: id }]
    });
    console.log(`  [OK] Stored: ${label}`);
}

// ─────────────────────────────────────────────
// Accuracy Shield: make single-word names context-rich
// ─────────────────────────────────────────────
const MANUAL_NAME_MAP = {
    "lion": "Asiatic Lion",
    "tiger": "White Tiger",
    "elephant": "Indian Elephant",
    "monkey": "Bonnet Macaque",
    "rhino": "Indian Rhinoceros",
    "rhinoceros": "Indian Rhinoceros"
};

function enrichName(name, classification = '') {
    // Strip the number completely so it matches server.js canonicalNames
    let baseName = name.replace(/\s+\d+$/, '').trim();

    if (!baseName.includes(' ')) {
        const lower = baseName.toLowerCase();
        if (MANUAL_NAME_MAP[lower]) {
            return MANUAL_NAME_MAP[lower]; // No suffix re-attachment
        }
        if (classification) {
            const first = classification.split(/[/\s,]+/)[0];
            if (first && first.length > 2) {
                return `${baseName} ${first}`; // No suffix re-attachment
            }
        }
    }
    return baseName;
}

// ─────────────────────────────────────────────
// FILE HANDLERS
// ─────────────────────────────────────────────

// ── animals.json / exhibits / enclosures ──
async function processAnimals(file, data, collection, ollama, model) {
    const items = Array.isArray(data) ? data : data.data || [data];
    for (const animal of items) {
        try {
            const rawName =
                en(animal.common_name) ||
                en(animal.render_name) ||
                en(animal.name) ||
                animal.name ||
                en(animal.title) ||
                animal.title || '';

            if (!rawName || /^[0-9a-fA-F]{24}$/.test(rawName) ||
                rawName.toLowerCase().includes('waypoint') ||
                rawName.toLowerCase().includes('unknown')) continue;

            const scientificName = en(animal.scientific_name) || en(animal.species) || '';
            const category = en(animal.category) || '';
            const classification = en(animal.classification) || '';
            const cleanName = enrichName(rawName, classification);

            const habitat = en(animal.habitat) || '';
            const diet = en(animal.diet) || '';
            const distribution = en(animal.distribution) || '';
            const activity = en(animal.activity) || '';
            const lifespan = en(animal.lifespan?.average) || '';
            const weight = en(animal.physical?.weight) || '';
            const length = en(animal.physical?.length) || '';
            const likes = en(animal.likes) || '';
            const dislikes = en(animal.dislikes) || '';
            const locationName = en(animal.location?.location_name) || en(animal.location_name) || '';
            const beatNumber = en(animal.location?.beat_number) || '';
            const date = animal.date || animal.event_date || '';
            const time = animal.time || animal.event_time || '';
            const threatStatus = en(animal.threat_status) || en(animal.conservation?.iucn_status) || '';
            const legalProtection = en(animal.conservation?.legal_protection) || '';
            const mutationNotes = en(animal.conservation?.notes) || '';

            const narrative =
                en(animal.narrative) || en(animal.description) || en(animal.text) || '';
            const storyDescription = en(animal.story_description) || '';

            const funFacts = Array.isArray(animal.fun_facts)
                ? animal.fun_facts.map(f => typeof f === 'object' ? f.en || '' : f).filter(Boolean).join('. ')
                : (animal.fun_facts || '');

            const personalInfo = Array.isArray(animal.personalInfo)
                ? animal.personalInfo.map(p => `${en(p.name)}: ${en(p.about)}`).join('. ')
                : '';

            const synonyms = [];
            const lowerName = rawName.toLowerCase();
            if (lowerName.includes('peafowl')) synonyms.push('peacock', 'peahen');
            if (lowerName.includes('tiger')) synonyms.push('bagh'); // Removed 'sher'
            if (lowerName.includes('lion') && !lowerName.includes('macaque')) {
                synonyms.push('sher', 'babbar sher'); // Added 'sher' to Lion
            }
            if (lowerName.includes('rhinoceros')) synonyms.push('rhino', 'genda');
            if (lowerName.includes('elephant')) synonyms.push('hathi');
            const isCalendarEvent = !animal.common_name && !animal.render_name && (animal.title || animal.name);
            let eventTitleVariants = '';
            let eventKeyword = '';
            if (isCalendarEvent) {
                eventKeyword = rawName.replace(/\b(national|international|world|global|day|for|to|the|of|and|in|a)\b/gi, '').trim();
                const corePart = rawName.replace(/\b(national|international|world|global)\b/gi, '').trim();
                eventTitleVariants = [`National ${corePart}`, `International ${corePart}`, `World ${corePart}`, corePart].join('. ');
            }

            const details = [`Animal/Subject Name: ${rawName}`];
            if (synonyms.length > 0) details.push(`Synonyms: ${synonyms.join(', ')}`);
            if (isCalendarEvent && eventTitleVariants) details.push(`Event Title Variants: ${eventTitleVariants}`);
            if (isCalendarEvent && eventKeyword) details.push(`Core Subject: ${eventKeyword}`);
            if (scientificName) details.push(`Scientific Name: ${scientificName}`);
            if (category) details.push(`Category: ${category}`);
            if (classification) details.push(`Classification: ${classification}`);
            if (habitat) details.push(`Habitat: ${habitat}`);
            if (distribution) details.push(`Distribution: ${distribution}`);
            if (diet) details.push(`Diet: ${diet}`);
            if (activity) details.push(`Activity Pattern: ${activity}`);
            if (date || time) details.push(`Date/Time: ${date} ${time}`.trim());
            if (lifespan) details.push(`Lifespan: ${lifespan}`);
            if (weight) details.push(`Weight: ${weight}`);
            if (length) details.push(`Length: ${length}`);
            if (threatStatus) details.push(`Threat Status: ${threatStatus}`);
            if (legalProtection) details.push(`Legal Protection: ${legalProtection}`);
            if (mutationNotes) details.push(`Genetic Traits/Notes: ${mutationNotes}`);
            if (likes) details.push(`Likes: ${likes}`);
            if (dislikes) details.push(`Dislikes: ${dislikes}`);
            if (locationName) details.push(`Location: ${locationName}`);
            if (beatNumber) details.push(`Beat Number: ${beatNumber}`);
            if (narrative) details.push(`Description: ${narrative}`);
            if (storyDescription) details.push(`Story: ${storyDescription}`);
            if (funFacts) details.push(`Fun Facts: ${funFacts}`);
            if (personalInfo) details.push(`Individual Info: ${personalInfo}`);

            console.log(`- Embedding: ${cleanName}`);
            await store(collection, ollama, model, {
                id: cleanName,
                label: rawName,
                text: details.join('\n'),
                metadata: {
                    name: cleanName,
                    common_name: en(animal.common_name),
                    render_name: en(animal.render_name),
                    scientific_name: scientificName,
                    category,
                    classification,
                    habitat,
                    diet,
                    distribution,
                    activity,
                    threat_status: threatStatus,
                    legal_protection: legalProtection,
                    location: locationName,
                    beat_number: beatNumber,
                    lifespan,
                    weight,
                    length,
                    is_event: isCalendarEvent ? 'true' : 'false',
                    file_source: file,
                    full_data: JSON.stringify(animal)
                }
            });

        } catch (err) {
            console.error(`  [ERR] Animal item failed: ${err.message}`);
        }
    }
}

// ── facts.json ──
async function processFacts(file, data, collection, ollama, model) {
    const items = Array.isArray(data) ? data : (data.data || [data]);
    for (const item of items) {
        try {
            const title = en(item.title) || '';
            const text = en(item.text) || '';
            if (!title && !text) continue;
            const id = `fact_${item._id || title.slice(0, 40).replace(/\s+/g, '_')}`;
            const label = title || text.slice(0, 60);
            const doc = join([
                title ? `Fact Title: ${title}` : '',
                text ? `Fact: ${text}` : ''
            ]);
            console.log(`- Embedding fact: ${label.slice(0, 60)}`);
            await store(collection, ollama, model, {
                id, label, text: doc,
                metadata: { file_source: file, type: 'fact', title, full_data: JSON.stringify(item) }
            });
        } catch (err) {
            console.error(`  [ERR] Fact failed: ${err.message}`);
        }
    }
}

// ── fees.json ──
async function processFees(file, data, collection, ollama, model) {
    // data is { _id, sections: [{ title, rows: [{ label, value }] }] }
    const records = Array.isArray(data) ? data : (data.data ? (Array.isArray(data.data) ? data.data : [data.data]) : [data]);
    for (const record of records) {
        const sections = record.sections || [];
        for (const section of sections) {
            try {
                const sectionTitle = en(section.title) || 'Fee Section';
                const rows = section.rows || [];
                const rowLines = rows.map(r => `  ${en(r.label)}: ${en(r.value)}`).join('\n');
                const doc = join([
                    `Fee Category: ${sectionTitle}`,
                    rowLines ? `Pricing:\n${rowLines}` : ''
                ]);
                if (!doc.trim()) continue;
                const id = `fee_${section._id || sectionTitle.replace(/\s+/g, '_')}`;
                console.log(`- Embedding fee: ${sectionTitle}`);
                await store(collection, ollama, model, {
                    id, label: sectionTitle, text: doc,
                    metadata: { file_source: file, type: 'fee', section_title: sectionTitle, full_data: JSON.stringify(section) }
                });
            } catch (err) {
                console.error(`  [ERR] Fee section failed: ${err.message}`);
            }
        }
    }
}

// ── news.json ──
async function processNews(file, data, collection, ollama, model) {
    const items = Array.isArray(data) ? data : (data.data || [data]);
    for (const item of items) {
        try {
            const title = en(item.title) || '';
            const text = en(item.text) || en(item.description) || '';
            const url = item.url || '';
            if (!title && !text) continue;
            const id = `news_${item._id || title.slice(0, 40).replace(/\s+/g, '_')}`;
            const label = title || text.slice(0, 60);
            const doc = join([
                title ? `News Title: ${title}` : '',
                text ? `Content: ${text}` : '',
                url ? `URL: ${url}` : ''
            ]);
            console.log(`- Embedding news: ${label.slice(0, 60)}`);
            await store(collection, ollama, model, {
                id, label, text: doc,
                metadata: { file_source: file, type: 'news', title, url, full_data: JSON.stringify(item) }
            });
        } catch (err) {
            console.error(`  [ERR] News item failed: ${err.message}`);
        }
    }
}

// ── rules.json ──
async function processRules(file, data, collection, ollama, model) {
    const records = Array.isArray(data) ? data : (data.data ? (Array.isArray(data.data) ? data.data : [data.data]) : [data]);
    for (const record of records) {
        const ruleSections = record.rules || [];
        for (const section of ruleSections) {
            try {
                const sectionTitle = en(section.title) || 'Zoo Rule';
                const sectionDesc = en(section.description) || '';
                const points = section.points || [];
                const pointLines = points.map(p => `  - ${en(p.title)}${en(p.description) ? ': ' + en(p.description) : ''}`).join('\n');
                const doc = join([
                    `Rule Category: ${sectionTitle}`,
                    sectionDesc ? `Description: ${sectionDesc}` : '',
                    pointLines ? `Rules:\n${pointLines}` : ''
                ]);
                if (!doc.trim()) continue;
                const id = `rule_${section._id || sectionTitle.replace(/\s+/g, '_')}`;
                console.log(`- Embedding rule: ${sectionTitle}`);
                await store(collection, ollama, model, {
                    id, label: sectionTitle, text: doc,
                    metadata: { file_source: file, type: 'rule', section_title: sectionTitle, full_data: JSON.stringify(section) }
                });
            } catch (err) {
                console.error(`  [ERR] Rule section failed: ${err.message}`);
            }
        }
    }
}

// ── about.json ──
async function processAbout(file, data, collection, ollama, model) {
    const obj = data.data || data;
    try {
        // 1. Content blocks
        const contentBlocks = (obj.content || [])
            .filter(c => c.type === 'text' && en(c.text).trim())
            .map(c => en(c.text));

        if (contentBlocks.length > 0) {
            const doc = join([
                'About the Zoo:',
                contentBlocks.join('\n\n')
            ]);
            console.log(`- Embedding: About Zoo (content)`);
            await store(collection, ollama, model, {
                id: 'about_zoo_content',
                label: 'About Zoo',
                text: doc,
                metadata: { file_source: file, type: 'about', full_data: JSON.stringify(obj.content) }
            });
        }

        // 2. Key Facts
        const keyFacts = obj.keyFacts || [];
        if (keyFacts.length > 0) {
            const factLines = keyFacts.map(f => `${en(f.label)}: ${en(f.value)}`).join('\n');
            const doc = `Zoo Key Facts:\n${factLines}`;
            console.log(`- Embedding: Zoo Key Facts`);
            await store(collection, ollama, model, {
                id: 'about_zoo_keyfacts',
                label: 'Zoo Key Facts',
                text: doc,
                metadata: { file_source: file, type: 'key_facts', full_data: JSON.stringify(keyFacts) }
            });
        }

    } catch (err) {
        console.error(`  [ERR] About processing failed: ${err.message}`);
    }
}

// ── contact.json ──
async function processContact(file, data, collection, ollama, model) {
    const obj = data.data || data;
    try {
        // Contact methods
        const methods = obj.contactMethods || [];
        if (methods.length > 0) {
            const lines = methods.map(m => `${en(m.label)}: ${en(m.value)}${m.actionUrl ? ` (${m.actionUrl})` : ''}`).join('\n');
            const doc = `Zoo Contact Information:\n${lines}`;
            console.log(`- Embedding: Zoo Contact Info`);
            await store(collection, ollama, model, {
                id: 'contact_info',
                label: 'Zoo Contact Info',
                text: doc,
                metadata: { file_source: file, type: 'contact', full_data: JSON.stringify(methods) }
            });
        }

        // Social links
        const social = obj.socialLinks || {};
        const socialEntries = Object.entries(social).filter(([, v]) => v);
        if (socialEntries.length > 0) {
            const lines = socialEntries.map(([k, v]) => `${k}: ${v}`).join('\n');
            const doc = `Zoo Social Media Links:\n${lines}`;
            console.log(`- Embedding: Zoo Social Links`);
            await store(collection, ollama, model, {
                id: 'contact_social',
                label: 'Zoo Social Links',
                text: doc,
                metadata: { file_source: file, type: 'social_links', full_data: JSON.stringify(social) }
            });
        }

        // Ticket booking
        const ticket = obj.ticket || {};
        if (ticket.bookingUrl) {
            const doc = `Zoo Ticket Booking:\nBooking URL: ${ticket.bookingUrl}`;
            console.log(`- Embedding: Ticket Booking`);
            await store(collection, ollama, model, {
                id: 'contact_tickets',
                label: 'Ticket Booking',
                text: doc,
                metadata: { file_source: file, type: 'tickets', booking_url: ticket.bookingUrl }
            });
        }

        // Address & name from root object
        if (obj.address || obj.name) {
            const doc = join([
                obj.name ? `Zoo Name: ${obj.name}` : '',
                obj.address ? `Address: ${obj.address}` : '',
                obj.venueName ? `Venue: ${obj.venueName}` : ''
            ]);
            if (doc.trim()) {
                console.log(`- Embedding: Zoo Address & Name`);
                await store(collection, ollama, model, {
                    id: 'contact_address',
                    label: 'Zoo Address',
                    text: doc,
                    metadata: { file_source: file, type: 'address' }
                });
            }
        }

    } catch (err) {
        console.error(`  [ERR] Contact processing failed: ${err.message}`);
    }
}

// ── zootime.json ──
async function processZooTime(file, data, collection, ollama, model) {
    const obj = data.data || data;
    try {
        // Timings per day
        const timings = obj.timings || [];
        if (timings.length > 0) {
            const lines = timings.map(t =>
                `${t.day}: ${t.isOpen ? `${t.openTime} – ${t.closeTime}` : 'Closed'}`
            ).join('\n');
            const doc = `Zoo Opening Hours:\n${lines}`;
            console.log(`- Embedding: Zoo Timings`);
            await store(collection, ollama, model, {
                id: 'zootime_timings',
                label: 'Zoo Timings / Opening Hours',
                text: doc,
                metadata: { file_source: file, type: 'timings', full_data: JSON.stringify(timings) }
            });
        }

        // Special closures / holidays
        const holidays = obj.holidays || obj.closedDays || [];
        if (holidays.length > 0) {
            const lines = holidays.map(h => `- ${en(h.name) || h}`).join('\n');
            const doc = `Zoo Holidays / Closure Days:\n${lines}`;
            console.log(`- Embedding: Zoo Holidays`);
            await store(collection, ollama, model, {
                id: 'zootime_holidays',
                label: 'Zoo Holidays',
                text: doc,
                metadata: { file_source: file, type: 'holidays', full_data: JSON.stringify(holidays) }
            });
        }

        // Any extra top-level timing info (closedOn, lastEntry, etc.)
        const extras = [];
        if (obj.closedOn) extras.push(`Closed on: ${obj.closedOn}`);
        if (obj.lastEntry) extras.push(`Last entry: ${obj.lastEntry}`);
        if (obj.note || obj.notes) extras.push(`Note: ${en(obj.note || obj.notes)}`);
        if (extras.length > 0) {
            const doc = `Zoo Timing Notes:\n${extras.join('\n')}`;
            console.log(`- Embedding: Zoo Timing Notes`);
            await store(collection, ollama, model, {
                id: 'zootime_notes',
                label: 'Zoo Timing Notes',
                text: doc,
                metadata: { file_source: file, type: 'timing_notes' }
            });
        }

    } catch (err) {
        console.error(`  [ERR] ZooTime processing failed: ${err.message}`);
    }
}

// ─────────────────────────────────────────────
// MAIN INGESTION
// ─────────────────────────────────────────────
async function ingest() {
    const chroma = new ChromaClient({ path: 'http://localhost:8000' });

    const embedder = new OllamaEmbeddingFunction({
        url: 'http://127.0.0.1:11434',
        model: 'nomic-embed-text'
    });

    const ollama = new Ollama();
    const embedModel = 'nomic-embed-text';
    const collectionName = 'zoo_collection';
    const dataDir = path.join(__dirname, 'zoo-data');

    console.log('--- Shera AI: Enhanced Chroma Ingestion Started ---');

    try {
        // Wipe and recreate collection
        try {
            await chroma.deleteCollection({ name: collectionName });
            console.log(`Deleted existing collection: ${collectionName}`);
        } catch (_) { /* didn't exist */ }

        const collection = await chroma.createCollection({
            name: collectionName,
            embeddingFunction: embedder,
            metadata: { "hnsw:space": "cosine" }
        });
        console.log(`Created fresh ChromaDB collection: ${collectionName}`);

        if (!fs.existsSync(dataDir)) {
            console.error(`Zoo data directory missing: ${dataDir}`);
            return;
        }

        const files = fs.readdirSync(dataDir).filter(f => f.endsWith('.json'));
        console.log(`Found ${files.length} data files.\n`);

        for (const file of files) {
            console.log(`Processing: ${file}`);

            // Skip only the heavy geometry file – it contains lat/lng coords, not textual knowledge
            if (file.includes('geojson') || file.includes('floorplan')) {
                console.log(`  Skipping geometry/map file: ${file}\n`);
                continue;
            }

            const filePath = path.join(dataDir, file);

            try {
                const raw = fs.readFileSync(filePath, 'utf8');
                const parsed = JSON.parse(raw);

                // Route to the right handler
                if (file === 'contact.json') {
                    await processContact(file, parsed, collection, ollama, embedModel);

                } else if (file === 'zootime.json') {
                    await processZooTime(file, parsed, collection, ollama, embedModel);

                } else if (file === 'about.json') {
                    await processAbout(file, parsed, collection, ollama, embedModel);

                } else if (file === 'fees.json') {
                    await processFees(file, parsed, collection, ollama, embedModel);

                } else if (file === 'rules.json') {
                    await processRules(file, parsed, collection, ollama, embedModel);

                } else if (file === 'news.json') {
                    await processNews(file, parsed, collection, ollama, embedModel);

                } else if (file === 'facts.json') {
                    const items = Array.isArray(parsed) ? parsed : (parsed.data || [parsed]);
                    await processFacts(file, items, collection, ollama, embedModel);

                } else {
                    // Generic handler: animals, calendar, events, tour, etc.
                    await processAnimals(file, parsed, collection, ollama, embedModel);
                }

            } catch (fileErr) {
                console.error(`[ERR] Failed file ${file}: ${fileErr.message}`);
            }

            console.log('');
        }

        console.log('--- Enhanced Ingestion Completed Successfully ---');

    } catch (globalErr) {
        console.error('Critical Error:', globalErr.message);
    }
}

ingest();