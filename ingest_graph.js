const fs = require('fs');
const path = require('path');
const { Ollama } = require('ollama');

/**
 * Antigravity GraphRAG Multi-Source Ingestion Script
 * Optimized for speed, local LLM execution, and comprehensive zoo data structure support.
 */

const ollama = new Ollama();
const DATA_DIR = path.join(__dirname, 'zoo-data');
const STORE_DIR = path.join(__dirname, 'graph_data');
const EMBED_MODEL = 'nomic-embed-text';

// In-memory cache to avoid re-embedding the same entity multiple times
const embedCache = new Map();

// Limits to prevent oversized ingestion on low-resource environments
const ITEM_LIMITS = {
    'geojson.json': 50,
    'facts.json': 100,
};

/**
 * Simple Async Pool for concurrency control
 */
async function asyncPool(limit, array, fn) {
    const ret = [];
    const executing = [];
    for (const item of array) {
        const p = Promise.resolve().then(() => fn(item));
        ret.push(p);
        if (limit <= array.length) {
            const e = p.then(() => executing.splice(executing.indexOf(e), 1));
            executing.push(e);
            if (executing.length >= limit) {
                await Promise.race(executing);
            }
        }
    }
    return Promise.all(ret);
}

function getSourceType(filename) {
    if (filename.includes('animal')) return 'BIOLOGY';
    if (filename.includes('tour') || filename.includes('geojson')) return 'GEOGRAPHY';
    if (filename.includes('calendar') || filename.includes('event')) return 'SCHEDULE';
    if (filename.includes('fee') || filename.includes('rules') || filename.includes('zootime')) return 'ADMIN';
    if (filename.includes('about') || filename.includes('contact')) return 'INFO';
    if (filename.includes('news') || filename.includes('facts')) return 'NEWS';
    return 'GENERAL';
}

function flattenFile(file, items) {
    if (file === 'about.json') {
        const out = [];
        for (const item of items) {
            if (item.content && Array.isArray(item.content)) {
                for (const block of item.content) {
                    if (block.type === 'text' && block.text?.en?.trim()) {
                        out.push({
                            type: 'about_block',
                            text: block.text.en,
                            parent: 'About Zoo'
                        });
                    }
                }
            }
            if (item.keyFacts && Array.isArray(item.keyFacts)) {
                for (const fact of item.keyFacts) {
                    out.push({
                        type: 'key_fact',
                        label: fact.label?.en,
                        value: fact.value?.en,
                        icon: fact.icon,
                        parent: 'About Zoo Key Facts'
                    });
                }
            }
        }
        return out.length ? out : items;
    }

    if (file === 'contact.json') {
        const out = [];
        for (const item of items) {
            if (item.contactMethods && Array.isArray(item.contactMethods)) {
                for (const method of item.contactMethods) {
                    out.push({
                        type: 'contact_method',
                        label: method.label?.en,
                        value: method.value?.en,
                        actionUrl: method.actionUrl,
                        parent: 'Contact Information'
                    });
                }
            }
        }
        return out.length ? out : items;
    }

    if (file === 'fees.json') {
        const out = [];
        for (const item of items) {
            if (item.sections && Array.isArray(item.sections)) {
                for (const section of item.sections) {
                    out.push({
                        ...section,
                        parent_context: 'Zoo Fees & Charges'
                    });
                }
            }
        }
        return out.length ? out : items;
    }

    if (file === 'rules.json') {
        const out = [];
        for (const item of items) {
            if (item.rules && Array.isArray(item.rules)) {
                for (const rule of item.rules) {
                    if (rule.points && Array.isArray(rule.points)) {
                        for (const point of rule.points) {
                            out.push({
                                type: 'rule_point',
                                category: rule.title?.en,
                                title: point.title?.en,
                                description: point.description?.en,
                                icon: point.icon,
                                parent_context: 'Zoo Rules'
                            });
                        }
                    } else {
                        out.push({
                            ...rule,
                            parent_context: 'Zoo Rules'
                        });
                    }
                }
            }
        }
        return out.length ? out : items;
    }

    if (file === 'tour.json') {
        const out = [];
        for (const tour of items) {
            if (tour.stops && Array.isArray(tour.stops)) {
                for (const stop of tour.stops) {
                    const stopData = { ...stop.stopId };
                    if (stopData.items && Array.isArray(stopData.items)) {
                        stopData.animals_present = stopData.items
                            .map(i => i.animalId?.common_name?.en || i.location_name?.en)
                            .filter(Boolean);
                        delete stopData.items; 
                    }
                    out.push({
                        ...stopData,
                        parent_tour: tour.name?.en,
                        tour_description: tour.description?.en
                    });
                }
            }
        }
        return out.length ? out : items;
    }

    if (file === 'zootime.json') {
        const out = [];
        for (const item of items) {
            if (item.timings && Array.isArray(item.timings)) {
                for (const timing of item.timings) {
                    out.push({
                        type: 'zoo_timing',
                        day: timing.day,
                        openTime: timing.openTime,
                        closeTime: timing.closeTime,
                        isOpen: timing.isOpen
                    });
                }
            }
            if (item.ticket) {
                out.push({
                    type: 'ticket_info',
                    bookingUrl: item.ticket.bookingUrl
                });
            }
            if (item.socialLinks) {
                out.push({
                    type: 'social_links',
                    ...item.socialLinks
                });
            }
        }
        return out.length ? out : items;
    }

    if (file === 'calendar.json') {
        return items.map(ev => ({
            type: 'calendar_event',
            title: ev.title?.en || ev.title,
            body: ev.body?.en || ev.body,
            date: ev.date
        }));
    }

    if (file === 'events.json') {
        return items.map(ev => ({
            type: 'event',
            name: ev.name?.en || ev.name,
            description: ev.description?.en || ev.description,
            from_datetime: ev.from_datetime,
            to_datetime: ev.to_datetime,
            location_name: ev.location_name?.en
        }));
    }

    if (file === 'news.json') {
        return items.map(n => ({
            type: 'news_item',
            title: n.title?.en || n.title,
            text: n.text?.en || n.text
        }));
    }

    if (file === 'facts.json') {
        return items.map(f => ({
            type: 'fact',
            title: f.title?.en || f.title,
            text: f.text?.en || f.text
        }));
    }

    return items;
}

function extractProgrammatically(item, type, file) {
    const entities = [];
    const relationships = [];

    const name = item.properties?.name || item.common_name?.en || item.render_name?.en || item.name?.en || item.title?.en || item.title || item.name || item.label || item.day || 'Item';
    
    if (file === 'animals.json') {
        const animalName = item.render_name?.en || item.common_name?.en || name;
        const scientificName = item.scientific_name?.en || '';
        const category = item.category?.en || '';
        const diet = item.diet?.en || '';
        const habitat = item.habitat?.en || '';
        const conservation = item.conservation_status?.en || '';
        const lifespan = item.lifespan?.en || '';
        const desc = item.physical_description?.en || item.about_animal?.en || scientificName;

        entities.push({
            id: animalName,
            type: 'Animal',
            description: `${animalName} (${scientificName}) is a ${category}. ${desc}`
        });

        if (habitat) {
            entities.push({ id: habitat, type: 'Habitat', description: `Habitat of animals: ${habitat}` });
            relationships.push({ source: animalName, target: habitat, type: 'LIVES_IN' });
        }
        if (diet) {
            entities.push({ id: diet, type: 'Diet', description: `Diet: ${diet}` });
            relationships.push({ source: animalName, target: diet, type: 'EATS' });
        }
        if (category) {
            entities.push({ id: category, type: 'Classification', description: `Animal class: ${category}` });
            relationships.push({ source: animalName, target: category, type: 'IS_A' });
        }
        if (conservation) {
            entities.push({ id: conservation, type: 'Conservation', description: `Conservation status: ${conservation}` });
            relationships.push({ source: animalName, target: conservation, type: 'STATUS_IS' });
        }
        if (lifespan) {
            entities.push({ id: lifespan, type: 'Lifespan', description: `Lifespan of ${animalName}: ${lifespan}` });
            relationships.push({ source: animalName, target: lifespan, type: 'LIVES_FOR' });
        }
    } else if (file === 'rules.json') {
        const title = item.title || name;
        const desc = item.description || 'Zoo rule/policy';
        entities.push({
            id: `Rule: ${title}`,
            type: 'Rule',
            description: `${title}. Category: ${item.category || ''}. Description: ${desc}`
        });
    } else if (file === 'fees.json') {
        const title = item.title?.en || item.title || name;
        const rate = item.rate || item.value || '';
        entities.push({
            id: `Fee: ${title}`,
            type: 'Price',
            description: `Fee/charge for ${title}. Price details: ${rate}`
        });
    } else if (file === 'tour.json') {
        const stopName = item.name?.en || item.name || name;
        entities.push({
            id: `Tour Stop: ${stopName}`,
            type: 'Stop',
            description: `Stop in tour. Parent tour: ${item.parent_tour || ''}. Description: ${item.tour_description || ''}`
        });
        if (item.animals_present && Array.isArray(item.animals_present)) {
            for (const animal of item.animals_present) {
                relationships.push({ source: `Tour Stop: ${stopName}`, target: animal, type: 'CONTAINS_ANIMAL' });
            }
        }
    } else if (file === 'zootime.json') {
        if (item.type === 'zoo_timing') {
            entities.push({
                id: `Timing: ${item.day}`,
                type: 'Timing',
                description: `Zoo opening hours for ${item.day}: Open ${item.openTime}, Close ${item.closeTime}. Status: ${item.isOpen ? 'Open' : 'Closed'}`
            });
        } else if (item.type === 'ticket_info') {
            entities.push({
                id: `Ticket Booking URL`,
                type: 'Service',
                description: `Official ticket booking page: ${item.bookingUrl}`
            });
        } else if (item.type === 'social_links') {
            entities.push({
                id: `Zoo Social Media`,
                type: 'SocialMedia',
                description: `Official social media links: Youtube: ${item.youtube}, Facebook: ${item.facebook}, Instagram: ${item.instagram}, Twitter: ${item.twitter}`
            });
        }
    } else if (file === 'about.json') {
        if (item.type === 'about_block') {
            entities.push({
                id: `About Zoo Block`,
                type: 'History',
                description: item.text
            });
        } else if (item.type === 'key_fact') {
            entities.push({
                id: `Key Fact: ${item.label}`,
                type: 'Fact',
                description: `${item.label}: ${item.value}`
            });
        }
    } else if (file === 'contact.json') {
        entities.push({
            id: `Contact: ${item.label}`,
            type: 'ContactMethod',
            description: `${item.label}: ${item.value}. Action: ${item.actionUrl || ''}`
        });
    } else if (file === 'calendar.json') {
        const title = item.title || name;
        entities.push({
            id: `Calendar Event: ${title}`,
            type: 'Event',
            description: `${title}. Date: ${item.date || ''}. Details: ${item.body || ''}`
        });
    } else if (file === 'events.json') {
        const title = item.name || name;
        entities.push({
            id: `Event: ${title}`,
            type: 'Event',
            description: `${title}. Location: ${item.location_name || ''}. Duration: ${item.from_datetime || ''} to ${item.to_datetime || ''}. Details: ${item.description || ''}`
        });
    } else if (file === 'news.json') {
        const title = item.title || name;
        entities.push({
            id: `News: ${title}`,
            type: 'Fact',
            description: `${title}. Details: ${item.text || ''}`
        });
    } else if (file === 'facts.json') {
        const title = item.title || name;
        entities.push({
            id: `Fact: ${title}`,
            type: 'Fact',
            description: `${title}. Details: ${item.text || ''}`
        });
    } else if (file === 'geojson.json') {
        const props = item.properties || {};
        const featName = props.name || name;
        entities.push({
            id: `Facility: ${featName}`,
            type: 'Facility',
            description: `Zoo facility or landmark. Name: ${featName}, Type: ${props.type || ''}, Floor: ${props.floor || 0}`
        });
    } else {
        entities.push({
            id: name,
            type: 'Entity',
            description: JSON.stringify(item)
        });
    }

    return { entities, relationships };
}

async function buildCrossGraphEdges() {
    console.log('\n--- Building Cross-Graph Edges ---');

    const storeFiles = fs.readdirSync(STORE_DIR).filter(f => f.endsWith('_graph.json'));

    const allGraphs = {};
    for (const f of storeFiles) {
        try {
            allGraphs[f] = JSON.parse(fs.readFileSync(path.join(STORE_DIR, f), 'utf8'));
        } catch (e) {
            console.error(`Failed to load ${f}:`, e.message);
        }
    }

    for (const [filename, graph] of Object.entries(allGraphs)) {
        if (graph.metadata?.type !== 'SCHEDULE') continue;

        let edgesAdded = 0;
        for (const node of graph.nodes) {
            const coreSubject = node.id
                .replace(/\b(national|international|world|global|day|for|to|the|of|and|in|a)\b/gi, '')
                .trim()
                .toLowerCase();

            if (!coreSubject) continue;

            for (const [bioFilename, bioGraph] of Object.entries(allGraphs)) {
                if (bioGraph.metadata?.type !== 'BIOLOGY') continue;

                for (const bioNode of bioGraph.nodes) {
                    const bioLower = bioNode.id.toLowerCase();
                    if (bioLower.includes(coreSubject) || coreSubject.includes(bioLower)) {
                        graph.edges.push({
                            source: node.id,
                            target: bioNode.id,
                            type: 'RELATED_TO_ANIMAL',
                            cross_graph: bioFilename
                        });
                        edgesAdded++;
                        console.log(`  [CROSS] "${node.id}" -> RELATED_TO_ANIMAL -> "${bioNode.id}"`);
                    }
                }
            }
        }

        if (edgesAdded > 0) {
            fs.writeFileSync(path.join(STORE_DIR, filename), JSON.stringify(graph, null, 2));
            console.log(`  [SAVED] ${filename} with ${edgesAdded} new cross-edges`);
        }
    }
}

async function processItem(item, graph, type, file) {
    const extraction = extractProgrammatically(item, type, file);

    for (const entity of extraction.entities || []) {
        const exists = graph.nodes.find(n => n.id === entity.id);
        if (!exists) {
            const cacheKey = `${entity.id}:${entity.description || entity.type}`;
            let embedding;

            if (embedCache.has(cacheKey)) {
                embedding = embedCache.get(cacheKey);
            } else {
                try {
                    const embedResp = await ollama.embed({
                        model: EMBED_MODEL,
                        input: cacheKey,
                        keep_alive: '1h'
                    });
                    embedding = embedResp.embeddings[0];
                    embedCache.set(cacheKey, embedding);
                } catch (e) {
                    console.error(`      [ERR] Embedding failed for ${entity.id}: ${e.message}`);
                    continue;
                }
            }

            graph.nodes.push({ ...entity, embedding });
        }
    }

    for (const rel of extraction.relationships || []) {
        graph.edges.push(rel);
    }
}

async function main() {
    console.log('--- Antigravity GraphRAG: Multi-Source Ingestion Started ---');

    if (!fs.existsSync(STORE_DIR)) {
        fs.mkdirSync(STORE_DIR, { recursive: true });
    }

    const files = fs.readdirSync(DATA_DIR).filter(f => f.endsWith('.json'));

    for (const file of files) {
        const filePath = path.join(DATA_DIR, file);
        const graphName = file.replace('.json', '_graph.json');
        const type = getSourceType(file);

        console.log(`\n>>> Processing Source: ${file} -> ${graphName}`);

        const graph = {
            nodes: [],
            edges: [],
            metadata: {
                created_at: new Date().toISOString(),
                source_file: file,
                type
            }
        };

        try {
            const content = fs.readFileSync(filePath, 'utf8');
            const jsonData = JSON.parse(content);
            
            // Robust parsing of array vs object structure
            const items = Array.isArray(jsonData) 
                ? jsonData 
                : (jsonData.data 
                    ? (Array.isArray(jsonData.data) ? jsonData.data : [jsonData.data]) 
                    : [jsonData]);

            let itemsToProcess = flattenFile(file, items);

            const limit = ITEM_LIMITS[file];
            if (limit && itemsToProcess.length > limit) {
                console.log(`  [LIMIT] Capping at ${limit} of ${itemsToProcess.length} items`);
                itemsToProcess = itemsToProcess.slice(0, limit);
            }

            const CONCURRENCY = 5;
            console.log(`  [POOL] Processing ${itemsToProcess.length} items with concurrency ${CONCURRENCY}...`);

            await asyncPool(CONCURRENCY, itemsToProcess, async (item) => {
                const name = item.properties?.name || item.common_name?.en || item.render_name?.en || item.name?.en || item.title?.en || item.title || item.name || item.label || item.day || 'Item';
                console.log(`    - Processing: ${name.toString().substring(0, 30)}`);
                await processItem(item, graph, type, file);
            });

            const graphPath = path.join(STORE_DIR, graphName);
            fs.writeFileSync(graphPath, JSON.stringify(graph, null, 2));
            console.log(`[OK] Saved ${graphName} (${graph.nodes.length} nodes, ${graph.edges.length} edges)`);

        } catch (err) {
            console.error(`[ERR] Failed to process ${file}:`, err.message);
        }
    }

    console.log('\n--- Building cross-graph edges ---');
    await buildCrossGraphEdges();

    console.log('\n--- Multi-Source Ingestion Completed ---');
}

main();