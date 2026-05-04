const fs = require('fs');
const path = require('path');
const { Ollama } = require('ollama');

/**
 * Antigravity GraphRAG Multi-Source Ingestion Script
 * Optimized for speed and local LLM execution.
 */

const ollama = new Ollama();
const DATA_DIR = path.join(__dirname, 'zoo-data');
const STORE_DIR = path.join(__dirname, 'graph_data');
const EMBED_MODEL = 'nomic-embed-text';
const EXTRACTION_MODEL = 'gemma4:e2b';

// In-memory cache to avoid re-embedding the same entity multiple times
const embedCache = new Map();

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

// ... (buildCrossGraphEdges function remains unchanged, omitted for brevity but keep it in your code) ...
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

async function main() {
    console.log('--- Antigravity GraphRAG: Multi-Source Ingestion Started ---');

    if (!fs.existsSync(STORE_DIR)) {
        fs.mkdirSync(STORE_DIR);
    }

    const files = fs.readdirSync(DATA_DIR).filter(f => f.endsWith('.json'));

    for (const file of files) {
        if (file.includes('calendar') || file.includes('event') || file.includes('facts') || file.includes('fees') || file.includes('news')) {
            console.log(`\n>>> Skipping Source: ${file} (Commented out)`);
            continue;
        }

        const filePath = path.join(DATA_DIR, file);
        const graphName = file.replace('.json', '_graph.json');
        console.log(`\n>>> Processing Source: ${file} -> ${graphName}`);

        const graph = {
            nodes: [],
            edges: [],
            metadata: {
                created_at: new Date().toISOString(),
                source_file: file,
                type: getSourceType(file)
            }
        };

        try {
            const content = fs.readFileSync(filePath, 'utf8');
            const jsonData = JSON.parse(content);
            const items = Array.isArray(jsonData) ? jsonData : (jsonData.data || [jsonData]);

            const limit = file.includes('geojson') ? 50 : items.length;
            let itemsToProcess = items.slice(0, limit);

            if (file === 'fees.json') {
                const flattenedFees = [];
                for (const item of items) {
                    if (item.sections && Array.isArray(item.sections)) {
                        for (const section of item.sections) {
                            flattenedFees.push({
                                ...section,
                                parent_context: "Zoo Fees & Charges"
                            });
                        }
                    }
                }
                if (flattenedFees.length > 0) itemsToProcess = flattenedFees;
            }

            if (file === 'tour.json') {
                const flattenedTours = [];
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
                            flattenedTours.push({
                                ...stopData,
                                parent_tour: tour.name?.en,
                                tour_description: tour.description?.en
                            });
                        }
                    }
                }
                if (flattenedTours.length > 0) itemsToProcess = flattenedTours;
            }

            // 🚀 SPEED FIX: Dropped concurrency to 2 for stable local LLM throughput
            const CONCURRENCY = 2;
            console.log(`  [POOL] Processing ${itemsToProcess.length} items with concurrency ${CONCURRENCY}...`);

            await asyncPool(CONCURRENCY, itemsToProcess, async (item) => {
                await processItem(item, graph, graph.metadata.type);
            });

            const graphPath = path.join(STORE_DIR, graphName);
            fs.writeFileSync(graphPath, JSON.stringify(graph, null, 2));
            console.log(`[OK] Saved ${graphName} (${graph.nodes.length} nodes, ${graph.edges.length} edges)`);

        } catch (err) {
            console.error(`[ERR] Failed to process ${file}:`, err.message);
        }
    }

    console.log('\n--- Multi-Source Ingestion Completed ---');
    // await buildCrossGraphEdges();
}

function getSourceType(filename) {
    if (filename.includes('animal')) return 'BIOLOGY';
    if (filename.includes('tour') || filename.includes('geojson')) return 'GEOGRAPHY';
    if (filename.includes('fee')) return 'ADMIN';
    if (filename.includes('news')) return 'NEWS';
    return 'GENERAL';
}

function getPromptForType(type, text) {
    const schemas = {
        'BIOLOGY': {
            entities: 'Animal|Habitat|Diet|Species|Conservation|Reproduction|Trait|Lifespan|Individual|Classification|Location',
            relationships: 'LIVES_IN|EATS|IS_A|BELONGS_TO|STATUS_IS|REPRODUCES_BY|HAS_TRAIT|LIVES_FOR|HAS_INDIVIDUAL|CATEGORIZED_AS|LOCATED_AT'
        },
        'SCHEDULE': {
            entities: 'Event|Date|Location|Activity|Description',
            relationships: 'OCCURS_AT|HAPPENS_ON|INVOLVES|DESCRIBED_AS'
        },
        'GEOGRAPHY': {
            entities: 'Location|PointOfInterest|Stop|Feature|Facility|Exhibit',
            relationships: 'PART_OF|CONNECTED_TO|NEARBY|HAS_FACILITY|CONTAINS_ANIMAL'
        },
        'ADMIN': {
            entities: 'Service|Price|Rule|Category|Policy|TargetAudience',
            relationships: 'COSTS|APPLIES_TO|CATEGORIZED_AS|GOVERNED_BY|VALID_FOR'
        },
        'NEWS': {
            entities: 'Program|Opportunity|Requirement|Benefit|Contact|Policy',
            relationships: 'PART_OF|REQUIRES|OFFERS|CONTACT_AT|STATED_IN'
        },
        'GENERAL': {
            entities: 'Entity|Topic|Detail',
            relationships: 'RELATED_TO|DESCRIBES'
        }
    };

    const schema = schemas[type] || schemas['GENERAL'];

    return `
        Extract entities and relationships from the following ${type} data for a zoo knowledge graph.
        Return ONLY a JSON object with this structure:
        {
          "entities": [{"id": "Name", "type": "${schema.entities}", "description": "..."}],
          "relationships": [{"source": "ID", "target": "ID", "type": "${schema.relationships}"}]
        }

        Data:
        ${text}
    `;
}

async function processItem(item, graph, type) {
    const cleanedItem = JSON.parse(JSON.stringify(item));
    
    // 🚀 SPEED FIX: Strip Bulky Geometry
    if (cleanedItem.geometry && cleanedItem.geometry.coordinates) {
        cleanedItem.geometry.coordinates = "[COORDINATES_OMITTED]";
    }
    
    // 🚀 SPEED FIX: Strip URLs and Media to reduce prompt bloat
    delete cleanedItem.icon;
    delete cleanedItem.media;
    delete cleanedItem.sound;
    delete cleanedItem.model_3d;
    delete cleanedItem.createdAt;
    delete cleanedItem.updatedAt;
    delete cleanedItem.__v;

    // 🚀 SPEED FIX: Recursively strip Hindi translations to halve token count
    function removeHindi(obj) {
        for (let prop in obj) {
            if (prop === 'hi') {
                delete obj[prop];
            } else if (typeof obj[prop] === 'object' && obj[prop] !== null) {
                removeHindi(obj[prop]);
            }
        }
    }
    removeHindi(cleanedItem);
    
    const props = cleanedItem.properties || {};
    const name = props.render_name?.en || props.name || props.common_name?.en || 
                 cleanedItem.render_name?.en || cleanedItem.common_name?.en || 
                 cleanedItem.name?.en || cleanedItem.title?.en || 
                 cleanedItem.title || cleanedItem.name || cleanedItem._id || 'Item';
    
    let text = `Name/Title: ${name}\n`;
    text += JSON.stringify(cleanedItem);

    console.log(`    - Extracting: ${name.toString().substring(0, 30)} (Prompt size: ${text.length} chars)`);

    const extraction = await extractWithRetry(text, type);

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
                        keep_alive: '1h' // 🚀 Keeps model in memory
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

async function extractWithRetry(text, type, retries = 2) {
    for (let attempt = 0; attempt <= retries; attempt++) {
        try {
            const response = await ollama.generate({
                model: EXTRACTION_MODEL,
                prompt: getPromptForType(type, text),
                format: 'json',
                stream: false,
                keep_alive: '1h', // 🚀 Keeps generation model loaded
                options: { temperature: 0.1 }
            });

            const extraction = JSON.parse(response.response);
            if (!Array.isArray(extraction.entities) || !Array.isArray(extraction.relationships)) {
                throw new Error('Invalid structure');
            }
            return extraction;
        } catch (err) {
            if (attempt === retries) return { entities: [], relationships: [] };
            console.log(`      [RETRY ${attempt + 1}] ${err.message}`);
        }
    }
}

// Just one main() call!
main();