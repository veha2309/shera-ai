const fs = require('fs');
const path = require('path');
const { Ollama } = require('ollama');

/**
 * Antigravity GraphRAG Multi-Source Ingestion Script
 * This script analyzes different zoo data types and creates specialized graphs for each.
 */

const ollama = new Ollama();
const DATA_DIR = path.join(__dirname, 'zoo-data');
const STORE_DIR = path.join(__dirname, 'antigravity_store');
const EMBED_MODEL = 'nomic-embed-text';
const EXTRACTION_MODEL = 'gemma2:2b';

// ✅ Add this entire function BEFORE main()
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

            // For very large files (like GeoJSON), we limit the items to avoid hanging
            const limit = file.includes('geojson') ? 50 : items.length;
            const itemsToProcess = items.slice(0, limit);

            for (const item of itemsToProcess) {
                await processItem(item, graph, graph.metadata.type);
            }

            // Save Individual Graph
            const graphPath = path.join(STORE_DIR, graphName);
            fs.writeFileSync(graphPath, JSON.stringify(graph, null, 2));
            console.log(`[OK] Saved ${graphName} (${graph.nodes.length} nodes, ${graph.edges.length} edges)`);

        } catch (err) {
            console.error(`[ERR] Failed to process ${file}:`, err.message);
        }
    }

    console.log('\n--- Multi-Source Ingestion Completed ---');

    await buildCrossGraphEdges();

    console.log('\n--- Cross-Graph Edge Building Completed ---');
}

function getSourceType(filename) {
    if (filename.includes('animal') || filename.includes('facts')) return 'BIOLOGY';
    if (filename.includes('event') || filename.includes('calendar')) return 'SCHEDULE';
    if (filename.includes('tour') || filename.includes('geojson')) return 'GEOGRAPHY';
    if (filename.includes('fee')) return 'ADMIN';
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
            entities: 'Location|PointOfInterest|Stop|Feature|Facility',
            relationships: 'PART_OF|CONNECTED_TO|NEARBY|HAS_FACILITY'
        },
        'ADMIN': {
            entities: 'Service|Price|Rule|Category|Policy',
            relationships: 'COSTS|APPLIES_TO|CATEGORIZED_AS|GOVERNED_BY'
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
    // 1. Prepare Text Representation
    const name = item.name?.en || item.common_name?.en || item.title?.en || item.title || item.name || item._id || 'Item';

    // Create a more readable text block for the LLM to process
    let text = `Name/Title: ${name}\n`;

    if (type === 'BIOLOGY') {
        text += `Scientific Name: ${item.scientific_name?.en || ''}\n`;
        text += `Category: ${item.category?.en || ''}\n`;
        text += `Classification: ${item.classification || ''}\n`;
        text += `Habitat: ${item.habitat?.en || ''}\n`;
        text += `Diet: ${item.diet?.en || ''}\n`;
        text += `Conservation: ${item.conservation?.iucn_status?.en || ''} - ${item.conservation?.notes?.en || ''}\n`;
        text += `Lifespan: ${item.lifespan?.average?.en || ''}\n`;
        text += `Physical: ${item.physical?.length?.en || ''}, ${item.physical?.weight?.en || ''}\n`;
        text += `Reproduction: ${item.reproduction?.breeding_pattern?.en || ''} (Gestation: ${item.reproduction?.gestation?.en || ''})\n`;
        text += `Location: ${item.location?.location_name?.en || ''}\n`;
        if (item.personalInfo) {
            item.personalInfo.forEach(p => {
                text += `Individual: ${p.name?.en} (${p.about?.en})\n`;
            });
        }
    } else if (type === 'SCHEDULE') {
        const titleEn = item.title?.en || item.title || '';
        // Extract core subject from title: "International Sloth Day" -> "Sloth"
        const coreSubject = titleEn
            .replace(/\b(national|international|world|global|day|for|to|the|of|and|in|a)\b/gi, '')
            .trim();

        text += `Date: ${item.date || ''}\n`;
        text += `Event Title: ${titleEn}\n`;
        text += `Core Subject: ${coreSubject}\n`;
        text += `Title Variants: National ${coreSubject} Day, International ${coreSubject} Day, World ${coreSubject} Day\n`;
        text += `Description: ${item.body?.en || item.body || 'Conservation awareness event'}\n`;
        text += `Related Animal: ${coreSubject}\n`; // hint for LLM extraction
    } else {
        text += JSON.stringify(item);
    }

    console.log(`  - Extracting: ${name.toString().substring(0, 30)}...`);

    // 2. Extraction
    async function extractWithRetry(text, type, retries = 2) {
        for (let attempt = 0; attempt <= retries; attempt++) {
            try {
                const response = await ollama.generate({
                    model: EXTRACTION_MODEL,
                    prompt: getPromptForType(type, text),
                    format: 'json',
                    stream: false,
                    options: { temperature: 0.1 }
                });

                const extraction = JSON.parse(response.response);

                if (!Array.isArray(extraction.entities) || !Array.isArray(extraction.relationships)) {
                    throw new Error('Invalid structure');
                }
                return extraction;

            } catch (err) {
                if (attempt === retries) return { entities: [], relationships: [] };
                console.log(`    [RETRY ${attempt + 1}] ${err.message}`);
            }
        }
    }
    try {

        // Inside processItem, replace the try block:
        const extraction = await extractWithRetry(text, type);

        // 3. Update Local Graph
        for (const entity of extraction.entities || []) {
            if (!graph.nodes.find(n => n.id === entity.id)) {
                // Get Embedding for Node
                const embedResp = await ollama.embed({
                    model: EMBED_MODEL,
                    input: `${entity.id}: ${entity.description || entity.type}`
                });

                graph.nodes.push({
                    ...entity,
                    embedding: embedResp.embeddings[0]
                });
            }
        }

        for (const rel of extraction.relationships || []) {
            graph.edges.push(rel);
        }

    } catch (err) {
        // console.error(`    [SKIP] ${name}:`, err.message);
    }
}

main();
