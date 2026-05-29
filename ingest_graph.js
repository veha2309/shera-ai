const fs = require('fs');
const path = require('path');

/**
 * Antigravity GraphRAG Multi-Source Ingestion Script - OPTIMIZED FOR GraphRAG
 * Ensures 100% data retention, no nulls, and dense semantic descriptions.
 */

const DATA_DIR = path.join(__dirname, 'zoo-data');
const STORE_DIR = path.join(__dirname, 'graph_data');
const animalNodeIds = new Set();
const animalBaseNames = new Map();
const ITEM_LIMITS = {};

async function asyncPool(limit, array, fn) {
    const ret = [];
    const executing = [];
    for (const item of array) {
        const p = Promise.resolve().then(() => fn(item));
        ret.push(p);
        if (limit <= array.length) {
            const e = p.then(() => executing.splice(executing.indexOf(e), 1));
            executing.push(e);
            if (executing.length >= limit) await Promise.race(executing);
        }
    }
    return Promise.all(ret);
}

// Helper: Safely extract nested .en strings from ANY object structure
function getVal(obj) {
    if (obj === null || obj === undefined) return '';
    if (typeof obj === 'string') return obj.trim();
    if (obj.en) return String(obj.en).trim();
    if (obj.value && obj.value.en) return String(obj.value.en).trim();
    return String(obj).trim(); // Fallback
}

// Helper: Split comma/and separated strings into discrete nodes
function splitList(str) {
    if (!str) return [];
    return str.split(/[,;]|\band\b|\bor\b/)
        .map(s => s.trim().replace(/\.$/, ''))
        .filter(s => s.length > 2);
}

const MANUAL_NAME_MAP = {
    "lion": "Asiatic Lion", "tiger": "White Tiger", "elephant": "Indian Elephant",
    "monkey": "Bonnet Macaque", "rhino": "Indian Rhinoceros", "rhinoceros": "Indian Rhinoceros"
};

function standardizeName(name, classification = '') {
    if (!name) return 'Unknown Animal';
    let baseName = name.replace(/\s+\d+$/, '').trim();
    if (!baseName.includes(' ')) {
        const lower = baseName.toLowerCase();
        if (MANUAL_NAME_MAP[lower]) return MANUAL_NAME_MAP[lower];
        if (classification) {
            const first = classification.split(/[/\s,]+/)[0];
            if (first && first.length > 2) return `${baseName} ${first}`;
        }
    }
    return baseName;
}

const escapeRegex = (str) => str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

function getSourceType(filename) {
    if (filename.includes('animal')) return 'BIOLOGY';
    if (filename.includes('tour') || filename.includes('geojson')) return 'GEOGRAPHY';
    if (filename.includes('calendar') || filename.includes('event')) return 'SCHEDULE';
    if (filename.includes('fee') || filename.includes('rules') || filename.includes('zootime')) return 'ADMIN';
    if (filename.includes('about') || filename.includes('contact')) return 'INFO';
    if (filename.includes('news') || filename.includes('facts')) return 'NEWS';
    return 'GENERAL';
}

function findActualAnimalNodeId(refName) {
    if (!refName || typeof refName !== 'string') return null;
    const cleanRef = refName.trim();
    if (animalNodeIds.has(cleanRef)) return cleanRef;
    const lowerRef = cleanRef.toLowerCase();
    if (animalBaseNames.has(lowerRef)) return animalBaseNames.get(lowerRef)[0];
    return null;
}

function flattenFile(file, items) {
    // Keep flatten logic similar, but utilize safe getVal
    if (file === 'about.json') {
        const out = [];
        for (const item of items) {
            if (item.content) item.content.forEach(b => {
                if (b.type === 'text' && b.text?.en) out.push({ type: 'about_block', text: b.text.en, parent: 'About Zoo', original: b });
            });
            if (item.keyFacts) item.keyFacts.forEach(f => {
                out.push({ type: 'key_fact', label: getVal(f.label), value: getVal(f.value), parent: 'About Zoo Key Facts', original: f });
            });
        }
        return out.length ? out : items;
    }
    if (file === 'tour.json') {
        const out = [];
        for (const tour of items) {
            if (tour.stops) tour.stops.forEach(stop => {
                const stopData = { ...stop.stopId };
                if (stopData.items) {
                    stopData.animals_present = stopData.items.map(i => getVal(i.animalId?.common_name) || getVal(i.location_name)).filter(Boolean);
                    delete stopData.items;
                }
                out.push({ ...stopData, parent_tour: getVal(tour.name), tour_description: getVal(tour.description), original: stop });
            });
        }
        return out.length ? out : items;
    }
    return items;
}

function extractProgrammatically(item, type, file) {
    const entities = [];
    const relationships = [];

    const rawNameString = (item.properties?.name || getVal(item.common_name) || getVal(item.render_name) || getVal(item.name) || getVal(item.title) || item.title || item.name || item.label || item.day || 'Item').toString().trim();

    if (file === 'animals.json') {
        const classification = getVal(item.classification);
        const animalName = standardizeName(rawNameString, classification);
        const scientificName = getVal(item.scientific_name);
        const category = getVal(item.category);
        const habitatStr = getVal(item.habitat);
        const dietStr = getVal(item.diet);
        const weight = getVal(item.physical?.weight);
        const length = getVal(item.physical?.length);
        const distributionStr = getVal(item.distribution);

        // Deep nested property lookups
        const conservation = getVal(item.conservation?.iucn_status) || getVal(item.conservation_status);
        const lifespan = getVal(item.lifespan?.average) || getVal(item.lifespan);
        const locationName = getVal(item.location?.location_name);
        const likes = getVal(item.likes);
        const dislikes = getVal(item.dislikes);

        // Massive Narrative fallback to ensure text is rich for GraphRAG
        const narrative = getVal(item.story_description) || getVal(item.narrative) || getVal(item.physical_description) || getVal(item.about_animal) || scientificName;

        // Extract Fun facts seamlessly
        let funFacts = "";
        if (item.fun_facts && Array.isArray(item.fun_facts)) {
            funFacts = item.fun_facts.map(f => getVal(f)).join(" ");
        }

        // 1. Core Animal Node - Semantically Rich for GraphRAG Embedding
        let fullDesc = `${animalName} (${scientificName}) is classified under ${category}. `;
        if (narrative) fullDesc += `Description: ${narrative}. `;
        if (distributionStr) fullDesc += `Found in: ${distributionStr}. `;
        if (likes) fullDesc += `Likes: ${likes}. `;
        if (dislikes) fullDesc += `Dislikes: ${dislikes}. `;
        if (funFacts) fullDesc += `Fun facts: ${funFacts} `;
        if (weight) fullDesc += `Weight: ${weight}. `;
        if (length) fullDesc += `Length: ${length}. `;
        entities.push({
            id: animalName,
            type: 'Animal',
            description: fullDesc.trim(),
            properties: item // PRESERVES 100% OF ORIGINAL DATA
        });

        // 2. Extracted Sub-Nodes (Habitats, Diets, Regions, Categories)
        if (habitatStr) splitList(habitatStr).forEach(h => {
            const capH = h.charAt(0).toUpperCase() + h.slice(1);
            entities.push({ id: capH, type: 'Habitat', description: `Habitat type: ${capH}`, properties: { name: capH } });
            relationships.push({ source: animalName, target: capH, type: 'LIVES_IN' });
        });

        if (dietStr) splitList(dietStr).forEach(d => {
            const capD = d.charAt(0).toUpperCase() + d.slice(1);
            entities.push({ id: capD, type: 'DietItem', description: `Diet item: ${capD}`, properties: { name: capD } });
            relationships.push({ source: animalName, target: capD, type: 'EATS' });
        });

        if (distributionStr) splitList(distributionStr).forEach(r => {
            const capR = r.charAt(0).toUpperCase() + r.slice(1);
            entities.push({ id: capR, type: 'Region', description: `Geographical Region: ${capR}`, properties: { name: capR } });
            relationships.push({ source: animalName, target: capR, type: 'FOUND_IN' });
        });

        if (category) {
            entities.push({ id: category, type: 'Classification', description: `Animal taxonomic class: ${category}`, properties: { name: category } });
            relationships.push({ source: animalName, target: category, type: 'IS_A' });
        }

        if (conservation) {
            entities.push({ id: conservation, type: 'ConservationStatus', description: `Conservation status: ${conservation}`, properties: { name: conservation } });
            relationships.push({ source: animalName, target: conservation, type: 'HAS_STATUS' });
        }

        if (lifespan) {
            entities.push({ id: lifespan, type: 'Lifespan', description: `Average lifespan: ${lifespan}`, properties: { name: lifespan } });
            relationships.push({ source: animalName, target: lifespan, type: 'HAS_LIFESPAN' });
        }

        if (locationName) {
            entities.push({ id: locationName, type: 'ZooLocation', description: `Zoo Enclosure/Location: ${locationName}`, properties: { name: locationName } });
            relationships.push({ source: animalName, target: locationName, type: 'HOUSED_AT' });
        }

        // 3. Extract Individual Animal Instances (e.g., specific names like "Siddhi")
        if (item.personalInfo && Array.isArray(item.personalInfo)) {
            item.personalInfo.forEach(person => {
                const pName = getVal(person.name);
                if (pName) {
                    const dob = person.dob ? person.dob.split('T')[0] : 'Unknown';
                    const pDesc = `${pName} is a specific ${animalName} living at the zoo. Gender: ${person.gender || 'Unknown'}. Date of birth: ${dob}.`;
                    entities.push({
                        id: `${pName} (${animalName})`,
                        type: 'AnimalInstance',
                        description: pDesc,
                        properties: person
                    });
                    relationships.push({ source: `${pName} (${animalName})`, target: animalName, type: 'INSTANCE_OF' });
                    if (locationName) relationships.push({ source: `${pName} (${animalName})`, target: locationName, type: 'LOCATED_AT' });
                }
            });
        }

    } else if (file === 'tour.json') {
        const stopName = getVal(item.name) || rawNameString;
        entities.push({
            id: `Tour Stop: ${stopName}`, type: 'Stop',
            description: `Stop in tour. Parent tour: ${item.parent_tour || ''}. Description: ${item.tour_description || ''}`,
            properties: item
        });
        if (item.animals_present && Array.isArray(item.animals_present)) {
            item.animals_present.forEach(animal => relationships.push({ source: `Tour Stop: ${stopName}`, target: animal.toString().trim(), type: 'CONTAINS_ANIMAL' }));
        }
    } else {
        // ULTIMATE FALLBACK: For ANY other file, stringify all readable fields into the GraphRAG description so zero data is lost.
        let safeProps = Object.entries(item)
            .filter(([k, v]) => k !== 'media' && k !== 'icon' && k !== 'sound' && v != null && v !== '')
            .map(([k, v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`)
            .join(' | ');

        entities.push({
            id: rawNameString,
            type: 'Entity',
            description: `Data Node. Extracted information: ${safeProps}`,
            properties: item
        });
    }

    return { entities, relationships };
}

async function buildCrossGraphEdges() {
    const storeFiles = fs.readdirSync(STORE_DIR).filter(f => f.endsWith('_graph.json'));
    const allGraphs = {};
    for (const f of storeFiles) {
        try { allGraphs[f] = JSON.parse(fs.readFileSync(path.join(STORE_DIR, f), 'utf8')); }
        catch (e) { }
    }
    const sortedAnimalBases = Array.from(animalBaseNames.keys()).sort((a, b) => b.length - a.length);

    for (const [filename, graph] of Object.entries(allGraphs)) {
        if (graph.metadata?.type === 'BIOLOGY') continue;
        let edgesAdded = 0;
        for (const node of graph.nodes) {
            const searchString = `${node.id} ${node.description || ''}`.toLowerCase();
            const matchedTargets = new Set();
            for (const base of sortedAnimalBases) {
                if (base.length < 3) continue;
                if (new RegExp(`\\b${escapeRegex(base)}\\b`, 'i').test(searchString)) {
                    const resolvedTarget = animalBaseNames.get(base)[0];
                    if (resolvedTarget && !matchedTargets.has(resolvedTarget)) {
                        matchedTargets.add(resolvedTarget);
                        if (!graph.edges.some(e => e.source === node.id && e.target === resolvedTarget)) {
                            const relType = graph.metadata?.type === 'GEOGRAPHY' ? 'HOUSED_IN' : (graph.metadata?.type === 'SCHEDULE' ? 'FEATURES_ANIMAL' : 'MENTIONS_ANIMAL');
                            graph.edges.push({ source: node.id, target: resolvedTarget, type: relType, cross_graph: 'animals_graph.json' });
                            edgesAdded++;
                        }
                    }
                }
            }
        }
        if (edgesAdded > 0) fs.writeFileSync(path.join(STORE_DIR, filename), JSON.stringify(graph, null, 2));
    }
}

async function processItem(item, graph, type, file) {
    const extraction = extractProgrammatically(item, type, file);
    for (const entity of extraction.entities || []) {
        if (!graph.nodes.find(n => n.id === entity.id)) {
            graph.nodes.push(entity);
            if (entity.type === 'Animal') {
                const trimmedId = entity.id.trim();
                animalNodeIds.add(trimmedId);
                const base = trimmedId.toLowerCase().replace(/\s+\d+$/, '').trim();
                if (!animalBaseNames.has(base)) animalBaseNames.set(base, []);
                animalBaseNames.get(base).push(trimmedId);
            }
        }
    }
    for (const rel of extraction.relationships || []) {
        if (['CONTAINS_ANIMAL', 'RELATED_TO_ANIMAL'].includes(rel.type)) {
            const resolvedTarget = findActualAnimalNodeId(rel.target);
            if (resolvedTarget) rel.target = resolvedTarget;
        }
        graph.edges.push(rel);
    }
}

async function main() {
    console.log('--- GraphRAG Optimized Ingestion Started ---');
    if (!fs.existsSync(STORE_DIR)) fs.mkdirSync(STORE_DIR, { recursive: true });

    const files = fs.readdirSync(DATA_DIR).filter(f => f.endsWith('.json'));
    files.sort((a, b) => a === 'animals.json' ? -1 : (b === 'animals.json' ? 1 : 0));

    for (const file of files) {
        const filePath = path.join(DATA_DIR, file);
        const graphName = file.replace('.json', '_graph.json');
        const type = getSourceType(file);

        const graph = { nodes: [], edges: [], metadata: { created_at: new Date().toISOString(), source_file: file, type } };
        try {
            const jsonData = JSON.parse(fs.readFileSync(filePath, 'utf8'));
            const items = Array.isArray(jsonData) ? jsonData : (jsonData.data ? (Array.isArray(jsonData.data) ? jsonData.data : [jsonData.data]) : [jsonData]);
            const itemsToProcess = flattenFile(file, items);

            await asyncPool(10, itemsToProcess, async (item) => await processItem(item, graph, type, file));
            fs.writeFileSync(path.join(STORE_DIR, graphName), JSON.stringify(graph, null, 2));
            console.log(`[OK] Saved ${graphName} (${graph.nodes.length} nodes, ${graph.edges.length} edges)`);
        } catch (err) {
            console.error(`[ERR] Failed to process ${file}:`, err.message);
        }
    }
    await buildCrossGraphEdges();
    console.log('\\n--- Ingestion Completed ---');
}

main();