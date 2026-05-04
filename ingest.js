const fs = require('fs');
const path = require('path');
const { ChromaClient } = require('chromadb');
const { OllamaEmbeddingFunction } = require('@chroma-core/ollama');
const { Ollama } = require('ollama');

/**
 * Shera AI - Enhanced Chroma Ingestion Script
 * Hybrid-ready semantic ingestion for Zoo Data
 * 
 * Optimizations Applied:
 * 1. Dynamic Text Builder: Prevents embedding dilution by omitting empty fields.
 * 2. Keep-Alive: Keeps the embedding model hot in memory for faster loop processing.
 */

async function ingest() {
    const chroma = new ChromaClient({
        path: "http://localhost:8000"
    });

    const embedder = new OllamaEmbeddingFunction({
        url: "http://127.0.0.1:11434",
        model: "nomic-embed-text"
    });

    const ollama = new Ollama();
    const embedModel = "nomic-embed-text";

    const collectionName = 'zoo_collection';
    const dataDir = path.join(__dirname, 'zoo-data');

    console.log('--- Shera AI: Enhanced Chroma Ingestion Started ---');

    try {
        // ---------------------------
        // Connect to Chroma and Wipe
        // ---------------------------
        try {
            await chroma.deleteCollection({ name: collectionName });
            console.log(`Deleted existing collection: ${collectionName}`);
        } catch (e) {
            // Collection might not exist, that's fine
        }

        const collection = await chroma.createCollection({
            name: collectionName,
            embeddingFunction: embedder
        });

        console.log(`Created fresh ChromaDB collection: ${collectionName}`);

        if (!fs.existsSync(dataDir)) {
            console.error(`Zoo data directory missing: ${dataDir}`);
            return;
        }

        const files = fs.readdirSync(dataDir).filter(file =>
            file.endsWith('.json')
        );

        console.log(`Found ${files.length} data files.`);

        // ---------------------------
        // Process Files
        // ---------------------------
        for (const file of files) {
            // Skip map/geometry and redundant facts that pollute search results
            if (file.includes('geojson') || file.includes('floorplan') || file.includes('facts')) {
                console.log(`Skipping file: ${file}`);
                continue;
            }

            const filePath = path.join(dataDir, file);
            console.log(`\nProcessing: ${file}`);

            try {
                const raw = fs.readFileSync(filePath, 'utf8');
                const parsed = JSON.parse(raw);

                const animals = Array.isArray(parsed)
                    ? parsed
                    : parsed.data || [parsed];

                for (const animal of animals) {
                    try {
                        // ---------------------------
                        // Core Identity
                        // ---------------------------
                        const name =
                            animal.render_name?.en ||
                            animal.common_name?.en ||
                            animal.name?.en ||
                            animal.name ||
                            animal.title?.en ||
                            animal.title ||
                            '';

                        // Skip items that have no readable name, only a Hex ID, or are unknown system markers
                        if (!name || /^[0-9a-fA-F]{24}$/.test(name) || name.toLowerCase().includes('waypoint') || name.toLowerCase().includes('unknown')) {
                            continue;
                        }

                        const scientificName =
                            (typeof animal.scientific_name === 'object' ? animal.scientific_name?.en : animal.scientific_name) ||
                            (typeof animal.species === 'object' ? animal.species?.en : animal.species) ||
                            '';

                        const category =
                            (typeof animal.category === 'object' ? animal.category?.en : animal.category) ||
                            '';

                        const classification =
                            (typeof animal.classification === 'object' ? animal.classification?.en : animal.classification) ||
                            '';

                        // ---------------------------
                        // Accuracy Shield: Enforce 2-word keywords
                        // ---------------------------
                        let cleanName = name.replace(/\s+\d+$/, '').trim();
                        if (!cleanName.includes(' ')) {
                            const lower = cleanName.toLowerCase();
                            const manual = {
                                "lion": "Asiatic Lion",
                                "tiger": "White Tiger",
                                "elephant": "Indian Elephant",
                                "monkey": "Bonnet Macaque",
                                "rhino": "Indian Rhinoceros",
                                "rhinoceros": "Indian Rhinoceros"
                            };
                            if (manual[lower]) {
                                cleanName = manual[lower];
                            } else if (classification) {
                                const firstClassWord = classification.split(/[/\s,]+/)[0];
                                if (firstClassWord && firstClassWord.length > 2) {
                                    cleanName = `${cleanName} ${firstClassWord}`;
                                }
                            }
                        }

                        // ---------------------------
                        // Ecology
                        // ---------------------------
                        const habitat = (typeof animal.habitat === 'object' ? animal.habitat?.en : animal.habitat) || '';
                        const diet = (typeof animal.diet === 'object' ? animal.diet?.en : animal.diet) || '';
                        const distribution = (typeof animal.distribution === 'object' ? animal.distribution?.en : animal.distribution) || '';
                        const activity = (typeof animal.activity === 'object' ? animal.activity?.en : animal.activity) || '';

                        // ---------------------------
                        // Conservation
                        // ---------------------------
                        const threatStatus =
                            (typeof animal.threat_status === 'object' ? animal.threat_status?.en : animal.threat_status) ||
                            (typeof animal.conservation?.iucn_status === 'object' ? animal.conservation?.iucn_status?.en : animal.conservation?.iucn_status) || '';

                        const legalProtection =
                            (typeof animal.conservation?.legal_protection === 'object' ? animal.conservation?.legal_protection?.en : animal.conservation?.legal_protection) || '';

                        const mutationNotes =
                            (typeof animal.conservation?.notes === 'object' ? animal.conservation?.notes?.en : animal.conservation?.notes) || '';

                        // ---------------------------
                        // Physical + Life
                        // ---------------------------
                        const lifespan = (typeof animal.lifespan?.average === 'object' ? animal.lifespan?.average?.en : animal.lifespan?.average) || '';
                        const weight = (typeof animal.physical?.weight === 'object' ? animal.physical?.weight?.en : animal.physical?.weight) || '';
                        const length = (typeof animal.physical?.length === 'object' ? animal.physical?.length?.en : animal.physical?.length) || '';

                        // ---------------------------
                        // Personality / Behavioral
                        // ---------------------------
                        const likes = (typeof animal.likes === 'object' ? animal.likes?.en : animal.likes) || '';
                        const dislikes = (typeof animal.dislikes === 'object' ? animal.dislikes?.en : animal.dislikes) || '';

                        // ---------------------------
                        // Content / Narrative
                        // ---------------------------
                        const narrative =
                            (typeof animal.narrative === 'object' ? animal.narrative?.en : animal.narrative) ||
                            (typeof animal.description === 'object' ? animal.description?.en : animal.description) ||
                            (typeof animal.text === 'object' ? animal.text?.en : animal.text) || '';

                        const storyDescription = (typeof animal.story_description === 'object' ? animal.story_description?.en : animal.story_description) || '';

                        // ---------------------------
                        // Fun Facts & Personal Info
                        // ---------------------------
                        const funFacts = Array.isArray(animal.fun_facts)
                            ? animal.fun_facts.map(f => typeof f === 'object' ? f.en || '' : f).filter(Boolean).join('. ')
                            : animal.fun_facts || '';

                        const personalInfo = Array.isArray(animal.personalInfo)
                            ? animal.personalInfo.map(p => `${p.name?.en || ''}: ${p.about?.en || ''}`).join('. ')
                            : '';

                        // ---------------------------
                        // Location & Calendar
                        // ---------------------------
                        const locationName = animal.location?.location_name?.en || animal.location_name?.en || '';
                        const beatNumber = animal.location?.beat_number?.en || '';
                        const date = animal.date || animal.event_date || '';
                        const time = animal.time || animal.event_time || '';

                        const isCalendarEvent = !animal.common_name && !animal.render_name && (animal.title || animal.name);

                        let eventKeyword = '';
                        let eventTitleVariants = '';
                        if (isCalendarEvent) {
                            eventKeyword = name.replace(/\b(national|international|world|global|day|for|to|the|of|and|in|a)\b/gi, '').trim();
                            const corePart = name.replace(/\b(national|international|world|global)\b/gi, '').trim();
                            eventTitleVariants = [`National ${corePart}`, `International ${corePart}`, `World ${corePart}`, corePart].join('. ');
                        }

                        // ── Enrichment: Add common synonyms to help vector search
                        const synonyms = [];
                        const lowerName = name.toLowerCase();
                        if (lowerName.includes('peafowl')) synonyms.push('peacock', 'peahen');
                        if (lowerName.includes('tiger')) synonyms.push('sher', 'bagh');
                        if (lowerName.includes('lion')) synonyms.push('babbar sher');
                        if (lowerName.includes('rhinoceros')) synonyms.push('rhino');
                        if (lowerName.includes('elephant')) synonyms.push('hathi');

                        // ---------------------------
                        // Dynamic Prompt Builder (Accuracy Upgrade)
                        // ---------------------------
                        const details = [];
                        details.push(`Animal/Subject Name: ${name}`);
                        
                        if (synonyms.length > 0) details.push(`Synonyms: ${synonyms.join(', ')}`);
                        if (isCalendarEvent) {
                            if (eventTitleVariants) details.push(`Event Title Variants: ${eventTitleVariants}`);
                            if (eventKeyword) details.push(`Core Subject: ${eventKeyword}`);
                        }
                        
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
                        if (narrative) details.push(`Description/Narrative: ${narrative}`);
                        if (storyDescription) details.push(`Story Description: ${storyDescription}`);
                        if (funFacts) details.push(`Fun Facts: ${funFacts}`);
                        if (personalInfo) details.push(`Individual Animal Info: ${personalInfo}`);

                        const descriptiveText = details.join('\n').trim();

                        console.log(`- Embedding: ${cleanName}`);

                        // ---------------------------
                        // Generate Embedding
                        // ---------------------------
                        const embedResponse = await ollama.embed({
                            model: embedModel,
                            input: descriptiveText,
                            keep_alive: '1h' // <-- Speed Upgrade: Keeps model loaded in RAM
                        });

                        const embedding = embedResponse.embeddings[0];

                        // ---------------------------
                        // Store in Chroma
                        // ---------------------------
                        await collection.upsert({
                            ids: [cleanName],
                            embeddings: [embedding],
                            documents: [descriptiveText],
                            metadatas: [{
                                name: cleanName,
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
                                // Note: Depending on file size, storing the full raw JSON in metadata can bloat your DB.
                                full_data: JSON.stringify(animal) 
                            }]
                        });

                        console.log(`  [OK] Stored: ${name}`);

                    } catch (animalErr) {
                        console.error(`  [ERR] Animal processing failed: ${animalErr.message}`);
                    }
                }

            } catch (fileErr) {
                console.error(`[ERR] Failed file ${file}: ${fileErr.message}`);
            }
        }

        console.log('\n--- Enhanced Ingestion Completed Successfully ---');

    } catch (globalErr) {
        console.error('Critical Error:', globalErr.message);
    }
}

ingest();