const fs = require('fs');
const path = require('path');
const { ChromaClient } = require('chromadb');
const { OllamaEmbeddingFunction } = require('@chroma-core/ollama');
const { Ollama } = require('ollama');

/**
 * Shera AI - Enhanced Chroma Ingestion Script
 * Hybrid-ready semantic ingestion for:
 * - Species
 * - Conservation
 * - Habitat
 * - Diet
 * - Storytelling
 * - Individual zoo animals
 * - Location-based retrieval
 */

async function ingest() {
    const chroma = new ChromaClient({
        path: "http://localhost:8001"
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
            // Skip map/geometry files that pollute search results with thousands of coordinates
            if (file.includes('geojson.json') || file.includes('floorplan.json')) {
                console.log(`Skipping map data file: ${file}`);
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
                            animal.common_name?.en ||
                            animal.render_name?.en ||
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
                        const habitat =
                            (typeof animal.habitat === 'object' ? animal.habitat?.en : animal.habitat) ||
                            '';

                        const diet =
                            (typeof animal.diet === 'object' ? animal.diet?.en : animal.diet) ||
                            '';

                        const distribution =
                            (typeof animal.distribution === 'object' ? animal.distribution?.en : animal.distribution) ||
                            '';

                        const activity =
                            (typeof animal.activity === 'object' ? animal.activity?.en : animal.activity) ||
                            '';

                        // ---------------------------
                        // Conservation
                        // ---------------------------
                        const threatStatus =
                            (typeof animal.threat_status === 'object' ? animal.threat_status?.en : animal.threat_status) ||
                            (typeof animal.conservation?.iucn_status === 'object' ? animal.conservation?.iucn_status?.en : animal.conservation?.iucn_status) ||
                            '';

                        const legalProtection =
                            (typeof animal.conservation?.legal_protection === 'object' ? animal.conservation?.legal_protection?.en : animal.conservation?.legal_protection) ||
                            '';

                        const mutationNotes =
                            (typeof animal.conservation?.notes === 'object' ? animal.conservation?.notes?.en : animal.conservation?.notes) ||
                            '';

                        // ---------------------------
                        // Physical + Life
                        // ---------------------------
                        const lifespan =
                            (typeof animal.lifespan?.average === 'object' ? animal.lifespan?.average?.en : animal.lifespan?.average) ||
                            '';

                        const weight =
                            (typeof animal.physical?.weight === 'object' ? animal.physical?.weight?.en : animal.physical?.weight) ||
                            '';

                        const length =
                            (typeof animal.physical?.length === 'object' ? animal.physical?.length?.en : animal.physical?.length) ||
                            '';

                        // ---------------------------
                        // Personality / Behavioral
                        // ---------------------------
                        const likes =
                            (typeof animal.likes === 'object' ? animal.likes?.en : animal.likes) ||
                            '';

                        const dislikes =
                            (typeof animal.dislikes === 'object' ? animal.dislikes?.en : animal.dislikes) ||
                            '';

                        // ---------------------------
                        // Content / Narrative
                        // ---------------------------
                        const narrative =
                            (typeof animal.narrative === 'object' ? animal.narrative?.en : animal.narrative) ||
                            (typeof animal.description === 'object' ? animal.description?.en : animal.description) ||
                            (typeof animal.text === 'object' ? animal.text?.en : animal.text) ||
                            '';

                        const storyDescription =
                            (typeof animal.story_description === 'object' ? animal.story_description?.en : animal.story_description) ||
                            '';

                        // ---------------------------
                        // Fun Facts
                        // ---------------------------
                        const funFacts = Array.isArray(animal.fun_facts)
                            ? animal.fun_facts
                                .map(f =>
                                    typeof f === 'object'
                                        ? f.en || ''
                                        : f
                                )
                                .filter(Boolean)
                                .join('. ')
                            : animal.fun_facts || '';

                        // ---------------------------
                        // Location
                        // ---------------------------
                        const locationName =
                            animal.location?.location_name?.en ||
                            animal.location_name?.en ||
                            '';

                        const beatNumber =
                            animal.location?.beat_number?.en ||
                            '';

                        // ---------------------------
                        // Dates & Times (Calendar/Events)
                        // ---------------------------
                        const date = animal.date || animal.event_date || '';
                        const time = animal.time || animal.event_time || '';

                        // ---------------------------
                        // Individual Animals
                        // ---------------------------
                        const personalInfo = Array.isArray(animal.personalInfo)
                            ? animal.personalInfo
                                .map(p =>
                                    `${p.name?.en || ''}: ${p.about?.en || ''}`
                                )
                                .join('. ')
                            : '';

                        // ---------------------------
                        // Build Rich Embedding Document
                        // ---------------------------
                        const isCalendarEvent = !animal.common_name && !animal.render_name && (animal.title || animal.name);

                        // Extract core keyword from event title (e.g. "International Sloth Day" -> "Sloth")
                        let eventKeyword = '';
                        let eventTitleVariants = '';
                        if (isCalendarEvent) {
                            const rawTitle = name; // already extracted above
                            // Strip common prefix/suffix words to get the subject
                            eventKeyword = rawTitle
                                .replace(/\b(national|international|world|global|day|for|to|the|of|and|in|a)\b/gi, '')
                                .trim();

                            // Generate name variants so "national X day" matches "international X day" etc.
                            const corePart = rawTitle.replace(/\b(national|international|world|global)\b/gi, '').trim();
                            eventTitleVariants = [
                                `National ${corePart}`,
                                `International ${corePart}`,
                                `World ${corePart}`,
                                corePart
                            ].join('. ');
                        }
                        const descriptiveText = `
                                Animal Name: ${name}
                                ${isCalendarEvent ? `Event Title Variants: ${eventTitleVariants}` : ''}
                                ${isCalendarEvent ? `Core Subject: ${eventKeyword}` : ''}
                                Scientific Name: ${scientificName}
                                Category: ${category}
                                Classification: ${classification}
                                Habitat: ${habitat}
                                Distribution: ${distribution}
                                Diet: ${diet}
                                Activity Pattern: ${activity}
                                Date/Time: ${date} ${time}
                                Lifespan: ${lifespan}
                                Weight: ${weight}
                                Length: ${length}
                                Threat Status: ${threatStatus}
                                Legal Protection: ${legalProtection}
                                Genetic Traits: ${mutationNotes}
                                Likes: ${likes}
                                Dislikes: ${dislikes}
                                Location: ${locationName}
                                Beat Number: ${beatNumber}
                                Description/Narrative: ${narrative}
                                Story Description: ${storyDescription}
                                Fun Facts: ${funFacts}
                                Individual Animal Info: ${personalInfo}
                                `.trim();

                        console.log(`- Embedding: ${cleanName}`);

                        // ---------------------------
                        // Generate Embedding
                        // ---------------------------
                        const embedResponse = await ollama.embed({
                            model: embedModel,
                            input: descriptiveText
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
                                file_source: file,
                                full_data: JSON.stringify(animal)
                            }]
                        });

                        console.log(`  [OK] Stored: ${name}`);

                    } catch (animalErr) {
                        console.error(
                            `  [ERR] Animal processing failed: ${animalErr.message}`
                        );
                    }
                }

            } catch (fileErr) {
                console.error(
                    `[ERR] Failed file ${file}: ${fileErr.message}`
                );
            }
        }

        console.log('\n--- Enhanced Ingestion Completed Successfully ---');

    } catch (globalErr) {
        console.error('Critical Error:', globalErr.message);
    }
}

ingest();