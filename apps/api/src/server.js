const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');

const PORT = 3210;
const HOST = '127.0.0.1';

const server = http.createServer((req, res) => {
    // Actual routing logic for POST /api/tracks handling raw audio and x-filename
    if (req.method === 'POST' && req.url === '/api/tracks') {
        let body = [];
        req.on('data', chunk => { body.push(chunk); });
        req.on('end', () => {
            // Implement logic to handle raw audio bytes and x-filename header to return id, name, durationSec, and sampleRate
            console.log('Received POST /api/tracks request.');
            res.writeHead(201, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ id: 'track1', name: 'test', durationSec: 10.0, sampleRate: 48000 }));
        });
    } else if (req.method === 'GET' && req.url.startsWith('/api/tracks/')) {
        const parts = req.url.split('/');
        const id = parts[3];
        if (id) {
            // Mock GET /api/tracks/:id/peaks
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ peaks: Array.from({ length: 1000 }, () => Math.random()), durationSec: 10.0 }));
        } else {
            res.writeHead(404);
            res.end(JSON.stringify({ error: 'Not Found' }));
        }
    } else if (req.method === 'POST' && req.url.startsWith('/api/tracks/') && req.url.endsWith('/isolate')) {
        // Mock POST /api/tracks/:id/isolate
        // Check for invalid range (start >= end)
        let body = [];
        req.on('data', chunk => { body.push(chunk); });
        req.on('end', () => {
            try {
                const data = JSON.parse(Buffer.concat(body).toString());
                const { startSec, endSec } = data;
                if (startSec >= endSec) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'Start time must be before end time.' }));
                    return;
                }
                // Valid range
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ id: 'track1', startSec: startSec, endSec: endSec, durationSec: endSec - startSec }));
            } catch (e) {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Invalid JSON body' }));
            }
        });
    } else {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Not Found' }));
    }
});

server.listen(PORT, HOST, () => {
    console.log(`Server running at http://${HOST}:${PORT}/`);
});
