/**
 * Monte Carlo Simulation UI Server
 *
 * Express server that:
 * 1. Serves static files from public/
 * 2. Proxies /api/* and /health requests to FastAPI backend at localhost:8001
 */

const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const API_URL = process.env.API_URL || 'http://localhost:8001';

// Proxy configuration for FastAPI backend
const apiProxy = createProxyMiddleware({
  target: API_URL,
  changeOrigin: true,
  logLevel: 'warn',
  onError: (err, req, res) => {
    console.error('Proxy error:', err.message);
    res.status(502).json({
      error: 'Backend API unavailable',
      message: `Could not connect to ${API_URL}`,
      details: err.message
    });
  }
});

// Proxy API routes to FastAPI
app.use('/api', apiProxy);
app.use('/health', apiProxy);

// Serve static files from public directory
app.use(express.static(path.join(__dirname, 'public')));

// Fallback to index.html for SPA-style routing
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Start server
app.listen(PORT, () => {
  console.log('='.repeat(60));
  console.log('Monte Carlo Simulation UI');
  console.log('='.repeat(60));
  console.log(`Frontend: http://localhost:${PORT}`);
  console.log(`Backend API: ${API_URL}`);
  console.log('='.repeat(60));
});
