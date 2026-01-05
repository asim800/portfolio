/**
 * Monte Carlo Simulation API Client
 *
 * Handles all API calls to the FastAPI backend at /api/mc/*
 */

class MCApiClient {
  constructor(baseUrl = '') {
    this.baseUrl = baseUrl;
  }

  /**
   * Make a fetch request with error handling
   */
  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const defaultOptions = {
      headers: {
        'Content-Type': 'application/json',
      },
    };

    try {
      const response = await fetch(url, { ...defaultOptions, ...options });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || data.message || `HTTP ${response.status}`);
      }

      return data;
    } catch (error) {
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        throw new Error('Cannot connect to API server');
      }
      throw error;
    }
  }

  // ===========================================================================
  // Health & Status
  // ===========================================================================

  /**
   * Check API health status
   * GET /health
   */
  async healthCheck() {
    return this.request('/health');
  }

  // ===========================================================================
  // Simulation
  // ===========================================================================

  /**
   * Run Monte Carlo simulation
   * POST /api/mc/simulate
   */
  async runSimulation(config) {
    return this.request('/api/mc/simulate', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  // ===========================================================================
  // Parameter Sweeps
  // ===========================================================================

  /**
   * Run single parameter sweep
   * POST /api/mc/sweep
   * @param {object} baseConfig - Base simulation configuration
   * @param {string} paramName - Parameter to sweep
   * @param {number} start - Start value
   * @param {number} end - End value
   * @param {number} step - Step size
   * @param {boolean} skipBootstrap - Skip bootstrap sampling
   */
  async runSweep(baseConfig, paramName, start, end, step, skipBootstrap = true) {
    return this.request('/api/mc/sweep', {
      method: 'POST',
      body: JSON.stringify({
        base_config: baseConfig,
        param_name: paramName,
        param_range: { start, end, step },
        skip_bootstrap: skipBootstrap,
      }),
    });
  }

  /**
   * Run 2D grid parameter sweep
   * POST /api/mc/grid-sweep
   * @param {object} baseConfig - Base simulation configuration
   * @param {string} param1Name - First parameter to sweep
   * @param {object} param1Range - {start, end, step} for first param
   * @param {string} param2Name - Second parameter to sweep
   * @param {object} param2Range - {start, end, step} for second param
   * @param {boolean} skipBootstrap - Skip bootstrap sampling
   */
  async runGridSweep(baseConfig, param1Name, param1Range, param2Name, param2Range, skipBootstrap = true) {
    return this.request('/api/mc/grid-sweep', {
      method: 'POST',
      body: JSON.stringify({
        base_config: baseConfig,
        param1_name: param1Name,
        param1_range: param1Range,
        param2_name: param2Name,
        param2_range: param2Range,
        skip_bootstrap: skipBootstrap,
      }),
    });
  }

  // ===========================================================================
  // Configuration
  // ===========================================================================

  /**
   * Get list of sweepable parameters
   * GET /api/mc/config/sweep-params
   */
  async getSweepParams() {
    return this.request('/api/mc/config/sweep-params');
  }

  /**
   * Validate configuration before running
   * POST /api/mc/config/validate
   */
  async validateConfig(config, checkDataAvailability = true) {
    return this.request('/api/mc/config/validate', {
      method: 'POST',
      body: JSON.stringify({
        config: config,
        check_data_availability: checkDataAvailability,
      }),
    });
  }

  /**
   * Get JSON schema for configuration
   * GET /api/mc/config/schema
   */
  async getConfigSchema() {
    return this.request('/api/mc/config/schema');
  }

  // ===========================================================================
  // Jobs Management
  // ===========================================================================

  /**
   * Get status of a specific job
   * GET /api/mc/jobs/{jobId}
   */
  async getJobStatus(jobId) {
    return this.request(`/api/mc/jobs/${jobId}`);
  }

  /**
   * Cancel a running job
   * POST /api/mc/jobs/{jobId}/cancel
   */
  async cancelJob(jobId) {
    return this.request(`/api/mc/jobs/${jobId}/cancel`, {
      method: 'POST',
    });
  }

  /**
   * List all jobs, optionally filtered by status
   * GET /api/mc/jobs?status=
   */
  async listJobs(status = null) {
    const query = status ? `?status=${status}` : '';
    return this.request(`/api/mc/jobs${query}`);
  }

  // ===========================================================================
  // Helpers
  // ===========================================================================

  /**
   * Poll a job until it completes
   * @param {string} jobId - Job ID to poll
   * @param {function} onProgress - Callback for progress updates (0-1)
   * @param {number} interval - Polling interval in ms
   * @returns {Promise} - Resolves with final job status
   */
  async pollJob(jobId, onProgress = null, interval = 1000) {
    return new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const status = await this.getJobStatus(jobId);

          if (onProgress && status.progress !== undefined) {
            onProgress(status.progress);
          }

          if (status.status === 'completed') {
            resolve(status);
          } else if (status.status === 'failed') {
            reject(new Error(status.error || 'Job failed'));
          } else if (status.status === 'cancelled') {
            reject(new Error('Job was cancelled'));
          } else {
            // Still running, poll again
            setTimeout(poll, interval);
          }
        } catch (error) {
          reject(error);
        }
      };

      poll();
    });
  }
}

// Create global instance
const api = new MCApiClient();
