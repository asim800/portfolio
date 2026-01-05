/**
 * Monte Carlo Simulation UI Application Logic
 */

// State
let currentJobId = null;
let accChart = null;
let decChart = null;
let sweepChart = null;
let sweepAbortController = null;

// Default sweep ranges for parameters
const SWEEP_DEFAULTS = {
  annual_withdrawal_amount: { start: 20000, end: 80000, step: 10000 },
  initial_portfolio_value: { start: 500000, end: 2000000, step: 250000 },
  simulation_horizon_years: { start: 10, end: 40, step: 5 },
  inflation_rate: { start: 0.01, end: 0.06, step: 0.01 },
  contribution_amount: { start: 0, end: 2000, step: 500 },
};

// ===========================================================================
// Initialization
// ===========================================================================

document.addEventListener('DOMContentLoaded', () => {
  // Set default retirement date (10 years from now)
  const retirementDate = new Date();
  retirementDate.setFullYear(retirementDate.getFullYear() + 10);
  document.getElementById('retirement_date').value = formatDate(retirementDate);

  // Set default dates
  document.getElementById('start_date').value = '2005-01-01';
  document.getElementById('end_date').value = '2025-01-01';

  // Bind form submit
  document.getElementById('simulation-form').addEventListener('submit', handleSubmit);

  // Update weight sum on input
  document.getElementById('tickers-container').addEventListener('input', updateWeightSum);

  // Check API health
  checkApiHealth();
  setInterval(checkApiHealth, 30000); // Check every 30 seconds

  // Load jobs
  refreshJobs();
});

// ===========================================================================
// API Health
// ===========================================================================

async function checkApiHealth() {
  const statusDot = document.getElementById('api-status');
  const statusText = document.getElementById('api-status-text');

  try {
    const result = await api.healthCheck();
    statusDot.className = 'status-dot online';
    statusText.textContent = `API: ${result.status}`;
  } catch (error) {
    statusDot.className = 'status-dot offline';
    statusText.textContent = 'API: Offline';
  }
}

// ===========================================================================
// Form Handling
// ===========================================================================

function formatDate(date) {
  return date.toISOString().split('T')[0];
}

function getFormConfig() {
  const form = document.getElementById('simulation-form');

  // Collect tickers
  const tickerRows = document.querySelectorAll('.ticker-row');
  const tickers = [];
  tickerRows.forEach(row => {
    const symbol = row.querySelector('.ticker-symbol').value.trim().toUpperCase();
    const weight = parseFloat(row.querySelector('.ticker-weight').value) || 0;
    if (symbol) {
      tickers.push({ symbol, weight });
    }
  });

  return {
    initial_portfolio_value: parseFloat(form.initial_portfolio_value.value),
    retirement_date: form.retirement_date.value,
    simulation_horizon_years: parseInt(form.simulation_horizon_years.value),
    tickers: tickers,
    start_date: form.start_date.value,
    end_date: form.end_date.value,
    num_simulations: parseInt(form.num_simulations.value),
    simulation_frequency: form.simulation_frequency.value,
    sampling_method: form.sampling_method.value,
    seed: parseInt(form.seed.value),
    contribution_amount: parseFloat(form.contribution_amount.value),
    contribution_frequency: form.contribution_frequency.value,
    employer_match_rate: parseFloat(form.employer_match_rate.value) / 100,
    withdrawal_strategy: form.withdrawal_strategy.value,
    annual_withdrawal_amount: parseFloat(form.annual_withdrawal_amount.value),
    withdrawal_frequency: form.withdrawal_frequency.value,
    inflation_rate: parseFloat(form.inflation_rate.value) / 100,
  };
}

function updateWeightSum() {
  const weights = document.querySelectorAll('.ticker-weight');
  let sum = 0;
  weights.forEach(w => {
    sum += parseFloat(w.value) || 0;
  });

  const sumDisplay = document.getElementById('weight-sum');
  const percentage = (sum * 100).toFixed(0);
  sumDisplay.textContent = `Total: ${percentage}%`;
  sumDisplay.className = Math.abs(sum - 1) < 0.01 ? 'weight-info valid' : 'weight-info invalid';
}

function addTicker() {
  const container = document.getElementById('tickers-container');
  const row = document.createElement('div');
  row.className = 'ticker-row';
  row.innerHTML = `
    <input type="text" name="ticker_symbol" placeholder="Symbol" class="ticker-symbol">
    <input type="number" name="ticker_weight" value="0" min="0" max="1" step="0.05" class="ticker-weight">
    <button type="button" class="btn-remove-ticker" onclick="removeTicker(this)">-</button>
  `;
  container.appendChild(row);
}

function removeTicker(button) {
  const rows = document.querySelectorAll('.ticker-row');
  if (rows.length > 1) {
    button.parentElement.remove();
    updateWeightSum();
  }
}

// ===========================================================================
// Validation
// ===========================================================================

async function validateConfig() {
  const messagesDiv = document.getElementById('validation-messages');
  messagesDiv.innerHTML = '';
  messagesDiv.className = 'messages';

  try {
    showLoading('Validating configuration...');
    const config = getFormConfig();
    const result = await api.validateConfig(config);
    hideLoading();

    messagesDiv.classList.remove('hidden');

    if (result.valid) {
      messagesDiv.innerHTML = '<div class="message success">Configuration is valid!</div>';
    }

    if (result.errors && result.errors.length > 0) {
      result.errors.forEach(err => {
        messagesDiv.innerHTML += `<div class="message error">${err}</div>`;
      });
    }

    if (result.warnings && result.warnings.length > 0) {
      result.warnings.forEach(warn => {
        messagesDiv.innerHTML += `<div class="message warning">${warn}</div>`;
      });
    }

    if (result.data_availability) {
      const da = result.data_availability;
      if (da.bootstrap_available) {
        messagesDiv.innerHTML += `<div class="message info">Bootstrap data: ${da.periods} periods available</div>`;
      } else {
        messagesDiv.innerHTML += `<div class="message warning">Bootstrap data not available: ${da.error || 'Unknown error'}</div>`;
      }
    }
  } catch (error) {
    hideLoading();
    messagesDiv.classList.remove('hidden');
    messagesDiv.innerHTML = `<div class="message error">Validation failed: ${error.message}</div>`;
  }
}

// ===========================================================================
// Simulation
// ===========================================================================

async function handleSubmit(event) {
  event.preventDefault();

  const config = getFormConfig();
  const messagesDiv = document.getElementById('validation-messages');
  messagesDiv.classList.add('hidden');

  try {
    showLoading('Running simulation...');
    document.getElementById('run-btn').disabled = true;

    const result = await api.runSimulation(config);

    hideLoading();
    document.getElementById('run-btn').disabled = false;

    if (result.success) {
      displayResults(result);
    } else {
      showError(result.error || 'Simulation failed');
    }
  } catch (error) {
    hideLoading();
    document.getElementById('run-btn').disabled = false;
    showError(error.message);
  }

  // Refresh jobs list
  refreshJobs();
}

// ===========================================================================
// Results Display
// ===========================================================================

function displayResults(result) {
  const panel = document.getElementById('results-panel');
  panel.classList.remove('hidden');

  // Display summary table
  displaySummaryTable(result.summary_table);

  // Display charts
  if (result.accumulation && result.accumulation.parametric) {
    renderFanChart('acc-chart', result.accumulation.parametric.data, 'Accumulation');
  }
  if (result.decumulation && result.decumulation.parametric) {
    renderFanChart('dec-chart', result.decumulation.parametric.data, 'Decumulation');
  }

  // Display metadata
  displayMetadata(result.metadata);

  // Scroll to results
  panel.scrollIntoView({ behavior: 'smooth' });
}

function displaySummaryTable(table) {
  if (!table) return;

  const headerRow = document.getElementById('summary-header');
  const tbody = document.getElementById('summary-body');

  // Clear existing
  headerRow.innerHTML = '';
  tbody.innerHTML = '';

  // Build header
  table.columns.forEach(col => {
    const th = document.createElement('th');
    th.textContent = table.column_labels[col] || col;
    headerRow.appendChild(th);
  });

  // Build rows with styling
  if (table.styled_rows) {
    table.styled_rows.forEach(row => {
      const tr = document.createElement('tr');
      if (row.row_class) tr.className = row.row_class;

      table.columns.forEach(col => {
        const td = document.createElement('td');
        const cell = row.cells[col];

        if (cell) {
          td.textContent = cell.display;
          if (cell.color) td.style.backgroundColor = cell.color;
          if (cell.css_class) td.className = cell.css_class;
          if (cell.status) td.classList.add(`status-${cell.status}`);
        }

        tr.appendChild(td);
      });

      tbody.appendChild(tr);
    });
  } else {
    // Fallback to raw rows
    table.rows.forEach(row => {
      const tr = document.createElement('tr');
      table.columns.forEach(col => {
        const td = document.createElement('td');
        const value = row[col];
        td.textContent = formatValue(col, value);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }
}

function formatValue(key, value) {
  if (typeof value === 'number') {
    if (key.includes('rate') || key.includes('percentage')) {
      return (value * 100).toFixed(1) + '%';
    }
    if (value > 1000) {
      return '$' + value.toLocaleString(undefined, { maximumFractionDigits: 0 });
    }
    return value.toFixed(2);
  }
  return value;
}

function displayMetadata(metadata) {
  const container = document.getElementById('metadata');
  container.innerHTML = `
    <div class="meta-item"><strong>Simulations:</strong> ${metadata.num_simulations}</div>
    <div class="meta-item"><strong>Accumulation:</strong> ${metadata.accumulation_years} years (${metadata.accumulation_periods} periods)</div>
    <div class="meta-item"><strong>Decumulation:</strong> ${metadata.decumulation_years} years (${metadata.decumulation_periods} periods)</div>
    <div class="meta-item"><strong>Frequency:</strong> ${metadata.periods_per_year} periods/year</div>
    <div class="meta-item"><strong>Tickers:</strong> ${metadata.tickers.join(', ')}</div>
    <div class="meta-item"><strong>Methods:</strong> ${metadata.sampling_methods_used.join(', ')}</div>
    <div class="meta-item"><strong>Execution Time:</strong> ${metadata.execution_time_ms}ms</div>
  `;
}

// ===========================================================================
// Jobs Management
// ===========================================================================

async function refreshJobs() {
  try {
    const result = await api.listJobs();
    const tbody = document.getElementById('jobs-body');

    if (result.jobs.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="empty">No jobs yet</td></tr>';
      return;
    }

    tbody.innerHTML = '';
    result.jobs.forEach(job => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="job-id">${job.job_id.substring(0, 8)}...</td>
        <td class="status-${job.status}">${job.status}</td>
        <td>${job.progress !== undefined ? (job.progress * 100).toFixed(0) + '%' : '-'}</td>
        <td>${new Date(job.created_at).toLocaleTimeString()}</td>
        <td>
          ${job.status === 'running' || job.status === 'pending' ?
            `<button onclick="cancelJobById('${job.job_id}')" class="btn-small">Cancel</button>` : ''}
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (error) {
    console.error('Failed to load jobs:', error);
  }
}

async function cancelJobById(jobId) {
  try {
    await api.cancelJob(jobId);
    refreshJobs();
  } catch (error) {
    showError(`Failed to cancel job: ${error.message}`);
  }
}

function cancelJob() {
  if (currentJobId) {
    cancelJobById(currentJobId);
    currentJobId = null;
  }
}

// ===========================================================================
// UI Helpers
// ===========================================================================

function showLoading(message) {
  document.getElementById('loading-message').textContent = message;
  document.getElementById('loading-overlay').classList.remove('hidden');
}

function hideLoading() {
  document.getElementById('loading-overlay').classList.add('hidden');
}

function showError(message) {
  const messagesDiv = document.getElementById('validation-messages');
  messagesDiv.classList.remove('hidden');
  messagesDiv.innerHTML = `<div class="message error">${message}</div>`;
}

// ===========================================================================
// Parameter Sweep
// ===========================================================================

function toggleSweep() {
  const enabled = document.getElementById('enable-sweep').checked;
  const controls = document.getElementById('sweep-controls');
  const runBtn = document.getElementById('run-btn');
  const sweepBtn = document.getElementById('sweep-btn');

  if (enabled) {
    controls.classList.remove('hidden');
    runBtn.style.display = 'none';
    sweepBtn.style.display = 'inline-block';
  } else {
    controls.classList.add('hidden');
    runBtn.style.display = 'inline-block';
    sweepBtn.style.display = 'none';
  }
}

function updateSweepDefaults(paramNum) {
  const paramSelect = document.getElementById(`sweep-param${paramNum}`);
  const startInput = document.getElementById(`sweep-param${paramNum}-start`);
  const endInput = document.getElementById(`sweep-param${paramNum}-end`);
  const stepInput = document.getElementById(`sweep-param${paramNum}-step`);

  const paramName = paramSelect.value;
  if (paramName && SWEEP_DEFAULTS[paramName]) {
    const defaults = SWEEP_DEFAULTS[paramName];
    startInput.value = defaults.start;
    endInput.value = defaults.end;
    stepInput.value = defaults.step;
  } else {
    // Clear values if None selected
    startInput.value = '';
    endInput.value = '';
    stepInput.value = '';
  }
}

async function runSweep() {
  const config = getFormConfig();
  const messagesDiv = document.getElementById('validation-messages');
  messagesDiv.classList.add('hidden');

  const param1Name = document.getElementById('sweep-param1').value;
  const param2Name = document.getElementById('sweep-param2').value;
  const skipBootstrap = document.getElementById('skip-bootstrap-sweep').checked;

  if (!param1Name) {
    showError('Please select Parameter 1 for sweep');
    return;
  }

  const param1Start = parseFloat(document.getElementById('sweep-param1-start').value);
  const param1End = parseFloat(document.getElementById('sweep-param1-end').value);
  const param1Step = parseFloat(document.getElementById('sweep-param1-step').value);

  // Auto-detect 2D sweep if Parameter 2 is selected
  const is2D = !!param2Name;

  // Show stop button, hide sweep button
  document.getElementById('sweep-btn').style.display = 'none';
  document.getElementById('stop-sweep-btn').style.display = 'inline-block';

  try {
    if (is2D) {
      const param2Start = parseFloat(document.getElementById('sweep-param2-start').value);
      const param2End = parseFloat(document.getElementById('sweep-param2-end').value);
      const param2Step = parseFloat(document.getElementById('sweep-param2-step').value);

      showLoading('Running 2D grid sweep...');
      const result = await api.runGridSweep(
        config,
        param1Name,
        { start: param1Start, end: param1End, step: param1Step },
        param2Name,
        { start: param2Start, end: param2End, step: param2Step },
        skipBootstrap
      );
      hideLoading();

      if (result.success) {
        displayGridSweepResults(result);
      } else {
        showError(result.error || 'Grid sweep failed');
      }
    } else {
      showLoading('Running parameter sweep...');
      const result = await api.runSweep(
        config,
        param1Name,
        param1Start,
        param1End,
        param1Step,
        skipBootstrap
      );
      hideLoading();

      if (result.success) {
        displaySweepResults(result);
      } else {
        showError(result.error || 'Sweep failed');
      }
    }
  } catch (error) {
    hideLoading();
    if (error.name !== 'AbortError') {
      showError(error.message);
    }
  }

  // Hide stop button, show sweep button
  document.getElementById('stop-sweep-btn').style.display = 'none';
  document.getElementById('sweep-btn').style.display = 'inline-block';

  refreshJobs();
}

function stopSweep() {
  // Hide loading overlay and show stopped message
  hideLoading();
  showError('Sweep stopped by user');

  // Reset buttons
  document.getElementById('stop-sweep-btn').style.display = 'none';
  document.getElementById('sweep-btn').style.display = 'inline-block';
}

// ===========================================================================
// Sweep Results Display
// ===========================================================================

function displaySweepResults(result) {
  const panel = document.getElementById('sweep-results-panel');
  const results1D = document.getElementById('sweep-1d-results');
  const results2D = document.getElementById('sweep-2d-results');

  panel.classList.remove('hidden');
  results1D.classList.remove('hidden');
  results2D.classList.add('hidden');

  // Render sweep chart
  renderSweepChart(result);

  // Render sweep table
  renderSweepTable(result);

  // Display metadata
  displaySweepMetadata(result);

  // Scroll to results
  panel.scrollIntoView({ behavior: 'smooth' });
}

function renderSweepChart(result) {
  const canvas = document.getElementById('sweep-chart');
  const ctx = canvas.getContext('2d');

  // Destroy existing chart
  if (sweepChart) {
    sweepChart.destroy();
  }

  const labels = result.parametric_results.map(r => r.param_formatted);
  const successRates = result.parametric_results.map(r => r.success_rate * 100);

  sweepChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Success Rate (%)',
        data: successRates,
        borderColor: '#1976d2',
        backgroundColor: 'rgba(25, 118, 210, 0.1)',
        fill: true,
        tension: 0.2,
        pointRadius: 6,
        pointBackgroundColor: successRates.map(r =>
          r >= 80 ? '#4CAF50' : r >= 60 ? '#FFC107' : '#F44336'
        ),
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true },
        title: {
          display: true,
          text: `Success Rate by ${result.param_name}`,
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          title: { display: true, text: 'Success Rate (%)' }
        },
        x: {
          title: { display: true, text: result.param_description || result.param_name }
        }
      }
    }
  });
}

function renderSweepTable(result) {
  const thead = document.getElementById('sweep-table-header');
  const tbody = document.getElementById('sweep-table-body');

  thead.innerHTML = `
    <tr>
      <th>${result.param_description || result.param_name}</th>
      <th>Success Rate</th>
      <th>Acc Median</th>
      <th>Dec Median</th>
    </tr>
  `;

  tbody.innerHTML = '';
  result.parametric_results.forEach(r => {
    const tr = document.createElement('tr');
    const successClass = r.success_rate >= 0.8 ? 'success-high' :
                         r.success_rate >= 0.6 ? 'success-medium' : 'success-low';

    tr.innerHTML = `
      <td>${r.param_formatted}</td>
      <td class="${successClass}">${(r.success_rate * 100).toFixed(1)}%</td>
      <td>$${r.acc_p50.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
      <td>$${r.dec_p50.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
    `;
    tbody.appendChild(tr);
  });
}

function displayGridSweepResults(result) {
  const panel = document.getElementById('sweep-results-panel');
  const results1D = document.getElementById('sweep-1d-results');
  const results2D = document.getElementById('sweep-2d-results');

  panel.classList.remove('hidden');
  results1D.classList.add('hidden');
  results2D.classList.remove('hidden');

  // Render heatmap table
  renderHeatmapTable(result);

  // Render grid table
  renderGridTable(result);

  // Display metadata
  displaySweepMetadata(result);

  // Scroll to results
  panel.scrollIntoView({ behavior: 'smooth' });
}

function renderHeatmapTable(result) {
  const container = document.getElementById('heatmap-container');

  // Format param names for display (snake_case -> Title Case)
  const param1Display = formatParamName(result.param1_name);
  const param2Display = formatParamName(result.param2_name);

  let html = `<div class="heatmap-title">Success Rate (Parametric)</div>`;
  html += `<table class="heatmap-table cli-style">`;

  // Header row with param2 labels
  html += `<thead><tr>`;
  html += `<th class="corner-cell"></th>`;
  result.param2_labels.forEach(label => {
    html += `<th>${label}</th>`;
  });
  html += `</tr></thead>`;

  // Data rows
  html += `<tbody>`;
  result.param1_labels.forEach((label, i) => {
    html += `<tr>`;
    html += `<td class="row-header">${label}</td>`;
    result.parametric_success_matrix[i].forEach(successRate => {
      const percentage = (successRate * 100).toFixed(0);
      const bgColor = getSuccessColor(successRate);
      const textColor = successRate >= 0.5 ? 'white' : '#333';
      html += `<td style="background-color: ${bgColor}; color: ${textColor}; font-weight: 600;">${percentage}%</td>`;
    });
    html += `</tr>`;
  });
  html += `</tbody></table>`;

  // Add axis labels
  html += `<div class="axis-labels">`;
  html += `<div class="y-axis-label">${param1Display}</div>`;
  html += `<div class="x-axis-label">${param2Display}</div>`;
  html += `</div>`;

  container.innerHTML = html;
}

function formatParamName(paramName) {
  // Convert snake_case to Title Case
  return paramName
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function getSuccessColor(rate) {
  // Gradient from red (0%) to yellow (50%) to green (100%)
  if (rate >= 0.8) return '#4CAF50';  // Green
  if (rate >= 0.6) return '#8BC34A';  // Light green
  if (rate >= 0.4) return '#FFC107';  // Yellow
  if (rate >= 0.2) return '#FF9800';  // Orange
  return '#F44336';  // Red
}

function renderGridTable(result) {
  const thead = document.getElementById('grid-table-header');
  const tbody = document.getElementById('grid-table-body');

  const param1Display = formatParamName(result.param1_name);
  const param2Display = formatParamName(result.param2_name);

  thead.innerHTML = `
    <tr>
      <th>${param1Display}</th>
      <th>${param2Display}</th>
      <th>Success Rate</th>
      <th>Acc Median</th>
      <th>Dec Median</th>
    </tr>
  `;

  tbody.innerHTML = '';
  result.results.forEach(r => {
    const tr = document.createElement('tr');
    const successClass = r.success_rate >= 0.8 ? 'success-high' :
                         r.success_rate >= 0.6 ? 'success-medium' : 'success-low';

    tr.innerHTML = `
      <td>${r.param1_formatted}</td>
      <td>${r.param2_formatted}</td>
      <td class="${successClass}">${(r.success_rate * 100).toFixed(1)}%</td>
      <td>$${r.acc_median.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
      <td>$${r.dec_median.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
    `;
    tbody.appendChild(tr);
  });
}

function displaySweepMetadata(result) {
  const container = document.getElementById('sweep-metadata');

  if (result.param2_name) {
    // 2D sweep
    const param1Display = formatParamName(result.param1_name);
    const param2Display = formatParamName(result.param2_name);
    container.innerHTML = `
      <div class="meta-item"><strong>Parameter 1 (Rows):</strong> ${param1Display}</div>
      <div class="meta-item"><strong>Parameter 2 (Columns):</strong> ${param2Display}</div>
      <div class="meta-item"><strong>Grid Size:</strong> ${result.param1_values.length} x ${result.param2_values.length}</div>
      <div class="meta-item"><strong>Total Cells:</strong> ${result.results.length}</div>
      <div class="meta-item"><strong>Execution Time:</strong> ${(result.execution_time_ms / 1000).toFixed(1)}s</div>
    `;
  } else {
    // 1D sweep
    container.innerHTML = `
      <div class="meta-item"><strong>Parameter:</strong> ${formatParamName(result.param_name)}</div>
      <div class="meta-item"><strong>Description:</strong> ${result.param_description}</div>
      <div class="meta-item"><strong>Values Tested:</strong> ${result.num_values}</div>
      <div class="meta-item"><strong>Execution Time:</strong> ${(result.execution_time_ms / 1000).toFixed(1)}s</div>
    `;
  }
}
