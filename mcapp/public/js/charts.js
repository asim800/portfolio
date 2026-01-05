/**
 * Chart.js Rendering Functions for Monte Carlo Simulation
 *
 * Renders fan charts showing percentile bands over time
 */

// Chart instances
let accChartInstance = null;
let decChartInstance = null;

/**
 * Render a fan chart showing percentile bands
 * @param {string} canvasId - ID of the canvas element
 * @param {Array} data - Array of TimeSeriesPoint objects
 * @param {string} title - Chart title
 */
function renderFanChart(canvasId, data, title) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');

  // Destroy existing chart if it exists
  if (canvasId === 'acc-chart' && accChartInstance) {
    accChartInstance.destroy();
  }
  if (canvasId === 'dec-chart' && decChartInstance) {
    decChartInstance.destroy();
  }

  // Extract data series
  const labels = data.map(d => d.date);
  const p5 = data.map(d => d.p5);
  const p25 = data.map(d => d.p25);
  const p50 = data.map(d => d.p50);
  const p75 = data.map(d => d.p75);
  const p95 = data.map(d => d.p95);
  const mean = data.map(d => d.mean);

  const chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        // 95th percentile (top band)
        {
          label: '95th Percentile',
          data: p95,
          borderColor: 'rgba(33, 150, 243, 0.3)',
          backgroundColor: 'rgba(33, 150, 243, 0.1)',
          fill: '+1',
          pointRadius: 0,
          borderWidth: 1,
        },
        // 75th percentile
        {
          label: '75th Percentile',
          data: p75,
          borderColor: 'rgba(33, 150, 243, 0.5)',
          backgroundColor: 'rgba(33, 150, 243, 0.2)',
          fill: '+1',
          pointRadius: 0,
          borderWidth: 1,
        },
        // Median (50th percentile) - highlighted
        {
          label: 'Median (50th)',
          data: p50,
          borderColor: 'rgba(25, 118, 210, 1)',
          backgroundColor: 'rgba(33, 150, 243, 0.3)',
          fill: '+1',
          pointRadius: 0,
          borderWidth: 2,
        },
        // 25th percentile
        {
          label: '25th Percentile',
          data: p25,
          borderColor: 'rgba(33, 150, 243, 0.5)',
          backgroundColor: 'rgba(33, 150, 243, 0.2)',
          fill: '+1',
          pointRadius: 0,
          borderWidth: 1,
        },
        // 5th percentile (bottom band)
        {
          label: '5th Percentile',
          data: p5,
          borderColor: 'rgba(33, 150, 243, 0.3)',
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          borderWidth: 1,
        },
        // Mean line (if available)
        ...(mean[0] !== null ? [{
          label: 'Mean',
          data: mean,
          borderColor: 'rgba(255, 152, 0, 0.8)',
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          borderWidth: 2,
          borderDash: [5, 5],
        }] : []),
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      interaction: {
        intersect: false,
        mode: 'index',
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            usePointStyle: true,
            boxWidth: 10,
          },
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const value = context.parsed.y;
              return `${context.dataset.label}: ${formatCurrency(value)}`;
            },
          },
        },
      },
      scales: {
        x: {
          display: true,
          title: {
            display: true,
            text: 'Date',
          },
          ticks: {
            maxTicksLimit: 10,
            maxRotation: 45,
          },
        },
        y: {
          display: true,
          title: {
            display: true,
            text: 'Portfolio Value ($)',
          },
          ticks: {
            callback: function(value) {
              return formatCurrency(value);
            },
          },
        },
      },
    },
  });

  // Store reference
  if (canvasId === 'acc-chart') {
    accChartInstance = chart;
  } else {
    decChartInstance = chart;
  }

  return chart;
}

/**
 * Render distribution histogram
 * @param {string} canvasId - ID of the canvas element
 * @param {Array} bins - Array of DistributionBin objects
 * @param {string} title - Chart title
 */
function renderDistribution(canvasId, bins, title) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');

  const labels = bins.map(b => formatCurrency((b.bin_start + b.bin_end) / 2));
  const counts = bins.map(b => b.count);

  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Frequency',
        data: counts,
        backgroundColor: 'rgba(33, 150, 243, 0.6)',
        borderColor: 'rgba(25, 118, 210, 1)',
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          display: false,
        },
        title: {
          display: true,
          text: title,
        },
      },
      scales: {
        x: {
          display: true,
          title: {
            display: true,
            text: 'Portfolio Value',
          },
        },
        y: {
          display: true,
          title: {
            display: true,
            text: 'Count',
          },
        },
      },
    },
  });
}

/**
 * Format number as currency
 */
function formatCurrency(value) {
  if (Math.abs(value) >= 1000000) {
    return '$' + (value / 1000000).toFixed(1) + 'M';
  } else if (Math.abs(value) >= 1000) {
    return '$' + (value / 1000).toFixed(0) + 'K';
  }
  return '$' + value.toFixed(0);
}
