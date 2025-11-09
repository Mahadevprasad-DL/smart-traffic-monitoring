import {
  bangaloreAreas,
  getTrafficDataForArea,
  getHourlyTrafficData,
  getVehicleTypeDistribution,
  getWeeklyTrend
} from './bangaloreAreas.js';

const areaSelect = document.getElementById('areaSelect');

// Populate dropdown
bangaloreAreas.forEach(area => {
  const option = document.createElement('option');
  option.value = area;
  option.textContent = area;
  areaSelect.appendChild(option);
});

// Chart.js instances
let congestionChart, vehicleChart, weeklyChart;

function createOrUpdateChart(ctx, data, label, bgColor, borderColor, type = 'bar') {
  if (ctx.chart) ctx.chart.destroy();

  ctx.chart = new Chart(ctx, {
    type: type,
    data: {
      labels: data.labels,
      datasets: [{
        label: label,
        data: data.values,
        backgroundColor: bgColor,
        borderColor: borderColor,
        borderWidth: 2,
        fill: type === 'line'
      }]
    },
    options: {
      responsive: true,
      plugins: {
        tooltip: {
          enabled: true,
          backgroundColor: '#1e293b',
          titleColor: '#f8fafc',
          bodyColor: '#f97316',
          borderColor: '#f97316',
          borderWidth: 1
        },
        legend: { labels: { color: '#f8fafc' } }
      },
      scales: {
        x: { ticks: { color: '#94a3b8' } },
        y: { ticks: { color: '#94a3b8' } }
      }
    }
  });
}

// When area is selected
areaSelect.addEventListener('change', () => {
  const selectedArea = areaSelect.value;
  if (!selectedArea) return;

  const hourlyData = getHourlyTrafficData(selectedArea);
  const vehicleData = getVehicleTypeDistribution(selectedArea);
  const weeklyData = getWeeklyTrend(selectedArea);

  // Congestion chart
  const congestionCtx = document.getElementById('congestionChart').getContext('2d');
  createOrUpdateChart(
    congestionCtx,
    {
      labels: hourlyData.map(d => d.hour),
      values: hourlyData.map(d => d.congestion)
    },
    `Hourly Congestion - ${selectedArea}`,
    'rgba(249, 115, 22, 0.5)',
    '#f97316'
  );

  // Vehicle type chart
  const vehicleCtx = document.getElementById('vehicleChart').getContext('2d');
  createOrUpdateChart(
    vehicleCtx,
    {
      labels: vehicleData.map(v => v.type),
      values: vehicleData.map(v => v.count)
    },
    `Vehicle Distribution - ${selectedArea}`,
    [
      'rgba(59, 130, 246, 0.6)',
      'rgba(34, 197, 94, 0.6)',
      'rgba(244, 63, 94, 0.6)',
      'rgba(234, 179, 8, 0.6)',
      'rgba(168, 85, 247, 0.6)'
    ],
    '#0ea5e9',
    'pie'
  );

  // Weekly trend chart
  const weeklyCtx = document.getElementById('weeklyChart').getContext('2d');
  createOrUpdateChart(
    weeklyCtx,
    {
      labels: weeklyData.map(w => w.day),
      values: weeklyData.map(w => w.avgCongestion)
    },
    `Weekly Congestion Trend - ${selectedArea}`,
    'rgba(99, 102, 241, 0.5)',
    '#6366f1',
    'line'
  );
});
