export const bangaloreAreas = [
  'Whitefield',
  'Electronic City',
  'Koramangala',
  'Indiranagar',
  'Jayanagar',
  'Marathahalli',
  'HSR Layout',
  'BTM Layout',
  'Bellandur',
  'Sarjapur Road',
  'Bannerghatta Road',
  'Hebbal',
  'Yeshwanthpur',
  'Rajajinagar',
  'Malleshwaram',
  'MG Road',
  'Brigade Road',
  'Silk Board',
  'KR Puram',
  'Yelahanka',
  'JP Nagar',
  'Banashankari',
  'Vijayanagar',
  'Kengeri',
  'Peenya',
  'RT Nagar',
  'Kalyan Nagar',
  'CV Raman Nagar',
  'Domlur',
  'Old Airport Road'
].sort();

/**
 * Generate traffic data for a specific area
 */
export const getTrafficDataForArea = (area) => {
  const seed = area.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);

  const random = (min, max, index) => {
    const x = Math.sin(seed + index) * 10000;
    return min + ((x - Math.floor(x)) * (max - min));
  };

  return {
    area,
    peakHourCongestion: Math.round(random(30, 95, 1)),
    avgSpeed: Math.round(random(15, 45, 2)),
    vehicleCount: Math.round(random(5000, 25000, 3)),
    accidentRate: Math.round(random(2, 15, 4) * 10) / 10,
    airQualityIndex: Math.round(random(50, 250, 5))
  };
};

/**
 * Generate hourly traffic data for a given area
 */
export const getHourlyTrafficData = (area) => {
  const seed = area.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const hours = ['6 AM', '8 AM', '10 AM', '12 PM', '2 PM', '4 PM', '6 PM', '8 PM', '10 PM'];

  return hours.map((hour, index) => {
    const x = Math.sin(seed + index) * 10000;
    const normalized = x - Math.floor(x);
    const peakMultiplier = (index === 1 || index === 6) ? 1.5 : 1;

    return {
      hour,
      congestion: Math.round(normalized * 60 * peakMultiplier + 30),
      speed: Math.round((1 - normalized) * 30 + 15),
      vehicles: Math.round(normalized * 15000 * peakMultiplier + 5000)
    };
  });
};

/**
 * Generate distribution of vehicle types for an area
 */
export const getVehicleTypeDistribution = (area) => {
  const seed = area.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const random = (index) => {
    const x = Math.sin(seed + index) * 10000;
    return x - Math.floor(x);
  };

  const base = [
    { type: 'Cars', count: random(1) * 5000 + 8000 },
    { type: 'Bikes', count: random(2) * 6000 + 10000 },
    { type: 'Buses', count: random(3) * 1000 + 1500 },
    { type: 'Auto Rickshaws', count: random(4) * 3000 + 4000 },
    { type: 'Trucks', count: random(5) * 2000 + 2000 }
  ];

  return base.map(item => ({
    ...item,
    count: Math.round(item.count)
  }));
};

/**
 * Generate weekly congestion and incidents trend
 */
export const getWeeklyTrend = (area) => {
  const seed = area.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

  return days.map((day, index) => {
    const x = Math.sin(seed + index * 10) * 10000;
    const normalized = x - Math.floor(x);
    const weekendMultiplier = (index >= 5) ? 0.7 : 1;

    return {
      day,
      avgCongestion: Math.round(normalized * 50 * weekendMultiplier + 40),
      incidents: Math.round(normalized * 8 * weekendMultiplier + 2)
    };
  });
};
