export function getDistance(v1: number[], v2: number[]): number {
  let sum = 0;
  for (let i = 0; i < v1.length; i++) {
    sum += Math.pow(v1[i] - v2[i], 2);
  }
  return Math.sqrt(sum);
}

export function cosineSimilarity(v1: number[], v2: number[]): number {
  let dotProduct = 0;
  let m1 = 0;
  let m2 = 0;
  for (let i = 0; i < v1.length; i++) {
    dotProduct += v1[i] * v2[i];
    m1 += v1[i] * v1[i];
    m2 += v2[i] * v2[i];
  }
  if (m1 === 0 || m2 === 0) return 0;
  return dotProduct / (Math.sqrt(m1) * Math.sqrt(m2));
}

export function chooseDistantPoints(data: number[][], col: number = 0): [number, number] {
  if (data.length === 0) return [0, 0];
  let a = 0;
  let maxDist = 0;
  let b = 0;
  
  for (let i = 0; i < data.length; i++) {
    const d = getDistance(data[a], data[i]);
    if (d > maxDist) {
      maxDist = d;
      b = i;
    }
  }
  
  a = b;
  maxDist = 0;
  for (let i = 0; i < data.length; i++) {
    const d = getDistance(data[a], data[i]);
    if (d > maxDist) {
      maxDist = d;
      b = i;
    }
  }
  
  if (maxDist === 0) {
      return [0, Math.min(1, data.length - 1)];
  }
  
  return [a, b];
}

export function fastMap2D(data: number[][]): { x: number, y: number }[] {
  if (data.length === 0) return [];
  if (data.length === 1) return [{ x: 0.5, y: 0.5 }];
  if (data.length === 2) return [{ x: 0.1, y: 0.5 }, { x: 0.9, y: 0.5 }];

  const [a1, b1] = chooseDistantPoints(data);
  const distA1B1 = getDistance(data[a1], data[b1]);
  
  const x = data.map((_, i) => {
    if (i === a1) return 0;
    if (i === b1) return distA1B1;
    const diA1 = getDistance(data[i], data[a1]);
    const diB1 = getDistance(data[i], data[b1]);
    if (distA1B1 === 0) return 0;
    return (diA1 * diA1 + distA1B1 * distA1B1 - diB1 * diB1) / (2 * distA1B1);
  });

  const dist2 = (i: number, j: number) => {
    const d2 = Math.pow(getDistance(data[i], data[j]), 2) - Math.pow(x[i] - x[j], 2);
    return Math.sqrt(Math.max(0, d2));
  }

  let a2 = 0;
  let maxD2 = 0;
  let b2 = 0;
  for (let i = 0; i < data.length; i++) {
    const d = dist2(a2, i);
    if (d > maxD2) {
      maxD2 = d;
      b2 = i;
    }
  }
  a2 = b2;
  maxD2 = 0;
  for (let i = 0; i < data.length; i++) {
    const d = dist2(a2, i);
    if (d > maxD2) {
      maxD2 = d;
      b2 = i;
    }
  }

  const distA2B2 = dist2(a2, b2);
  const y = data.map((_, i) => {
    if (i === a2) return 0;
    if (i === b2) return distA2B2;
    const diA2 = dist2(i, a2);
    const diB2 = dist2(i, b2);
    if (distA2B2 === 0) return 0;
    return (diA2 * diA2 + distA2B2 * distA2B2 - diB2 * diB2) / (2 * distA2B2);
  });

  const minX = Math.min(...x);
  const maxX = Math.max(...x);
  const minY = Math.min(...y);
  const maxY = Math.max(...y);

  const rangeX = Math.max(maxX - minX, 1e-6);
  const rangeY = Math.max(maxY - minY, 1e-6);
  const maxRange = Math.max(rangeX, rangeY);

  return x.map((xv, i) => ({
    x: 0.5 + (((xv - minX) - (maxX - minX)/2) / maxRange) * 0.8,
    y: 0.5 + (((y[i] - minY) - (maxY - minY)/2) / maxRange) * 0.8
  }));
}

export function kMeans(data: { x: number, y: number }[], k: number, maxIterations = 100) {
  if (data.length === 0) return [];
  if (data.length <= k) return data.map((_, i) => i);

  let centroids = data.slice(0, k).map(p => ({ x: p.x, y: p.y }));
  let assignments = new Array(data.length).fill(0);

  for (let iter = 0; iter < maxIterations; iter++) {
    let changed = false;

    for (let i = 0; i < data.length; i++) {
      let minDist = Infinity;
      let cluster = 0;
      for (let j = 0; j < k; j++) {
        const dist = Math.pow(data[i].x - centroids[j].x, 2) + Math.pow(data[i].y - centroids[j].y, 2);
        if (dist < minDist) {
          minDist = dist;
          cluster = j;
        }
      }
      if (assignments[i] !== cluster) {
        assignments[i] = cluster;
        changed = true;
      }
    }

    if (!changed) break;

    const counts = new Array(k).fill(0);
    const newCentroids = new Array(k).fill(0).map(() => ({ x: 0, y: 0 }));
    
    for (let i = 0; i < data.length; i++) {
      const cluster = assignments[i];
      newCentroids[cluster].x += data[i].x;
      newCentroids[cluster].y += data[i].y;
      counts[cluster]++;
    }

    for (let j = 0; j < k; j++) {
      if (counts[j] > 0) {
        centroids[j].x = newCentroids[j].x / counts[j];
        centroids[j].y = newCentroids[j].y / counts[j];
      }
    }
  }

  return assignments;
}
