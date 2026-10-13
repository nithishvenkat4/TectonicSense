import React, { useEffect, useRef, useState } from 'react';
import { Map, Marker, NavigationControl, setWorkerUrl } from 'maplibre-gl';
import maplibreglWorkerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?url';
import type { MapMouseEvent } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { PB2002_CLASSES } from '../data/constants';
import type { PB2002Class } from '../types';

setWorkerUrl(maplibreglWorkerUrl);

interface TectonicMapProps {
  latitude: number;
  longitude: number;
  onCoordinateChange?: (lat: number, lon: number) => void;
  predictedClass?: PB2002Class | null;
}

export const TectonicMap: React.FC<TectonicMapProps> = ({
  latitude,
  longitude,
  onCoordinateChange,
  predictedClass,
}) => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const mapRef = useRef<Map | null>(null);
  const markerRef = useRef<Marker | null>(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [showLegend, setShowLegend] = useState(true);

  // Initialize Map
  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return;

    const map = new Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          'esri-light': {
            type: 'raster',
            tiles: [
              'https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
            ],
            tileSize: 256,
            attribution: '© Esri, DeLorme, NAVTEQ, PB2002',
          },
        },
        layers: [
          {
            id: 'esri-light-layer',
            type: 'raster',
            source: 'esri-light',
            minzoom: 0,
            maxzoom: 16,
          },
        ],
      },
      center: [isNaN(longitude) ? 0 : longitude, isNaN(latitude) ? 20 : latitude],
      zoom: 1.8,
      minZoom: 1,
      maxZoom: 12,
      renderWorldCopies: true,
    });

    map.addControl(new NavigationControl({ showCompass: false }), 'top-right');

    map.on('load', () => {
      // Add PB2002 GeoJSON lines
      map.addSource('pb2002-boundaries', {
        type: 'geojson',
        data: '/pb2002_boundaries.geojson',
      });

      map.addLayer({
        id: 'pb2002-lines-casing',
        type: 'line',
        source: 'pb2002-boundaries',
        paint: {
          'line-color': '#ffffff',
          'line-width': 4,
          'line-opacity': 0.7,
        },
      });

      map.addLayer({
        id: 'pb2002-lines',
        type: 'line',
        source: 'pb2002-boundaries',
        paint: {
          'line-color': [
            'match',
            ['get', 'class'],
            'SUB', PB2002_CLASSES.SUB.color,
            'OSR', PB2002_CLASSES.OSR.color,
            'OTF', PB2002_CLASSES.OTF.color,
            'CRB', PB2002_CLASSES.CRB.color,
            'CTF', PB2002_CLASSES.CTF.color,
            'CCB', PB2002_CLASSES.CCB.color,
            'OCB', PB2002_CLASSES.OCB.color,
            '#334155', // fallback slate
          ],
          'line-width': 2.2,
          'line-opacity': 0.9,
        },
      });

      // Add observation marker
      const el = document.createElement('div');
      el.className = 'custom-earthquake-marker';
      el.style.width = '20px';
      el.style.height = '20px';
      el.style.borderRadius = '50%';
      el.style.backgroundColor = '#ea580c';
      el.style.border = '3px solid #ffffff';
      el.style.boxShadow = '0 1px 4px rgba(0,0,0,0.4)';
      el.style.cursor = 'pointer';

      const marker = new Marker({ element: el })
        .setLngLat([longitude, latitude])
        .addTo(map);

      markerRef.current = marker;
      setMapLoaded(true);
    });

    // Map click updates coordinates if callback provided
    map.on('click', (e: MapMouseEvent) => {
      if (onCoordinateChange) {
        const clickedLng = Number(e.lngLat.lng.toFixed(4));
        const clickedLat = Number(e.lngLat.lat.toFixed(4));
        onCoordinateChange(clickedLat, clickedLng);
      }
    });

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Sync marker position when lat/lng change
  useEffect(() => {
    if (!markerRef.current || !mapLoaded || isNaN(latitude) || isNaN(longitude)) return;

    markerRef.current.setLngLat([longitude, latitude]);

    // Update marker styling if prediction exists
    const markerEl = markerRef.current.getElement();
    if (predictedClass && PB2002_CLASSES[predictedClass]) {
      markerEl.style.backgroundColor = PB2002_CLASSES[predictedClass].color;
    } else {
      markerEl.style.backgroundColor = '#ea580c';
    }
  }, [latitude, longitude, mapLoaded, predictedClass]);

  const recenterMap = () => {
    if (mapRef.current && !isNaN(latitude) && !isNaN(longitude)) {
      mapRef.current.flyTo({
        center: [longitude, latitude],
        zoom: 4,
        essential: true,
      });
    }
  };

  return (
    <div className="tectonic-map-container">
      {/* Map Control Bar */}
      <div className="map-control-bar">
        <div className="map-control-left">
          <span className="map-control-title">
            PB2002 Tectonic Plate Boundaries
          </span>
          <span className="map-control-sep">|</span>
          <span className="map-control-hint">Click map to set coordinates</span>
        </div>
        <div className="map-control-right">
          <button
            id="btn-recenter-map"
            type="button"
            onClick={recenterMap}
            className="map-control-btn"
            title="Focus on observation epicenter"
          >
            Center on Event
          </button>
          <button
            id="btn-toggle-legend"
            type="button"
            onClick={() => setShowLegend(!showLegend)}
            className="map-control-btn"
          >
            {showLegend ? 'Hide Legend' : 'Legend'}
          </button>
        </div>
      </div>

      {/* Map Canvas Container */}
      <div ref={mapContainer} className="map-canvas" />

      {/* Map Legend Overlay */}
      {showLegend && (
        <div className="map-legend-overlay">
          <div className="legend-title">
            PB2002 Boundary Legend
          </div>
          <div className="legend-grid">
            {(Object.keys(PB2002_CLASSES) as PB2002Class[]).map((cls) => {
              const meta = PB2002_CLASSES[cls];
              return (
                <div key={cls} className="legend-item">
                  <span
                    className="legend-color-bar"
                    style={{ backgroundColor: meta.color }}
                  />
                  <span className="legend-code font-mono">
                    {cls}
                  </span>
                  <span className="legend-name">
                    {meta.name.split(' ')[0]}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="legend-footer">
            <span className="legend-epicenter-dot" />
            <span>Earthquake Observation Epicenter</span>
          </div>
        </div>
      )}
    </div>
  );
};
