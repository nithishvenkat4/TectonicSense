# Dataset Documentation

This document will record the dataset sources, schema, preprocessing, spatial association, quality checks, and final ML dataset construction.

## Planned Sources

### USGS Earthquake Data
Used for seismic event information such as location, time, depth, and magnitude. Availability of additional focal-mechanism features will be assessed during dataset inspection.

### PB2002
Used as the global tectonic plate-boundary reference layer.

## Dataset Construction Plan

Earthquake event
→ spatial nearest-boundary association
→ boundary attributes
→ feature engineering
→ target-label construction
→ quality checks
→ final ML dataset

## Important

Final target classes and feature availability will be decided only after inspecting the source datasets.
