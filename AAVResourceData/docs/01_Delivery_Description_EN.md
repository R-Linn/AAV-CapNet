# Delivery Description

## 1. Delivery Objective

The objective of this delivery is to organize, filter, merge, and document AAV variant information, AAV-to-cancer-cell-line assay information, and relevant CCLE cancer cell line features from publicly verifiable sources into a usable, extensible, and source-traceable research-oriented delivery package.

## 2. Delivered Contents

### AAV Sequence Table

File: `data/aav_sequence_base.csv`

Main contents:

- AAV variant or serotype name
- full amino acid sequence
- sequence length
- evidence level
- source bucket
- accession / direct-source verification status
- notes

### AAV-to-Cell-Line Assay Pairing Table

File: `data/aav_assay_pairs.csv`

Main contents:

- AAV variant
- original cell line name
- normalized cell line match
- cancer context
- assay metric type
- raw assay value and unit
- metric family
- comparable scope
- source reference and notes

### AAV Support-Layer Table

File: `data/aav_fit4function_support.csv`

Main contents:

- public support-layer insertion library data
- retained separately from the main full-length table
- intended for support and extension analyses, not for inflating main-table completion

### CCLE Feature Table

File: `data/ccle_features.csv`

Main contents:

- cell line ID
- DepMap ID
- cancer type / subtype / tissue origin
- expression fields
- RPPA fields
- release and unit notes

### Source and Delivery Summary

Files:

- `meta/source_manifest_client.csv`
- `meta/delivery_summary.json`
- `meta/data_checksums.sha256`

## 3. Core Value Of This Delivery

The value of this delivery is not just table assembly. It is:

- restricted to publicly verifiable sources
- explicit about source and evidence boundaries
- careful to preserve original assay semantics when direct comparison is not justified
- strict about not inventing missing sequences or values
- suitable as a clear starting base for downstream research and extension

## 4. Intended Use

This delivery is suitable for:

- research preparation and literature-backed data organization
- AAV variant and cancer cell line indexing
- preliminary AAV × cell line analysis baselining
- further extension with additional public sources

