# Limitations

## 1. Source Boundary

This delivery is restricted to publicly verifiable sources.  
If a piece of information is not explicitly provided in the public source record, it is not artificially filled in.

## 2. Experimental System Differences

Experimental systems differ across papers, including:

- MOI or dose settings
- reporter systems
- timepoints
- readout methods
- cell culture conditions

Therefore:

- many assay values can be preserved in one table for reference
- but not all of them should be interpreted as directly comparable under a single strict quantitative scale

## 3. Sequence Completeness

Not all engineered AAV studies publicly provide full-length sequences.  
Some records only expose:

- parent anchors
- mutation lists
- region-level evidence
- barcode / library information
- assay-first literature evidence

Therefore:

- not every record is guaranteed to contain a complete full-length sequence
- missing items are labeled as such rather than artificially reconstructed beyond the evidence boundary

## 4. Boundary Of Support-Layer Data

`aav_fit4function_support.csv` is a support-layer table retained for support and extension analyses.  
It should not be interpreted as direct completion of the full-length main table.

## 5. Current Maturity Boundary

This delivery is suitable as a research-grade integration base, but it should not be overstated as:

- a fully unit-harmonized benchmark dataset
- a complete coverage resource for all high-value engineered AAV variants
- a final integrated atlas sufficient on its own for strong statistical conclusions

