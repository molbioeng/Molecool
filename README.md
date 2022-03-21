# Molecool

This is a repository for the programming part of our year 3 Group project analysing HRV values.

Processed data can be found in our Google drive:

Code is separated into modules:
  1. Initial processing - initial viewing of the database, loading in the relevant parts, and converting to a standard format
  2. Snippet extraction - Extracting 5 minute snippets of 'clean' ecg data for peak detection
  3. R-peak detection - HRV extraction through applying the Pan-Tompkins algorithm, noise removal also occures here as part of the PT algorithm
  4. HRV Parameter Extraction - extracting HRV data from detected peaks, plotting frequency and time domain graphs, and extracting relevant HRV parameters
  5. Parameter analysis - Statistical analysis of HRV parameters and CAN variables

Other files (last 2 are stored in the Unmodularised_code folder):
  - n.0.ipynb stores the functions used in module n
  - Filter testing - checking how filters + other methods affect the signal
  - Cross correlation matrices - observing correlations between variables within our data to compare with literature
