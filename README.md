# Molecool

This is a repository for the programming part of our year 3 Group project analysing HRV values.

Original and processed data can be found in the 'database' folder.

Code is separated into a couple of files:
  1. Initial data cleaning - initial viewing of the data, loading in the relevant parts
  2. Data extraction - Extracting 5 minutes of 'clean' ECG signal for HRV analysis
  3. Error removal - Removing noise for better R-peak detection
  4. R-peak detection - HRV extraction through applying the Pan-Tompkins algorithm
  5. HRV graphs + analysis

Other files:
  - Filter testing - checking how filters + other methods affect the signal
