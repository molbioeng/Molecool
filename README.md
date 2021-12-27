# Molecool

This is a repository for the programming part of our year 3 Group project analysing HRV values.

Original and processed data can be found in the 'database' folder.

Code is separated into a couple of files:
  1. Initial processing - initial viewing of the data, loading in the relevant parts, extracting 5 minutes of 'clean' ECG signal for HRV analysis
  2. R-peak detection - HRV extraction through applying the Pan-Tompkins algorithm, noise removal also occure here as part of the PT algorithm
  3. HRV graphs + analysis

Other files:
  - Filter testing - checking how filters + other methods affect the signal
  - Cross correlation matrices - observing correlations between variables within our data to compare with literature
