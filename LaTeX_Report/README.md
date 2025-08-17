# CGNet LaTeX Report

This directory contains the complete LaTeX source files for the CGNet (Change Guiding Network) research report.

## Files Structure

### Main Document
- `CGNet_Report_2025.tex` - Main LaTeX document
- `cgnet_title_page.tex` - Title page
- `cgnet_abstract.tex` - Abstract
- `CGNet_Bibliography.bib` - Bibliography with 40+ references

### Chapters
- `chapters/cgnet/introduction.tex` - Introduction and background
- `chapters/cgnet/literature_review.tex` - Literature review
- `chapters/cgnet/methodology.tex` - CGNet architecture and methodology
- `chapters/cgnet/dataset_analysis.tex` - Dataset analysis (LEVIR-CD-256, SYSU-CD)
- `chapters/cgnet/results_discussion.tex` - Results and discussion
- `chapters/cgnet/conclusion.tex` - Conclusion and future work

## Compilation Instructions

### Prerequisites
- MiKTeX (LaTeX distribution)
- pdflatex compiler
- bibtex for bibliography

### Compilation Steps
1. Compile LaTeX: `pdflatex CGNet_Report_2025.tex`
2. Generate bibliography: `bibtex CGNet_Report_2025`
3. Compile LaTeX twice more: 
   - `pdflatex CGNet_Report_2025.tex`
   - `pdflatex CGNet_Report_2025.tex`

### VS Code (with LaTeX Workshop)
1. Open `CGNet_Report_2025.tex`
2. Press `Ctrl+Alt+B` to build
3. Press `Ctrl+Alt+V` to view PDF

## Document Features

- **Academic Quality**: Professional formatting using pkmthesis class
- **Comprehensive Content**: 6 main chapters with detailed technical content
- **Technical Elements**: 15+ tables, 5+ algorithms, equations, and diagrams
- **Bibliography**: 40+ academic references in IEEE format
- **Page Count**: Expected 50+ pages when compiled

## Figure Requirements

The document references several figures that need to be added to the `figures/` directory:
- `cgnet_architecture.png` - CGNet architecture diagram
- `change_detection_examples.png` - Example results
- `dataset_samples.png` - Dataset sample images
- `training_curves.png` - Training performance curves
- And others (see `figures/placeholder_figures.txt`)

## Notes

- All technical content is complete and ready for compilation
- Figures are referenced but need to be added as actual image files
- The document follows academic standards for research reports
- Content covers the complete CGNet project from dataset analysis to results

## Related Files

This LaTeX report complements the following files in the main repository:
- `COMPREHENSIVE_DATASET_REPORT.md` - Detailed dataset analysis
- `analyze_dataset.py` - Dataset analysis script
- `network/CGNet.py` - Model implementation
- Other project files

For questions or modifications, refer to the main project documentation.
