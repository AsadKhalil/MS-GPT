# MSQA-Bench Paper Build
# Usage: make (builds PDF), make clean (removes build artifacts)

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib neurips_2026.sty
	$(LATEX) -interaction=nonstopmode $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) -interaction=nonstopmode $(MAIN)
	$(LATEX) -interaction=nonstopmode $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).pdf $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).nav \
	      $(MAIN).snm $(MAIN).vrb $(MAIN).synctex.gz
