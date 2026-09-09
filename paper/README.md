# MORPH-32 technical report

Editable Typst source and a compiled scientific report about the existing MORPH-32 experiments. Building the paper does not load a model or run benchmarks.

## Build

From the repository root:

```bash
uv run --no-project --with typst==0.15.0 python paper/build.py
```

Output: `paper/output/pdf/morph32.pdf`.

The source uses Times New Roman, available on the measured Mac. Install that family on other systems, or change the explicit font setting in `main.typ`. The mathematical font is supplied by Typst. Figures are vector SVGs with embedded glyph outlines.

To regenerate the figure:

```bash
uv run --no-project --with matplotlib==3.10.8 python paper/plot.py
```

The figure uses the frozen `data/summary.json`. `data/provenance.json` records hashes of the source records. Main result tables read the same snapshot directly. Process-footprint numbers are excluded at the author's request; memory measurements are explicitly labeled as resident arrays or MLX allocation.

## Layout and editorial choices

The report imports [`@preview/bloated-neurips:0.8.0`](https://typst.app/universe/package/bloated-neurips/) by Daniel Bershatsky, from the [daskol/typst-templates](https://github.com/daskol/typst-templates) collection. It uses the actual `neurips2026` show rule in preprint mode (`accepted: none`), not a locally recreated layout. Typst downloads this pinned MIT-licensed package on the first build.

The six-page report uses the template's US Letter pages, single 5.5-inch text column, 10 pt Times body, 17 pt title between horizontal rules, and normal heading and caption rules. Short subtopics use the template's `paragraph` helper. References use the permitted 9 pt size. The project byline avoids inventing author names or affiliations. This is a community NeurIPS-style template; it does not imply conference submission, acceptance, or endorsement.

Writing guidance reviewed:

- [Nature Methods: So you're writing a paper](https://www.nature.com/articles/nmeth.4532): make observations, methods, and interpretation easy to distinguish.
- [Nature: Scientific writing 101](https://www.nature.com/articles/nsmb0210-139): use plain language and keep the introduction focused.
- [Wikipedia: Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing): remove inflated significance, vague attribution, stock contrasts, decorative formatting, and repetitive conclusions. This is an editing reference, not a reliable authorship detector.

The paper uses concrete methods and recorded outcomes, including the speed cost. It does not claim quality equivalence, scientific priority, or peer review. The literature sources support the related-work discussion; the local evidence supports the results. Author names and affiliations can be added before circulation.

The manuscript uses archived measurements. Profile labels were standardized across the repository; no model benchmarks were rerun for the rename.
