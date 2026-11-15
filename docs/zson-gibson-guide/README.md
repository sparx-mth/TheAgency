# ZSON / Gibson end-to-end guide

**Start with [complete.html](complete.html)** — the complete illustrated guide on one page, with linked contents, expandable details and print/save-PDF support.

[index.html](index.html) offers shorter chapter views. Use the complete guide's print/save-PDF button for a self-contained PDF; the generated export is kept local and is not versioned.

## Coverage

- Actual frontier and bounded planar FALCON execution paths, including planner responsibilities, RPT/LLM call triggers, action clocks, target interrupts and no-room/single-room behavior.
- Gibson/Habitat sensors, legal actions, real GT arrays, the private scoring boundary, train/validation/test distinctions and exact SemExp-derived SR/SPL/DTG semantics.
- Real action/burst walkthroughs and the complete 15-building training-development comparison.
- The three supplied papers, with numerical comparison restricted to Gibson. OSG Table 2 is kept separate from our development results; no HM3D GT-variant score is mislabeled Gibson.
- Fifteen evidence-labeled limitations, proposed improvements and application to broader ZSON goals.
- End-to-end source references and a glossary, without raw code listings.

## Viewing and portability

Open the HTML directly in a browser; no server, dependency installation or network service is required. Keep `assets/` beside the HTML. The complete page embeds its styles; chapter views use `style.css`.

The PDF is self-contained. Optional links to source files and original run dashboards require the checkout or this workstation's existing data directories. The essential explanations, tables and included illustrations do not require those links.

## Audit snapshot

- Date: 15 September 2026.
- Checkout: `feat/objnav-habitat-gibson-nadav`, commit `f18cc5a4`.
- Audited Python-source fingerprint begins `6ddb9daf`, matching the saved 15-building campaign.
- 1,892 relevant CPU-only tests passed; 19 non-failing warnings.
- All HTML tags, local links, anchors, image references and encodings checked.
- Headless-browser validation passed at desktop and 390-pixel mobile widths; all four images decoded; detail expansion and complete PDF printing were exercised.


This is documentation and analysis, not an algorithm change or a new full benchmark run. Proposed fixes are not presented as implemented. PDF figure-viewing limitations, simulator-version differences, known validation exposure and missing current full-validation results are explicitly disclosed in the guide.

## Assets

The GT illustration was generated from the installed Collierville validation floor array. Sensor/dashboard images were decoded or copied from existing Shelbyville/Newfields training-development recordings. No licensed scene meshes, full datasets, model weights, complete research PDFs or third-party implementation code are bundled.

