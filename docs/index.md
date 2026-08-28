---
hide:
  - navigation
  - toc
---

<section class="bsx-hero" markdown>
<div class="bsx-hero__copy" markdown>
<p class="bsx-hero__eyebrow">Pose-assisted behavior analysis</p>
<h1 class="bsx-hero__title">BehaviorScope<span>-X</span></h1>
<p class="bsx-hero__lede">
Turn animal-behavior recordings into reviewable temporal predictions while
keeping annotation, data splits, pose features, model training, and evaluation
visible at every step.
</p>
<div class="bsx-hero__actions" markdown>
[Install BehaviorScope-X](installation.md){ .md-button .md-button--primary }
[Explore the workflow](#research-workflow){ .md-button }
[Read the preprint](citation.md){ .md-button .bsx-preprint-button }
</div>
<div class="bsx-specs">
  <span><b>Desktop</b> annotation</span>
  <span><b>3</b> pose paths</span>
  <span><b>Explicit</b> data splits</span>
  <span><b>Reviewable</b> outputs</span>
</div>
</div>

<div class="bsx-hero__visual" aria-label="BehaviorScope-X workflow summary">
  <div class="bsx-console__bar"><span></span><span></span><span></span><b>RESEARCH WORKFLOW</b></div>
  <div class="bsx-console__stage">
    <span class="bsx-console__number">01</span>
    <div><b>CURATE</b><small>label bouts · assign splits · quality control</small></div>
  </div>
  <div class="bsx-console__connector"></div>
  <div class="bsx-console__stage">
    <span class="bsx-console__number">02</span>
    <div><b>TRAIN</b><small>build pose cache · train · validate · test</small></div>
  </div>
  <div class="bsx-console__connector"></div>
  <div class="bsx-console__stage">
    <span class="bsx-console__number">03</span>
    <div><b>INTERPRET</b><small>predictions · ethograms · review videos</small></div>
  </div>
  <div class="bsx-console__status"><span></span> EVIDENCE READY FOR REVIEW</div>
</div>
</section>

## One interface, three pose paths

Choose the representation that matches your study. Every path returns to the
same temporal behavior workflow, so annotation, training, held-out testing, and
output review remain comparable.

<div class="bsx-path-grid">
  <article>
    <span class="bsx-path-grid__index">01 / DIRECT</span>
    <h3><a href="yolo-pose-workflow/">YOLO-pose</a></h3>
    <p>Build caches, train the classifier, run inference, and create annotated review videos from the GUI-centered path.</p>
    <a class="bsx-text-link" href="yolo-pose-workflow/">Open workflow →</a>
  </article>
  <article>
    <span class="bsx-path-grid__index">02 / COMPARISON</span>
    <h3><a href="mobilenetv3-workflow/">MobileNetV3</a></h3>
    <p>Use a controlled pose-and-backbone descriptor path with shared temporal modeling and evaluation.</p>
    <a class="bsx-text-link" href="mobilenetv3-workflow/">Open workflow →</a>
  </article>
  <article>
    <span class="bsx-path-grid__index">03 / TOP-DOWN</span>
    <h3><a href="deeplabcut-hrnet-workflow/">DeepLabCut-HRNet</a></h3>
    <p>Stage SuperAnimal pose, HRNet feature caches, classifier training, and held-out evaluation.</p>
    <a class="bsx-text-link" href="deeplabcut-hrnet-workflow/">Open workflow →</a>
  </article>
</div>

## Research workflow

The map follows the primary path from research recordings to evidence you can
inspect. Open it separately for pan, zoom, and a larger canvas.

<div class="bsx-map-heading">
  <span>BEHAVIORSCOPE-X / END-TO-END</span>
  <a href="workflow-map.html">Open interactive map ↗</a>
</div>
<div class="workflow-map-scroll">
  <iframe
    class="workflow-map-frame"
    src="workflow-map.html?embed=1"
    title="Interactive BehaviorScope-X research workflow"
    loading="eager"
  ></iframe>
</div>

<p><small>Workflow map generated with <a href="https://github.com/tt-a1i/archify">Archify</a>, an MIT-licensed interactive diagram renderer. Special thanks to <strong>Shaun Andrade</strong> for introducing the project to Archify and for assisting with pilot studies and architecture testing during the early builds of BehaviorScope. See <a href="acknowledgments/">Acknowledgments</a>.</small></p>

## Keep the research decisions visible

<div class="bsx-principles">
  <article><b>CURATE ONCE</b><p>Annotate bouts, record review state, and keep train, validation, held-out test, and excluded videos explicit.</p></article>
  <article><b>TRAIN DELIBERATELY</b><p>Select a supported pose path, build compatible caches, and evaluate the temporal classifier on held-out data.</p></article>
  <article><b>REVIEW THE EVIDENCE</b><p>Inspect prediction tables, ethograms, bout summaries, metrics, and optional annotated videos before interpretation.</p></article>
</div>

## Special thanks

BehaviorScope-X gratefully acknowledges **Shaun Andrade**, an undergraduate
Computer Science major at the University of Maryland, Baltimore County (UMBC),
for assisting with pilot studies and architecture testing during the early builds
of the BehaviorScope project. Shaun also introduced the project to
[Archify](https://github.com/tt-a1i/archify), the workflow-diagram repository
used to generate this site's interactive research map.

[Read the full acknowledgments](acknowledgments.md){ .md-button }

<div class="bsx-final-cta" markdown>
<div markdown>
### Start with the guided desktop workflow

Install the application, annotate a first video, and follow one documented pose
path through model training and output review.
</div>
<div class="bsx-hero__actions" markdown>
[Read the GUI guide](gui-overview.md){ .md-button .md-button--primary }
[Browse all documentation](documentation.md){ .md-button }
</div>
</div>
