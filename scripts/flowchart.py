from graphviz import Digraph

g = Digraph(format="png")
g.attr(
    rankdir="LR",
    splines="ortho",
    nodesep="0.12",
    ranksep="0.22",
    dpi="220",
)

g.attr(
    "node",
    shape="box",
    style="rounded",
    fontsize="16",
    margin="0.08,0.05",
    width="1.25",
    height="0.0",
)

g.node("A", "Setup\nanchor / scene\nburst / sweep")
g.node("B", "Synthesis\n2 targets + nuisance\n+ noisy OFDM cube")
g.node("C", "Masked obs.\ngrid masking\n+ symbols")
g.node("D", "Recovery\nusable sensing cube")
g.node("E", "FFT\nmasked front-end\npeaks + refine")
g.node("F", "MUSIC\naz -> r -> d\nFBSS + refine")
g.node("G", "Scoring\nassign / detect\nresolve / RMSE / runtime")
g.node("H", "Aggregate\ntrials + sweeps\nFFT vs MUSIC")

g.edges([
    ("A","B"),
    ("B","C"),
    ("C","D"),
    ("D","E"),
    ("D","F"),
    ("E","G"),
    ("F","G"),
    ("G","H"),
])

g.render("flowchart_169_png", cleanup=True)