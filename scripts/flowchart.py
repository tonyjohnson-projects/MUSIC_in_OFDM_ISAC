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

z = Digraph(format="png")
z.attr(
    rankdir="LR",
    splines="ortho",
    nodesep="0.10",
    ranksep="0.24",
    dpi="220",
)

z.attr(
    "node",
    shape="box",
    style="rounded",
    fontsize="14",
    margin="0.08,0.05",
    width="1.25",
    height="0.0",
)

z.node("I", "Known RE cube\nsymbol-divided\nzero-filled")
z.node("F1", "Embed span\n+ support gain")
z.node("F2", "Windowed\naz / r / d FFT")
z.node("F3", "Local maxima\nnoise threshold")
z.node("F4", "Backfill\ncandidate pool")
z.node("F5", "MF refine\ncoord descent")
z.node("F6", "Final NMS\n2 estimates")
z.node("M1", "Spatial cov\n+ model order")
z.node("M2", "FBSS\nsubarrays")
z.node("M3", "Az MUSIC\nnoise subspace")
z.node("M4", "Beamform\neach az peak")
z.node("M5", "Range / Doppler\ncontig support")
z.node("M6", "MF refine\n+ NMS")
z.node("O", "Target\nestimates")

z.edges([
    ("I","F1"),
    ("F1","F2"),
    ("F2","F3"),
    ("F3","F4"),
    ("F4","F5"),
    ("F5","F6"),
    ("F6","O"),
    ("I","M1"),
    ("M1","M2"),
    ("M2","M3"),
    ("M3","M4"),
    ("M4","M5"),
    ("M5","M6"),
    ("M6","O"),
])

z.render("flowchart_zoom_169_png", cleanup=True)
