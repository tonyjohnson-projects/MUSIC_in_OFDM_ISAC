import fs from "node:fs/promises";
import path from "node:path";

const { FileBlob, Presentation, PresentationFile } = await import("@oai/artifact-tool");

const [, , payloadPathArg, outputPptxArg, previewDirArg] = process.argv;
if (!payloadPathArg || !outputPptxArg || !previewDirArg) {
  throw new Error("Usage: node builder.mjs <payload.json> <output.pptx> <preview-dir>");
}

const payloadPath = path.resolve(payloadPathArg);
const outputPptxPath = path.resolve(outputPptxArg);
const previewDir = path.resolve(previewDirArg);

const WIDTH = 1280;
const HEIGHT = 720;

const COLOR = {
  bg: "#F6F3EC",
  panel: "#FFFDFC",
  text: "#10233F",
  muted: "#5F6B7A",
  accent: "#2F6B9A",
  accentSoft: "#DCE7F1",
  gold: "#D9953B",
  goldSoft: "#F4E7D4",
  border: "#D7D0C7",
  green: "#2FA66A",
  white: "#FFFFFF",
};

const FONT = {
  title: "Aptos Display",
  body: "Aptos",
  mono: "Menlo",
  math: "Cambria Math",
};

await fs.mkdir(previewDir, { recursive: true });
const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
const sourceById = Object.fromEntries(payload.sources.map((source) => [source.id, source]));

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function addShape(slide, { geometry = "rect", position, fill = COLOR.panel, line = { style: "solid", fill: COLOR.border, width: 1.2 } }) {
  return slide.shapes.add({ geometry, position, fill, line });
}

function addTextBox(
  slide,
  {
    left,
    top,
    width,
    height,
    text,
    fontSize = 26,
    bold = false,
    color = COLOR.text,
    typeface = FONT.body,
    fill = null,
    line = null,
    alignment = "left",
    verticalAlignment = "top",
    inset = 14,
    autoFit = "shrinkText",
  },
) {
  const shape = slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill: fill ?? "#FFFFFF00",
    line: line ?? { style: "solid", fill: "#FFFFFF00", width: 0 },
  });
  shape.text = text;
  shape.text.fontSize = fontSize;
  shape.text.bold = bold;
  shape.text.color = color;
  shape.text.typeface = typeface;
  shape.text.alignment = alignment;
  shape.text.verticalAlignment = verticalAlignment;
  shape.text.insets = { left: inset, right: inset, top: inset, bottom: inset };
  shape.text.autoFit = autoFit;
  return shape;
}

async function addImage(slide, imagePath, position, alt, fit = "contain") {
  const image = slide.images.add({
    blob: await readImageBlob(imagePath),
    fit,
    alt,
  });
  image.position = position;
  return image;
}

function addSlideChrome(slide, slideSpec, slideIndex, slideCount) {
  slide.background.fill = COLOR.bg;
  addShape(slide, {
    geometry: "rect",
    position: { left: 0, top: 0, width: WIDTH, height: 16 },
    fill: COLOR.accent,
    line: { style: "solid", fill: COLOR.accent, width: 0 },
  });
  addTextBox(slide, {
    left: 72,
    top: 26,
    width: 220,
    height: 24,
    text: slideSpec.appendix ? "Appendix" : slideSpec.section,
    fontSize: 13,
    bold: true,
    color: COLOR.accent,
    typeface: FONT.body,
    inset: 0,
  });
  addTextBox(slide, {
    left: 72,
    top: 52,
    width: 1080,
    height: 58,
    text: slideSpec.title,
    fontSize: slideSpec.layout === "title" ? 34 : 28,
    bold: true,
    color: COLOR.text,
    typeface: FONT.title,
    inset: 0,
  });
  if (slideSpec.layout !== "title") {
    addShape(slide, {
      geometry: "roundRect",
      position: { left: 72, top: 112, width: 1136, height: 46 },
      fill: COLOR.goldSoft,
      line: { style: "solid", fill: COLOR.goldSoft, width: 0 },
    });
    addTextBox(slide, {
      left: 84,
      top: 118,
      width: 1112,
      height: 34,
      text: slideSpec.takeaway,
      fontSize: 16,
      color: COLOR.text,
      typeface: FONT.body,
      inset: 0,
    });
  }
  if (slideSpec.layout !== "title") {
    addTextBox(slide, {
      left: 1160,
      top: 678,
      width: 70,
      height: 18,
      text: `${slideIndex + 1}/${slideCount}`,
      fontSize: 11,
      color: COLOR.muted,
      typeface: FONT.body,
      alignment: "right",
      inset: 0,
    });
  }
}

function renderBulletPanel(slide, bullets, position, title = null) {
  addShape(slide, {
    geometry: "roundRect",
    position,
    fill: COLOR.panel,
    line: { style: "solid", fill: COLOR.border, width: 1.1 },
  });
  let cursorTop = position.top + 18;
  if (title) {
    addTextBox(slide, {
      left: position.left + 6,
      top: cursorTop,
      width: position.width - 12,
      height: 30,
      text: title,
      fontSize: 16,
      bold: true,
      color: COLOR.accent,
      typeface: FONT.body,
      inset: 12,
    });
    cursorTop += 32;
  }
  for (const bullet of bullets) {
    addTextBox(slide, {
      left: position.left + 14,
      top: cursorTop,
      width: position.width - 28,
      height: 78,
      text: `• ${bullet}`,
      fontSize: 18,
      color: COLOR.text,
      typeface: FONT.body,
      inset: 0,
    });
    cursorTop += 86;
  }
}

function renderCompactBulletPanel(slide, bullets, position, title = null, fontSize = 16) {
  addShape(slide, {
    geometry: "roundRect",
    position,
    fill: COLOR.panel,
    line: { style: "solid", fill: COLOR.border, width: 1.1 },
  });
  const lines = bullets.map((bullet) => `• ${bullet}`).join("\n");
  let titleOffset = 0;
  if (title) {
    addTextBox(slide, {
      left: position.left + 8,
      top: position.top + 12,
      width: position.width - 16,
      height: 22,
      text: title,
      fontSize: 12,
      bold: true,
      color: COLOR.accent,
      inset: 8,
    });
    titleOffset = 24;
  }
  addTextBox(slide, {
    left: position.left + 14,
    top: position.top + 14 + titleOffset,
    width: position.width - 28,
    height: position.height - 20 - titleOffset,
    text: lines,
    fontSize,
    color: COLOR.text,
    inset: 0,
  });
}

function renderKeyValueRows(slide, rows, frame) {
  addShape(slide, {
    geometry: "roundRect",
    position: frame,
    fill: COLOR.panel,
    line: { style: "solid", fill: COLOR.border, width: 1.1 },
  });
  const rowHeight = Math.floor((frame.height - 18) / rows.length);
  rows.forEach((row, index) => {
    const top = frame.top + 9 + index * rowHeight;
    addShape(slide, {
      geometry: "rect",
      position: { left: frame.left + 12, top, width: frame.width - 24, height: rowHeight - 8 },
      fill: index % 2 === 0 ? "#FFFFFF" : "#FAF7F2",
      line: { style: "solid", fill: "#FFFFFF00", width: 0 },
    });
    addTextBox(slide, {
      left: frame.left + 28,
      top: top + 4,
      width: Math.floor(frame.width * 0.45),
      height: rowHeight - 12,
      text: row[0],
      fontSize: 16,
      bold: true,
      color: COLOR.text,
      inset: 0,
    });
    addTextBox(slide, {
      left: frame.left + Math.floor(frame.width * 0.47),
      top: top + 4,
      width: Math.floor(frame.width * 0.45),
      height: rowHeight - 12,
      text: row[1],
      fontSize: 16,
      color: COLOR.text,
      inset: 0,
    });
  });
}

function renderSimpleTable(slide, headers, rows, frame) {
  addShape(slide, {
    geometry: "roundRect",
    position: frame,
    fill: COLOR.panel,
    line: { style: "solid", fill: COLOR.border, width: 1.1 },
  });
  const columnCount = headers.length;
  const columnWidth = (frame.width - 24) / columnCount;
  const headerTop = frame.top + 12;
  headers.forEach((header, index) => {
    addShape(slide, {
      geometry: "rect",
      position: { left: frame.left + 12 + index * columnWidth, top: headerTop, width: columnWidth - 2, height: 42 },
      fill: COLOR.accentSoft,
      line: { style: "solid", fill: "#FFFFFF00", width: 0 },
    });
    addTextBox(slide, {
      left: frame.left + 18 + index * columnWidth,
      top: headerTop + 6,
      width: columnWidth - 14,
      height: 28,
      text: header,
      fontSize: 14,
      bold: true,
      color: COLOR.text,
      inset: 0,
      alignment: "center",
    });
  });
  const rowHeight = Math.floor((frame.height - 66) / rows.length);
  rows.forEach((row, rowIndex) => {
    const top = frame.top + 58 + rowIndex * rowHeight;
    row.forEach((cell, colIndex) => {
      addShape(slide, {
        geometry: "rect",
        position: {
          left: frame.left + 12 + colIndex * columnWidth,
          top,
          width: columnWidth - 2,
          height: rowHeight - 4,
        },
        fill: rowIndex % 2 === 0 ? "#FFFFFF" : "#FAF7F2",
        line: { style: "solid", fill: "#FFFFFF00", width: 0 },
      });
      addTextBox(slide, {
        left: frame.left + 18 + colIndex * columnWidth,
        top: top + 6,
        width: columnWidth - 14,
        height: rowHeight - 16,
        text: cell,
        fontSize: 14,
        bold: colIndex === 0,
        color: COLOR.text,
        inset: 0,
      });
    });
  });
}

function renderCards(slide, cards, frame) {
  const gap = 18;
  const cardWidth = (frame.width - gap * (cards.length - 1)) / cards.length;
  cards.forEach((card, index) => {
    const left = frame.left + index * (cardWidth + gap);
    addShape(slide, {
      geometry: "roundRect",
      position: { left, top: frame.top, width: cardWidth, height: frame.height },
      fill: index === 1 ? COLOR.accentSoft : COLOR.panel,
      line: { style: "solid", fill: COLOR.border, width: 1.1 },
    });
    addTextBox(slide, {
      left: left + 10,
      top: frame.top + 12,
      width: cardWidth - 20,
      height: 38,
      text: card.title,
      fontSize: 18,
      bold: true,
      color: COLOR.accent,
      inset: 0,
    });
    addTextBox(slide, {
      left: left + 10,
      top: frame.top + 56,
      width: cardWidth - 20,
      height: frame.height - 70,
      text: card.body,
      fontSize: 17,
      color: COLOR.text,
      inset: 0,
    });
  });
}

function renderProcess(slide, steps, frame) {
  const columns = 3;
  const rows = 2;
  const gapX = 22;
  const gapY = 26;
  const cardWidth = (frame.width - gapX * (columns - 1)) / columns;
  const cardHeight = (frame.height - gapY * (rows - 1)) / rows;
  steps.forEach((step, index) => {
    const col = index % columns;
    const row = Math.floor(index / columns);
    const left = frame.left + col * (cardWidth + gapX);
    const top = frame.top + row * (cardHeight + gapY);
    addShape(slide, {
      geometry: "roundRect",
      position: { left, top, width: cardWidth, height: cardHeight },
      fill: COLOR.panel,
      line: { style: "solid", fill: COLOR.border, width: 1.1 },
    });
    addTextBox(slide, {
      left: left + 12,
      top: top + 10,
      width: 34,
      height: 34,
      text: `${index + 1}`,
      fontSize: 16,
      bold: true,
      color: COLOR.white,
      fill: COLOR.accent,
      line: { style: "solid", fill: COLOR.accent, width: 0 },
      alignment: "center",
      verticalAlignment: "middle",
      inset: 0,
    });
    addTextBox(slide, {
      left: left + 54,
      top: top + 12,
      width: cardWidth - 66,
      height: cardHeight - 20,
      text: step,
      fontSize: 18,
      color: COLOR.text,
      inset: 0,
    });
    if (col < columns - 1 && index + 1 < steps.length) {
      addShape(slide, {
        geometry: "rightArrow",
        position: { left: left + cardWidth + 4, top: top + cardHeight / 2 - 9, width: 14, height: 18 },
        fill: COLOR.gold,
        line: { style: "solid", fill: COLOR.gold, width: 0 },
      });
    }
  });
}

function addNotes(slide, slideSpec) {
  const sources = slideSpec.sources
    .map((sourceId) => sourceById[sourceId])
    .filter(Boolean)
    .map((source) => `${source.id}: ${source.citation}${source.url ? ` (${source.url})` : source.note ? ` (${source.note})` : ""}`);
  const noteLines = [
    `Takeaway: ${slideSpec.takeaway}`,
    "",
    "Presenter notes:",
    ...slideSpec.speaker_notes.map((note) => `- ${note}`),
    "",
    "Sources:",
    ...sources.map((source) => `- ${source}`),
  ];
  slide.speakerNotes.setText(noteLines.join("\n"));
}

async function renderSlide(slide, slideSpec, slideIndex, slideCount, figureManifest) {
  addSlideChrome(slide, slideSpec, slideIndex, slideCount);

  if (slideSpec.layout === "title") {
    addShape(slide, {
      geometry: "roundRect",
      position: { left: 72, top: 180, width: 760, height: 240 },
      fill: COLOR.panel,
      line: { style: "solid", fill: COLOR.border, width: 1.1 },
    });
    addTextBox(slide, {
      left: 96,
      top: 218,
      width: 712,
      height: 120,
      text: slideSpec.title,
      fontSize: 34,
      bold: true,
      typeface: FONT.title,
      inset: 0,
    });
    addTextBox(slide, {
      left: 96,
      top: 350,
      width: 700,
      height: 62,
      text: slideSpec.takeaway,
      fontSize: 18,
      color: COLOR.muted,
      inset: 0,
    });
    addShape(slide, {
      geometry: "roundRect",
      position: { left: 886, top: 180, width: 322, height: 240 },
      fill: COLOR.accentSoft,
      line: { style: "solid", fill: COLOR.accentSoft, width: 0 },
    });
    addTextBox(slide, {
      left: 914,
      top: 214,
      width: 266,
      height: 160,
      text: slideSpec.visible_copy.join("\n\n"),
      fontSize: 19,
      color: COLOR.text,
      inset: 0,
    });
    addTextBox(slide, {
      left: 72,
      top: 470,
      width: 480,
      height: 24,
      text: slideSpec.metadata.author ?? "",
      fontSize: 16,
      color: COLOR.text,
      inset: 0,
    });
    addTextBox(slide, {
      left: 72,
      top: 496,
      width: 480,
      height: 24,
      text: slideSpec.metadata.date ?? "",
      fontSize: 16,
      color: COLOR.muted,
      inset: 0,
    });
    addTextBox(slide, {
      left: 72,
      top: 612,
      width: 620,
      height: 26,
      text: "Claim boundary: conditional MUSIC benefit under communications-limited sensing",
      fontSize: 14,
      color: COLOR.accent,
      inset: 0,
    });
    addNotes(slide, slideSpec);
    return;
  }

  switch (slideSpec.layout) {
    case "figure_focus": {
      const figure = figureManifest[slideSpec.visuals[0]];
      if (!figure) {
        throw new Error(`Missing figure manifest entry for ${slideSpec.visuals[0]}`);
      }
      await addImage(
        slide,
        figure.path,
        { left: 72, top: 180, width: 804, height: 470 },
        figure.title,
        "contain",
      );
      renderBulletPanel(slide, slideSpec.visible_copy, { left: 900, top: 180, width: 308, height: 470 }, "Why it matters");
      if (slideSpec.metadata?.figure_caption) {
        addTextBox(slide, {
          left: 72,
          top: 656,
          width: 804,
          height: 24,
          text: slideSpec.metadata.figure_caption,
          fontSize: 12,
          color: COLOR.muted,
          inset: 0,
        });
      }
      break;
    }
    case "equation": {
      addShape(slide, {
        geometry: "roundRect",
        position: { left: 72, top: 190, width: 700, height: 204 },
        fill: COLOR.panel,
        line: { style: "solid", fill: COLOR.border, width: 1.1 },
      });
      if (slideSpec.visuals?.length) {
        const figure = figureManifest[slideSpec.visuals[0]];
        if (!figure) {
          throw new Error(`Missing figure manifest entry for ${slideSpec.visuals[0]}`);
        }
        await addImage(
          slide,
          figure.path,
          { left: 96, top: 214, width: 652, height: 104 },
          figure.title,
          "contain",
        );
      } else {
        addTextBox(slide, {
          left: 96,
          top: 220,
          width: 652,
          height: 92,
          text: slideSpec.metadata.equation ?? "",
          fontSize: slideSpec.appendix ? 26 : 28,
          color: COLOR.text,
          typeface: FONT.math,
          inset: 0,
          autoFit: "shrinkText",
        });
      }
      addTextBox(slide, {
        left: 96,
        top: 314,
        width: 652,
        height: 56,
        text: slideSpec.metadata.equation_note ?? "",
        fontSize: 15,
        color: COLOR.muted,
        inset: 0,
      });
      renderBulletPanel(slide, slideSpec.visible_copy, { left: 812, top: 190, width: 396, height: 340 }, "Interpretation");
      break;
    }
    case "cards": {
      renderCards(slide, slideSpec.metadata.cards ?? [], { left: 72, top: 204, width: 1136, height: 206 });
      renderBulletPanel(slide, slideSpec.visible_copy, { left: 72, top: 438, width: 1136, height: 196 }, "Working interpretation");
      break;
    }
    case "table_figure": {
      renderKeyValueRows(slide, slideSpec.metadata.table_rows ?? [], { left: 72, top: 186, width: 420, height: 370 });
      const figure = figureManifest[slideSpec.visuals[0]];
      await addImage(slide, figure.path, { left: 530, top: 186, width: 678, height: 370 }, figure.title, "contain");
      renderCompactBulletPanel(slide, slideSpec.visible_copy, { left: 72, top: 572, width: 1136, height: 100 }, null, 15);
      break;
    }
    case "table_cards": {
      renderSimpleTable(slide, slideSpec.metadata.table_headers ?? [], slideSpec.metadata.table_rows ?? [], { left: 72, top: 186, width: 1136, height: 250 });
      renderBulletPanel(slide, slideSpec.visible_copy, { left: 72, top: 458, width: 1136, height: 196 }, "Interpretation");
      break;
    }
    case "process": {
      renderProcess(slide, slideSpec.metadata.steps ?? [], { left: 72, top: 192, width: 1136, height: 248 });
      renderCompactBulletPanel(slide, slideSpec.visible_copy, { left: 72, top: 470, width: 1136, height: 186 }, "What was built", 16);
      break;
    }
    case "closing": {
      renderBulletPanel(slide, slideSpec.visible_copy, { left: 72, top: 190, width: 740, height: 420 }, "Final framing");
      addShape(slide, {
        geometry: "roundRect",
        position: { left: 848, top: 190, width: 360, height: 420 },
        fill: COLOR.accentSoft,
        line: { style: "solid", fill: COLOR.accentSoft, width: 0 },
      });
      addTextBox(slide, {
        left: 876,
        top: 222,
        width: 304,
        height: 160,
        text: "Next steps\n\n- masked-covariance order control\n- clutter-aware spatial processing\n- limited-symbol-knowledge sensing\n- broader waveform realism",
        fontSize: 20,
        color: COLOR.text,
        inset: 0,
      });
      addTextBox(slide, {
        left: 876,
        top: 438,
        width: 304,
        height: 90,
        text: "Main close: the result is useful because it says where MUSIC is worth the extra complexity, and where it is not.",
        fontSize: 16,
        color: COLOR.muted,
        inset: 0,
      });
      break;
    }
    case "table_bullets": {
      renderKeyValueRows(slide, slideSpec.metadata.table_rows ?? [], { left: 72, top: 192, width: 460, height: 410 });
      renderBulletPanel(slide, slideSpec.visible_copy, { left: 564, top: 192, width: 644, height: 410 }, "How to read the metrics");
      break;
    }
    case "references": {
      renderBulletPanel(slide, slideSpec.visible_copy, { left: 72, top: 192, width: 1136, height: 180 }, "Primary framing");
      const refIds = slideSpec.sources;
      let top = 406;
      for (const sourceId of refIds) {
        const source = sourceById[sourceId];
        if (!source) continue;
        addTextBox(slide, {
          left: 84,
          top,
          width: 1120,
          height: 42,
          text: `${source.id}: ${source.citation}`,
          fontSize: 13,
          color: COLOR.text,
          inset: 0,
        });
        top += 46;
      }
      break;
    }
    default: {
      renderBulletPanel(slide, slideSpec.visible_copy, { left: 72, top: 188, width: 1136, height: 446 });
      break;
    }
  }

  addNotes(slide, slideSpec);
}

const presentation = Presentation.create({
  slideSize: { width: WIDTH, height: HEIGHT },
});

for (let index = 0; index < payload.slides.length; index += 1) {
  const slideSpec = payload.slides[index];
  const slide = presentation.slides.add();
  await renderSlide(slide, slideSpec, index, payload.slides.length, payload.figures);
}

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(outputPptxPath);

const imported = await PresentationFile.importPptx(await FileBlob.load(outputPptxPath));
if (imported.slides.count !== payload.slides.length) {
  throw new Error(`Imported slide count ${imported.slides.count} did not match expected ${payload.slides.length}`);
}

for (let index = 0; index < presentation.slides.count; index += 1) {
  const slide = presentation.slides.getItem(index);
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  const pngBuffer = Buffer.from(await png.arrayBuffer());
  await fs.writeFile(path.join(previewDir, `slide-${String(index + 1).padStart(2, "0")}.png`), pngBuffer);
}
