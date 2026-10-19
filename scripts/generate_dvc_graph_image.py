"""Render the verified DVC stage graph as a compact poster-ready PNG."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "poster_screenshots" / "06_dvc_graph.png"


def font(size: int):
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def center(draw: ImageDraw.ImageDraw, box, text: str, text_font) -> None:
    left, top, right, bottom = box
    bounds = draw.multiline_textbbox((0, 0), text, font=text_font, align="center")
    x = (left + right - (bounds[2] - bounds[0])) / 2
    y = (top + bottom - (bounds[3] - bounds[1])) / 2
    draw.multiline_text((x, y), text, font=text_font, fill="#17324d", align="center")


def main() -> None:
    image = Image.new("RGB", (1400, 420), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(28)
    box_font = font(24)
    small_font = font(19)
    draw.text((40, 22), "DVC reproducible pipeline", font=title_font, fill="#123d68")

    boxes = [
        (40, 150, 310, 285, "Dataset\nDVC tracked"),
        (390, 150, 650, 285, "prepare\ntrain/test split"),
        (730, 150, 990, 285, "train\nHyperopt + MLflow"),
        (1070, 150, 1360, 285, "evaluate\nmetrics.json"),
    ]
    for left, top, right, bottom, label in boxes:
        draw.rounded_rectangle((left, top, right, bottom), radius=14, fill="#e3f1fb", outline="#1d5f91", width=3)
        center(draw, (left, top, right, bottom), label, box_font)

    for first, second in zip(boxes, boxes[1:]):
        x1 = first[2] + 12
        x2 = second[0] - 12
        y = (first[1] + first[3]) // 2
        draw.line((x1, y, x2, y), fill="#4e6c84", width=4)
        draw.polygon(((x2, y), (x2 - 12, y - 8), (x2 - 12, y + 8)), fill="#4e6c84")

    draw.text((40, 335), "Verified with dvc fetch/push; cache and S3 remote are in sync.", font=small_font, fill="#405466")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
