"""Generate SYNTHETIC prescription images for the Gemma vision demo.

Usage:  python samples/make_prescriptions.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "prescriptions"

# (file, clinic, date, patient, complaint, medicines)
RX = [
    ("rx_fever.png", "Sri Murugan Clinic (fictional)", "12/09/2026", "(synthetic)", "", [
        "1. Tab. Paracetamol 650 mg   1-0-1   x 3 days",
        "2. Tab. Cetirizine 10 mg     0-0-1   x 5 days",
        "3. Syp. Ambroxol 5 ml        1-1-1   x 5 days",
    ]),
    ("rx_bp_sugar.png", "Kaveri Health Centre (fictional)", "02/08/2026", "(synthetic)", "", [
        "1. Tab. Metformin 500 mg     1-0-1   continue",
        "2. Tab. Amlodipine 5 mg      1-0-0   continue",
        "3. Tab. Atorvastatin 10 mg   0-0-1   continue",
    ]),
    # Demo prescription in the developer's own name (with consent) for the upload demo.
    ("rx_vidhiya_cold.png", "Thendral Family Clinic (fictional)", "01/10/2026", "Vidhiya, 22 F", "C/o cold, running nose x 2 days", [
        "1. Tab. Cetirizine 10 mg      0-0-1   x 5 days",
        "2. Tab. Paracetamol 500 mg    1-0-1   SOS, x 3 days",
        "3. Tab. Vitamin C 500 mg      1-0-0   x 5 days",
        "4. Steam inhalation           twice daily",
    ]),
]


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# Handwritten-style samples: printed pad + blue-ink writing (Windows "Ink Free" font).
HANDWRITTEN = [
    ("rx_vidhiya_cold_handwritten.jpg", "Thendral Family Clinic (fictional)", [
        ("Name: Vidhiya        Age/Sex: 22/F", 0),
        ("Date: 1/10/26", 0),
        ("C/o: Cold, running nose - 2 days", 0),
        ("Rx", 1),
        ("1) T. Cetirizine 10mg      0-0-1  x 5d", 0),
        ("2) T. Paracetamol 500mg   1-0-1  SOS x 3d", 0),
        ("3) T. Vit C 500mg            1-0-0  x 5d", 0),
        ("4) Steam inhalation  BD", 0),
        ("Review if fever / not better", 0),
    ]),
]


def _hand_font(size: int) -> ImageFont.ImageFont:
    for name in ("Inkfree.ttf", "segoepr.ttf", "LHANDW.TTF", "comic.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return _font(size)


def _write_line(img: Image.Image, text: str, x: int, y: int, size: int, rng: "random.Random") -> None:
    """Draw one handwritten line with a slight tilt and wobbly baseline in blue ink."""
    font = _hand_font(size)
    layer = Image.new("RGBA", (img.width, size * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx = 0
    for word in text.split(" "):
        if not word:
            cx += size // 3
            continue
        ink = (20 + rng.randint(0, 25), 40 + rng.randint(0, 25), 140 + rng.randint(-15, 25), 235)
        d.text((cx, size // 3 + rng.randint(-3, 3)), word, font=font, fill=ink)
        cx += int(d.textlength(word + " ", font=font)) + rng.randint(-2, 4)
    layer = layer.rotate(rng.uniform(-1.4, 1.0), resample=Image.BICUBIC, expand=False)
    img.paste(layer, (x + rng.randint(-6, 6), y), layer)


def make_handwritten() -> None:
    import random

    from PIL import ImageFilter

    for filename, clinic, lines in HANDWRITTEN:
        rng = random.Random(filename)
        W, H = 900, 1180
        img = Image.new("RGB", (W, H), (250, 248, 240))
        # paper grain
        px = img.load()
        for _ in range(60000):
            x, y = rng.randrange(W), rng.randrange(H)
            g = rng.randint(-10, 4)
            r, gg, b = px[x, y]
            px[x, y] = (r + g, gg + g, b + g)
        d = ImageDraw.Draw(img)
        # printed letterhead
        d.text((40, 30), clinic, font=_font(36), fill=(27, 106, 111))
        d.text((40, 78), "Dr. A. Example, MBBS  ·  General Physician  ·  Reg. No. SAMPLE-0000", font=_font(18), fill=(70, 70, 70))
        d.text((40, 104), "12, Example Street, Madurai (fictional)  ·  Timings 9am-1pm, 5pm-8pm", font=_font(16), fill=(90, 90, 90))
        d.line([(40, 135), (W - 40, 135)], fill=(27, 106, 111), width=3)
        # faint ruled lines
        for ly in range(200, H - 140, 62):
            d.line([(40, ly), (W - 40, ly)], fill=(215, 222, 232), width=1)
        y = 150
        for text, big in lines:
            _write_line(img, text, 60 if not big else 50, y, 52 if big else 36, rng)
            y += 80 if big else 62
        # signature: cursive scrawl + underline flick
        sx, sy = 590, y + 40
        try:
            sig_font = ImageFont.truetype("segoesc.ttf", 40)
        except OSError:
            sig_font = _hand_font(40)
        sig = Image.new("RGBA", (300, 110), (0, 0, 0, 0))
        sd = ImageDraw.Draw(sig)
        sd.text((5, 10), "AExample", font=sig_font, fill=(25, 45, 150, 240))
        sd.line([(10, 78), (120, 72), (250, 64), (285, 52)], fill=(25, 45, 150, 230), width=3, joint="curve")
        sig = sig.rotate(8, resample=Image.BICUBIC, expand=True)
        img.paste(sig, (sx, sy - 20), sig)
        d.text((610, sy + 95), "Dr. A. Example", font=_font(18), fill=(70, 70, 70))
        d.text((40, H - 40), "SYNTHETIC SAMPLE - NOT A REAL PRESCRIPTION", font=_font(16), fill=(180, 60, 60))
        # look like a phone photo: slight rotation, soft blur, warm background
        photo = img.filter(ImageFilter.GaussianBlur(0.6)).rotate(rng.uniform(-2.5, 2.5), resample=Image.BICUBIC, expand=True, fillcolor=(120, 105, 90))
        photo.convert("RGB").save(OUT / filename, quality=88)
        print("wrote", OUT / filename)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    make_handwritten()
    for filename, clinic, date, patient, complaint, lines in RX:
        img = Image.new("RGB", (900, 620), (252, 252, 247))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 900, 90], fill=(27, 106, 111))
        d.text((30, 22), clinic, font=_font(34), fill="white")
        d.text((30, 110), f"Date: {date}", font=_font(24), fill=(40, 40, 40))
        d.text((520, 110), f"Patient: {patient}", font=_font(24), fill=(40, 40, 40))
        if complaint:
            d.text((30, 150), complaint, font=_font(22), fill=(70, 70, 70))
        d.text((30, 185), "Rx", font=_font(48), fill=(27, 106, 111))
        for i, line in enumerate(lines):
            d.text((60, 250 + i * 60), line, font=_font(28), fill=(20, 20, 60))
        d.text((600, 540), "Dr. A. Example, MBBS", font=_font(22), fill=(60, 60, 60))
        d.text((30, 580), "SYNTHETIC SAMPLE - NOT A REAL PRESCRIPTION", font=_font(16), fill=(180, 60, 60))
        img.save(OUT / filename)
        print("wrote", OUT / filename)


if __name__ == "__main__":
    main()
