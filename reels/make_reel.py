"""Instagram Reels (1080x1920) 用のショート動画を生成する。

使い方:
    python3 reels/make_reel.py [script.json] [出力ディレクトリ]

script.json のテキストを書き換えれば別の投稿もすぐ作れる。
"""
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1080, 1920
FONT_PATH = "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf"
HERE = Path(__file__).resolve().parent

# 夜明け → 朝 へ変化する空 (上 / 中 / 下)
SKY_START = [(22, 30, 72), (112, 96, 168), (246, 178, 150)]
SKY_END = [(58, 108, 182), (150, 190, 232), (253, 226, 200)]


def lerp(a, b, t):
    return a + (b - a) * t


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def sky(t):
    ys = np.linspace(0, 1, H)[:, None]
    cols = [np.array([lerp(s, e, t) for s, e in zip(cs, ce)]) for cs, ce in zip(SKY_START, SKY_END)]
    top, mid, bot = cols
    upper = top + (mid - top) * np.clip(ys / 0.55, 0, 1)
    lower = mid + (bot - mid) * np.clip((ys - 0.55) / 0.45, 0, 1)
    grad = np.where(ys < 0.55, upper, lower)
    return np.broadcast_to(grad[:, None, :], (H, W, 3)).astype(np.float32)


def sun_glow(t):
    """画面下からゆっくり昇る柔らかい光。"""
    cy = lerp(H * 1.05, H * 0.82, ease(t))
    yy, xx = np.ogrid[:H, :W]
    d = np.sqrt((xx - W / 2) ** 2 + (yy - cy) ** 2)
    g = np.exp(-(d / 480) ** 2) * (0.2 + 0.3 * ease(t))
    return g[:, :, None].astype(np.float32)


def make_cloud(rng, w, h):
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    for _ in range(26):
        cx = rng.uniform(w * 0.3, w * 0.7)
        cy = rng.uniform(h * 0.45, h * 0.55)
        r = rng.uniform(h * 0.1, h * 0.18)
        d.ellipse([cx - r * 1.4, cy - r, cx + r * 1.4, cy + r], fill=255)
    img = img.filter(ImageFilter.GaussianBlur(h * 0.09))
    return np.asarray(img, np.float32) / 255.0


def text_layer(lines):
    """シーンの各行を、グロー付きの RGBA 画像として個別に描画する。"""
    out = []
    total_h = sum(int(l["size"] * 1.7) for l in lines)
    y = 900 - total_h // 2
    for l in lines:
        lh = int(l["size"] * 1.7)
        if l["text"]:
            font = ImageFont.truetype(FONT_PATH, l["size"])
            tracking = l.get("tracking", 2)
            chars = list(l["text"])
            widths = [font.getlength(c) for c in chars]
            tw = sum(widths) + tracking * (len(chars) - 1)
            x = (W - tw) / 2
            mask = Image.new("L", (W, lh), 0)
            d = ImageDraw.Draw(mask)
            for c, cw in zip(chars, widths):
                d.text((x, lh / 2), c, font=font, fill=255, anchor="lm")
                x += cw + tracking
            shadow = mask.filter(ImageFilter.GaussianBlur(14))
            glow = mask.filter(ImageFilter.GaussianBlur(6))
            out.append({
                "y": y,
                "text": np.asarray(mask, np.float32) / 255.0,
                "shadow": np.asarray(shadow, np.float32) / 255.0,
                "glow": np.asarray(glow, np.float32) / 255.0,
            })
        y += lh
    return out


def blend(frame, y, a, color, alpha):
    h = a.shape[0]
    y0, y1 = max(y, 0), min(y + h, H)
    if y1 <= y0:
        return
    a = a[y0 - y:y1 - y, :, None] * alpha
    region = frame[y0:y1]
    region += (np.array(color, np.float32) - region) * a


def main():
    script_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "script.json"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(script_path.read_text(encoding="utf-8"))
    fps, dur = cfg["fps"], cfg["duration"]
    n = int(fps * dur)
    rng = random.Random(7)

    clouds = []
    for i in range(5):
        w, h = rng.randint(700, 1100), rng.randint(260, 380)
        clouds.append({
            "a": make_cloud(rng, w, h),
            "x": rng.uniform(-w, W),
            "y": int(rng.uniform(120, 1500)),
            "v": rng.uniform(12, 30) * (1 if i % 2 else -1),
            "o": rng.uniform(0.18, 0.32),
        })
    stars = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(1.5, 4), rng.uniform(0, 6.28),
              rng.uniform(1.5, 3.5), rng.uniform(8, 25)) for _ in range(70)]
    scenes = [dict(s, layers=text_layer(s["lines"])) for s in cfg["scenes"]]

    hfont = ImageFont.truetype(FONT_PATH, 36)
    handle = Image.new("L", (W, 80), 0)
    ImageDraw.Draw(handle).text((W / 2, 40), cfg["handle"], font=hfont, fill=255, anchor="mm")
    handle = np.asarray(handle, np.float32) / 255.0

    silent = out_dir / "soraspi_reel_silent.mp4"
    enc = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(silent),
    ], stdin=subprocess.PIPE)

    for f in range(n):
        t = f / fps
        p = t / dur
        frame = sky(p).copy()
        frame += (np.array([255, 240, 220], np.float32) - frame) * sun_glow(p)

        for c in clouds:
            ch, cw = c["a"].shape
            x = int((c["x"] + c["v"] * t) % (W + cw)) - cw
            x0, x1 = max(x, 0), min(x + cw, W)
            y0, y1 = c["y"], min(c["y"] + ch, H)
            if x1 > x0:
                a = c["a"][: y1 - y0, x0 - x:x1 - x, None] * c["o"]
                frame[y0:y1, x0:x1] += (255 - frame[y0:y1, x0:x1]) * a

        img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(img, "RGBA")
        for sx, sy, r, ph, sp, rise in stars:
            a = (0.5 + 0.5 * math.sin(ph + t * sp)) * (1 - 0.6 * p)
            yy = (sy - rise * t) % H
            d.ellipse([sx - r, yy - r, sx + r, yy + r], fill=(255, 255, 240, int(200 * a)))
        frame = np.asarray(img, np.float32).copy()

        for s in scenes:
            if not (s["start"] <= t <= s["end"]):
                continue
            out_a = 1 - ease((t - (s["end"] - 0.5)) / 0.5)
            for i, L in enumerate(s["layers"]):
                a = ease((t - s["start"] - i * 0.45) / 0.8) * out_a
                if a <= 0:
                    continue
                y = L["y"] + int((1 - a) * 24)
                blend(frame, y, L["shadow"], (30, 34, 80), 0.45 * a)
                blend(frame, y, L["glow"], (255, 250, 235), 0.55 * a)
                blend(frame, y, L["text"], (255, 255, 255), a)

        blend(frame, 1380, handle, (255, 255, 255), 0.85 * ease(t / 1.0) * (1 - ease((t - dur + 0.4) / 0.4)))

        # 冒頭と末尾のフェード
        fade = ease(t / 0.4) * (1 - ease((t - (dur - 0.3)) / 0.3))
        frame *= fade
        enc.stdin.write(np.clip(frame, 0, 255).astype(np.uint8).tobytes())

    enc.stdin.close()
    enc.wait()

    # やさしいアンビエント音 (Cmaj9 のパッド + ゆるい揺らぎ)
    freqs = [130.81, 196.0, 261.63, 329.63, 392.0, 587.33]
    expr = "+".join(f"0.11*sin(2*PI*{fr}*t)*(0.7+0.3*sin(2*PI*{0.07 + i * 0.05:.2f}*t))"
                    for i, fr in enumerate(freqs))
    final = out_dir / "soraspi_reel.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(silent),
        "-f", "lavfi", "-i", f"aevalsrc='{expr}':s=48000:d={dur}",
        "-filter_complex",
        f"[1:a]lowpass=f=1800,aecho=0.8:0.7:180|420:0.35|0.25,"
        f"afade=t=in:d=1.5,afade=t=out:st={dur - 2}:d=2,volume=0.9[a]",
        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", str(final),
    ], check=True)
    print(final)
    print(silent)


if __name__ == "__main__":
    main()
