"""A poster-style playoff bracket: black and gold, team-colour tiles with logo and seed, empty boxes for the
rounds still to be played. Returns a self-contained HTML string (inline SVG)."""
from __future__ import annotations

from functools import lru_cache
from html import escape

import pandas as pd

from cfp.config import DATA

GOLD, BLACK = "#c9a55a", "#0d0d0d"


@lru_cache(maxsize=1)
def _colors() -> dict[int, dict]:
    p = DATA / "team_colors.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    return {int(r.team_id): {"color": r.color, "abbr": r.abbreviation} for r in d.itertuples()}


def _text_on(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#111111" if (0.299 * r + 0.587 * g + 0.114 * b) > 165 else "#ffffff"


def bracket_poster(seeds: dict[int, dict], bowls: dict[int, str], title: str = "PROJECTED BRACKET",
                   subtitle: str = "", footer: str = "") -> str:
    """seeds: {1..12: {"name": str, "team_id": int}}. bowls: {1..4: quarterfinal bowl name}."""
    col = _colors()
    W, H = 760, 930
    TW, TH, SW = 118, 50, 40            # tile width / height, seed-box width
    x1, x2, x3, x4 = 34, 236, 446, 600  # column left edges
    BW3, BW4 = 132, 132                 # empty-box widths for semifinals and final
    y0s = [160, 340, 520, 700]
    pairs = [(12, 5, 4), (9, 8, 1), (11, 6, 3), (10, 7, 2)]   # (top tile, bottom tile, seed waiting in the quarterfinal)
    o: list[str] = []

    def tile(x: float, y: float, seed: int) -> None:
        t = seeds[seed]
        c = col.get(int(t["team_id"]), {})
        bg = c.get("color") or "#333333"
        fg = _text_on(bg)
        ab = escape(str(c.get("abbr") or t["name"][:4]).upper())
        logo = f"https://a.espncdn.com/i/teamlogos/ncaa/500/{int(t['team_id'])}.png"
        o.append(f'<g><title>{seed}. {escape(t["name"])}</title>'
                 f'<rect x="{x}" y="{y}" width="{SW}" height="{TH}" fill="{BLACK}" stroke="{GOLD}" stroke-width="2"/>'
                 f'<text x="{x + SW / 2}" y="{y + TH / 2 + 9}" text-anchor="middle" fill="#fff" font-size="25" font-weight="800">{seed}</text>'
                 f'<rect x="{x + SW}" y="{y}" width="{TW}" height="{TH}" fill="{bg}" stroke="{GOLD}" stroke-width="2"/>'
                 f'<rect x="{x + SW + 7}" y="{y + 6}" width="42" height="{TH - 12}" rx="6" fill="#ffffff"/>'
                 f'<image href="{logo}" x="{x + SW + 10}" y="{y + 7}" width="36" height="36"/>'
                 f'<text x="{x + SW + 56 + (TW - 56) / 2}" y="{y + TH / 2 + 6}" text-anchor="middle" fill="{fg}" font-size="16" '
                 f'font-weight="800" letter-spacing=".5">{ab}</text></g>')

    def empty(x: float, y: float, w: float) -> None:
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{TH}" fill="#050505" stroke="{GOLD}" stroke-width="2"/>')

    def label(x: float, y: float, text: str, size: int = 12, anchor: str = "middle", fill: str = "#ffffff") -> None:
        o.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" font-size="{size}" font-weight="800" '
                 f'letter-spacing="1.4">{escape(text)}</text>')

    def link(xa: float, ya: float, xb: float, yb: float) -> None:
        xm = (xa + xb) / 2
        o.append(f'<path d="M{xa} {ya} H{xm} V{yb} H{xb}" fill="none" stroke="{GOLD}" stroke-width="2"/>')

    # frame and title
    o.append(f'<rect width="{W}" height="{H}" fill="{BLACK}"/>')
    o.append(f'<rect x="10" y="10" width="{W - 20}" height="{H - 20}" fill="none" stroke="{GOLD}" stroke-width="3"/>')
    o.append(f'<rect x="18" y="18" width="{W - 36}" height="{H - 36}" fill="none" stroke="{GOLD}" stroke-width="1" opacity=".5"/>')
    o.append(f'<text x="{W / 2}" y="76" text-anchor="middle" fill="{GOLD}" font-size="46" font-weight="900" letter-spacing="3">{escape(title)}</text>')
    if subtitle:
        label(W / 2, 104, subtitle.upper(), 13, fill="#d9d9d9")

    label(x1 + (SW + TW) / 2, y0s[0] - 14, "FIRST ROUND", 13)
    label(x2 + (SW + TW) / 2, y0s[0] - 2, "QUARTERFINALS", 13)
    label(x3 + BW3 / 2, y0s[0] + 76, "SEMIFINALS", 13)
    label(x4 + BW4 / 2, y0s[1] + 72, "CHAMPIONSHIP", 13)

    qf_mid = []
    for y0, (a, b, top) in zip(y0s, pairs):
        tile(x1, y0, a); tile(x1, y0 + TH + 2, b)
        qy = y0 + 12
        empty(x2, qy, SW + TW)
        tile(x2, qy + TH + 2, top)
        link(x1 + SW + TW, y0 + TH + 1, x2, qy + TH / 2)
        label(x2 + (SW + TW) / 2, qy + 2 * TH + 22, f"{bowls.get(top, '')} BOWL".strip().upper(), 12)
        qf_mid.append(qy + TH + 1)

    sf_mid = []
    for j in range(2):
        m = (qf_mid[2 * j] + qf_mid[2 * j + 1]) / 2
        empty(x3, m - TH - 1, BW3); empty(x3, m + 1, BW3)
        link(x2 + SW + TW, qf_mid[2 * j], x3, m - TH / 2 - 1)
        link(x2 + SW + TW, qf_mid[2 * j + 1], x3, m + TH / 2 + 1)
        label(x3 + BW3 / 2, m + TH + 22, "SEMIFINAL", 12)
        sf_mid.append(m)
    m = (sf_mid[0] + sf_mid[1]) / 2
    empty(x4, m - TH - 1, BW4); empty(x4, m + 1, BW4)
    link(x3 + BW3, sf_mid[0], x4, m - TH / 2 - 1)
    link(x3 + BW3, sf_mid[1], x4, m + TH / 2 + 1)
    label(x4 + BW4 / 2, m + TH + 22, "JAN 25 · LAS VEGAS", 11)
    if footer:
        label(W / 2, H - 34, footer, 11, fill="#9a9a9a")

    return (f'<div style="display:flex;justify-content:center;background:transparent">'
            f'<svg viewBox="0 0 {W} {H}" style="width:100%;max-width:{W}px;height:auto;font-family:\'Arial Black\',\'Helvetica Neue\',Arial,sans-serif" '
            f'xmlns="http://www.w3.org/2000/svg">{"".join(o)}</svg></div>')
