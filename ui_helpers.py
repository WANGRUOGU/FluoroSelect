# ui_helpers.py
import numpy as np
import streamlit as st
from config import DEFAULT_COLORS

def ensure_colors(R: int) -> np.ndarray:
    """Return an array of R RGB colors in [0,1]."""
    if R <= len(DEFAULT_COLORS):
        return DEFAULT_COLORS[:R]
    hs = np.linspace(0, 1, R, endpoint=False)
    extra = np.stack([
        np.abs(np.sin(2 * np.pi * hs)) * 0.7 + 0.3,
        np.abs(np.sin(2 * np.pi * (hs + 0.33))) * 0.7 + 0.3,
        np.abs(np.sin(2 * np.pi * (hs + 0.66))) * 0.7 + 0.3,
    ], axis=1)
    return extra[:R]


def rgb01_to_plotly(col):
    r, g, b = (int(255 * float(x)) for x in col)
    return f"rgb({r},{g},{b})"


def pair_only_fluor(a: str, b: str) -> str:
    """Convert 'Probe – AF488' vs 'P – AF647' into 'AF488 vs AF647'."""
    fa = a.split(" – ", 1)[1] if " – " in a else a
    fb = b.split(" – ", 1)[1] if " – " in b else b
    return f"{fa} vs {fb}"


def prettify_name(label: str) -> str:
    """
    Map 'Probe – AF405' -> 'AF 405'; leave other names as-is.
    Used for short captions in grids / tables.
    """
    name = label.split(" – ", 1)[1] if " – " in label else label
    up = name.upper()
    if up.startswith("AF") and name[2:].isdigit():
        return f"AF {name[2:]}"
    return name


def html_two_row_table(row0_label, row1_label, row0_vals, row1_vals,
                       color_second_row=False, color_thresh=0.9, fmt2=False):
    """Two-row horizontal table: [fluor list] / [metric list]."""
    def esc(x):
        return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def fmtv(v):
        if fmt2:
            try:
                return f"{float(v):.3f}"
            except Exception:
                return esc(v)
        return esc(v)

    cells0 = "".join(
        f"<td style='padding:6px 10px;border:1px solid #ddd;'>{esc(v)}</td>"
        for v in row0_vals
    )
    tds0 = (
        f"<td style='padding:6px 10px;border:1px solid #ddd;white-space:nowrap;'>"
        f"{esc(row0_label)}</td>{cells0}"
    )

    tds1_list = []
    for v in row1_vals:
        style = "padding:6px 10px;border:1px solid #ddd;"
        if color_second_row:
            try:
                vv = float(v)
                style += f"color:{'red' if vv > color_thresh else 'green'};"
            except Exception:
                pass
        tds1_list.append(f"<td style='{style}'>{fmtv(v)}</td>")
    tds1 = (
        f"<td style='padding:6px 10px;border:1px solid #ddd;white-space:nowrap;'>"
        f"{esc(row1_label)}</td>{''.join(tds1_list)}"
    )

    st.markdown(
        f"""
        <div style="overflow-x:auto;">
          <table style="border-collapse:collapse;width:100%;table-layout:auto;">
            <tbody><tr>{tds0}</tr><tr>{tds1}</tr></tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_bw_grid(title, imgs_uint8, labels, cols_per_row=6):
    """Show many grayscale images in a tight grid with captions."""
    st.markdown(f"**{title}**")
    n = len(imgs_uint8)
    for i in range(0, n, cols_per_row):
        chunk_imgs = imgs_uint8[i : i + cols_per_row]
        chunk_labels = labels[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if j < len(chunk_imgs):
                cols[j].image(
                    chunk_imgs[j],
                    use_container_width=True,
                    clamp=True,
                )
                cols[j].caption(chunk_labels[j])
            else:
                cols[j].markdown("&nbsp;")


def metric_header(title: str, tooltip: str):
    """
    Render a section title with a small '?' tooltip icon next to it.
    Hovering the icon shows the explanation.
    """
    html = f"""
    <div style="display:flex;align-items:center;gap:6px;margin-top:0.5rem;margin-bottom:0.25rem;">
      <span style="font-weight:600;font-size:1.05rem;">{title}</span>
      <span
        title="{tooltip}"
        style="
          display:inline-block;
          width:16px;
          height:16px;
          line-height:16px;
          text-align:center;
          border-radius:50%;
          border:1px solid #666;
          font-size:0.75rem;
          cursor:help;
          background-color:#f5f5f5;
          color:#333;
        "
      >?</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
