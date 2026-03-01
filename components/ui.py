"""
Reusable HTML/Streamlit UI components.
All st.markdown calls that contain HTML must pass unsafe_allow_html=True.
"""
import streamlit as st
from utils.styles import COLORS, hex_to_rgba


# -- Hero banner ---------------------------------------------------------------
def hero(icon, title, subtitle, paper_title="", paper_url=""):
    if paper_url:
        paper_html = (
            '<a class="paper-link" href="' + paper_url + '" target="_blank">'
            + "&#128196; " + paper_title
            + "</a>"
        )
    else:
        paper_html = ""

    html = (
        '<div class="hero">'
        + "<h1>" + icon + " " + title + "</h1>"
        + "<p>" + subtitle + "</p>"
        + ("<br>" + paper_html if paper_html else "")
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# -- Section header ------------------------------------------------------------
def sec_header(text, color="var(--c1)"):
    st.markdown(
        '<div class="sec-header" style="color:' + color + ';">' + text + "</div>",
        unsafe_allow_html=True,
    )


# -- Info callout --------------------------------------------------------------
def callout(html_content, warning=False):
    css_class = "callout-warn" if warning else "callout"
    st.markdown(
        '<div class="' + css_class + '">' + html_content + "</div>",
        unsafe_allow_html=True,
    )


# -- Math block ----------------------------------------------------------------
def math_block(text):
    st.markdown(
        '<div class="math-block">' + text + "</div>",
        unsafe_allow_html=True,
    )


# -- Example block -------------------------------------------------------------
def example_block(text):
    st.markdown(
        '<div class="example-block">' + text + "</div>",
        unsafe_allow_html=True,
    )


# -- Viz caption ---------------------------------------------------------------
def viz_caption(text):
    st.markdown(
        '<div class="viz-caption">&#128161; ' + text + "</div>",
        unsafe_allow_html=True,
    )


# -- Pros / Cons card pair -----------------------------------------------------
def pros_cons(pros, cons):
    """pros / cons: list of (badge_label, explanation_sentence)"""

    def _build_items(items, badge_class):
        out = ""
        for label, explanation in items:
            out += (
                "<div style='margin-bottom:1.1rem;'>"
                + "<span class='badge " + badge_class + "'>" + label + "</span>"
                + "<div style='color:#cbd5e1;font-size:16px;margin-top:0.35rem;"
                + "line-height:1.75;padding-left:0.5rem;"
                + "border-left:2px solid rgba(255,255,255,0.1);'>"
                + explanation
                + "</div>"
                + "</div>"
            )
        return out

    c1, c2 = st.columns(2)

    with c1:
        html = (
            "<div class='card'>"
            + "<div class='sec-header' style='color:#34d399;'>&#9989; Pros</div>"
            + _build_items(pros, "badge-pro")
            + "</div>"
        )
        st.markdown(html, unsafe_allow_html=True)

    with c2:
        html = (
            "<div class='card'>"
            + "<div class='sec-header' style='color:#f87171;'>&#10060; Cons</div>"
            + _build_items(cons, "badge-con")
            + "</div>"
        )
        st.markdown(html, unsafe_allow_html=True)


# -- Notable uses card ---------------------------------------------------------
def notable_uses(uses):
    badges = "".join(
        "<span class='badge badge-use'>" + u + "</span>" for u in uses
    )
    html = (
        "<div class='card' style='margin-top:0.3rem;'>"
        + "<div class='sec-header'>&#127963; Notable Uses</div>"
        + badges
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# -- Radar fill colour helper --------------------------------------------------
def radar_fill(hex_color):
    return hex_to_rgba(hex_color, 0.12)
