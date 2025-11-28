import streamlit as st

st.write("### DEBUG: Vision Auditor build 2025-11-28")

# app.py
# "How X Makes Money" - Vision Auditor Edition
# VERSION: Single-File, Multi-Screenshot Reconciliation

import os
import json
import base64
import pandas as pd
import plotly.graph_objects as go

# Try importing OpenAI (handles missing library gracefully)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# -------------------------------------------------------------------
# 0. App configuration and styles
# -------------------------------------------------------------------
st.set_page_config(page_title="Financial Flow Auditor", layout="wide")

CATEGORY_COLORS = {
    "Revenue": "#4285F4",
    "COGS": "#DB4437",
    "Gross Profit": "#BDBDBD",
    "R&D": "#AB47BC",
    "Sales & Marketing": "#F4B400",
    "G&A": "#00ACC1",
    "Other Opex": "#8D6E63",
    "Tax": "#E91E63",
    "Net Income": "#0F9D58",
    "Unallocated": "#9E9E9E",
    "Eliminations": "#5f6368",
}

PALETTES = {
    "Okabe Ito (Blue Green Orange)": {
        "revenue": "#0072B2",
        "profit": "#009E73",
        "cost": "#E69F00",
    },
    "Colorblind Safe (Purple Teal Orange)": {
        "revenue": "#CC79A7",
        "profit": "#009E73",
        "cost": "#E69F00",
    },
    "High Contrast (Blue Green Red)": {
        "revenue": "#1f77b4",
        "profit": "#2ca02c",
        "cost": "#d62728",
    },
    "Muted (Navy Olive Maroon)": {
        "revenue": "#1b4f72",
        "profit": "#7d6608",
        "cost": "#922b21",
    },
    "Cyan Magenta Yellow": {
        "revenue": "#00bcd4",
        "profit": "#e91e63",
        "cost": "#ffc107",
    },
}

def hex_to_rgba(hex_color, alpha=1.0):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# -------------------------------------------------------------------
# 1. Backend logic: the AI vision auditor
# -------------------------------------------------------------------

def get_openai_client():
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)

def encode_image(image_file):
    if image_file is None:
        return None
    return base64.b64encode(image_file.read()).decode("utf-8")

def audit_financials_with_vision(pnl_image_b64, segment_image_b64):
    client = get_openai_client()
    if client is None:
        return None, None, None, None, "Error: OpenAI API Key not found."

    system_prompt = """
    You are an expert Financial Auditor. Use ONLY the most recent period.
    Output normalized JSON with Revenue, COGS, Opex, Tax, etc.
    """

    user_content = [{"type": "text", "text": "Perform the financial audit."}]

    if pnl_image_b64:
        user_content.append({"type": "text", "text": "IMAGE A: Master P&L"})
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pnl_image_b64}"}})

    if segment_image_b64:
        user_content.append({"type": "text", "text": "IMAGE B: Segments"})
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{segment_image_b64}"}})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)

        df = pd.DataFrame(data.get("lines", []))
        if "category" not in df.columns:
            df["category"] = "Other Opex"
        df = df.rename(columns={"item": "Item", "amount": "Amount", "category": "Category"})

        return df, data.get("company"), data.get("currency"), data.get("period"), data.get("audit_note")
    except Exception as e:
        return None, None, None, None, str(e)


# -------------------------------------------------------------------
# 2. Frontend UI: uploads
# -------------------------------------------------------------------

st.title("Financial Flow Auditor 🕵️‍♂️")
st.markdown("""
1. Upload P&L  
2. Upload Revenue Segments  
3. Get a reconciled Sankey  
""")

with st.sidebar:
    st.header("Visual settings")
    palette_name = st.selectbox("Color palette", list(PALETTES.keys()), index=0)

col1, col2 = st.columns(2)

with col1:
    pnl_file = st.file_uploader("Upload P&L", type=["png", "jpg", "jpeg"])
    if pnl_file:
        st.image(pnl_file, use_container_width=True)

with col2:
    seg_file = st.file_uploader("Upload Segments", type=["png", "jpg", "jpeg"])
    if seg_file:
        st.image(seg_file, use_container_width=True)

if pnl_file:
    if st.button("Audit & Visualize", use_container_width=True):
        pnl_b64 = encode_image(pnl_file)
        seg_b64 = encode_image(seg_file) if seg_file else None

        df, company, currency, period, note = audit_financials_with_vision(pnl_b64, seg_b64)

        if df is not None:
            st.session_state.raw_df = df
            st.session_state.company = company or "Apple Inc."
            st.session_state.currency = currency or "USD"
            st.session_state.period = period or "September 30, 2025"
            st.session_state.audit_note = note
            st.success("Audit completed!")
        else:
            st.error(f"Failed: {note}")
else:
    st.info("Upload your P&L to begin.")

# -------------------------------------------------------------------
# 3. Visualization
# -------------------------------------------------------------------

if "raw_df" in st.session_state:
    st.divider()

    df = st.session_state.raw_df.copy()

    st.header(f"Results: {st.session_state.company} ({st.session_state.currency})")

    if st.session_state.audit_note:
        st.info(st.session_state.audit_note)

    col_data, col_kpi = st.columns([2, 1])

    with col_data:
        edited = st.data_editor(
            df,
            column_config={
                "Category": st.column_config.SelectboxColumn("Category", options=list(CATEGORY_COLORS.keys())),
                "Amount": st.column_config.NumberColumn(format="%.0f"),
            },
            use_container_width=True,
            num_rows="dynamic",
        )

    df = edited.copy()
    grp = df.groupby("Category")["Amount"].sum()

    rev_segments = df[df["Category"] == "Revenue"]
    total_revenue = grp.get("Revenue", 0)
    cogs = grp.get("COGS", 0)
    gross_profit = total_revenue - cogs
    opex_cats = ["R&D", "Sales & Marketing", "G&A", "Other Opex"]
    total_opex = sum(grp.get(x, 0) for x in opex_cats)
    operating_profit = gross_profit - total_opex
    tax = grp.get("Tax", 0)
    net_income = operating_profit - tax

    with col_kpi:
        st.subheader("KPIs")
        st.metric("Revenue", f"{total_revenue:,.0f}")
        st.metric("Gross Profit", f"{gross_profit:,.0f}")
        if total_revenue > 0:
            st.metric("Operating Margin", f"{(operating_profit/total_revenue)*100:.1f}%")
            st.metric("Net Margin", f"{(net_income/total_revenue)*100:.1f}%")

    palette = PALETTES[palette_name]

    def node_role(name, kind=None):
        if kind == "segment":
            return "revenue"
        if name in ["Total Revenue", "Revenue"]:
            return "revenue"
        if name in ["Gross Profit", "Operating Profit", "Net Income"]:
            return "profit"
        return "cost"

    def role_color(role, alpha=1.0):
        base = palette.get(role, "#999999")
        return hex_to_rgba(base, alpha) if alpha < 1 else base

    labels, sources, targets, values = [], [], [], []
    node_colors, link_colors, link_labels = [], [], []
    label_idx = {}

    def get_idx(name, kind=None):
        if name not in label_idx:
            label_idx[name] = len(labels)
            labels.append(name)
            node_colors.append(role_color(node_role(name, kind), alpha=0.5))
        return label_idx[name]

    # Build flows
    for _, row in rev_segments.iterrows():
        s = get_idx(row["Item"], "segment")
        t = get_idx("Total Revenue")
        v = row["Amount"]
        sources.append(s)
        targets.append(t)
        values.append(v)
        link_colors.append(role_color("revenue"))
        link_labels.append(f"{row['Item']} · {v:,.0f}")

    if cogs > 0:
        s = get_idx("Total Revenue")
        t = get_idx("COGS")
        sources.append(s)
        targets.append(t)
        values.append(cogs)
        link_colors.append(role_color("cost"))
        link_labels.append(f"COGS · {cogs:,.0f}")

    s = get_idx("Total Revenue")
    t = get_idx("Gross Profit")
    sources.append(s)
    targets.append(t)
    values.append(gross_profit)
    link_colors.append(role_color("profit"))
    link_labels.append(f"Gross Profit · {gross_profit:,.0f}")

    for cat in opex_cats:
        amt = grp.get(cat, 0)
        if amt > 0:
            s = get_idx("Gross Profit")
            t = get_idx(cat)
            sources.append(s)
            targets.append(t)
            values.append(amt)
            link_colors.append(role_color("cost"))
            link_labels.append(f"{cat} · {amt:,.0f}")

    s = get_idx("Gross Profit")
    t = get_idx("Operating Profit")
    sources.append(s)
    targets.append(t)
    values.append(operating_profit)
    link_colors.append(role_color("profit"))
    link_labels.append(f"Operating Profit · {operating_profit:,.0f}")

    if tax > 0:
        s = get_idx("Operating Profit")
        t = get_idx("Tax")
        sources.append(s)
        targets.append(t)
        values.append(tax)
        link_colors.append(role_color("cost"))
        link_labels.append(f"Tax · {tax:,.0f}")

    s = get_idx("Operating Profit")
    t = get_idx("Net Income")
    sources.append(s)
    targets.append(t)
    values.append(net_income)
    link_colors.append(role_color("profit"))
    link_labels.append(f"Net Income · {net_income:,.0f}")

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=40,
                    thickness=10,
                    line=dict(color="rgba(150,150,150,0.25)", width=0.5),
                    label=labels,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    label=link_labels,
                )
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=f"{st.session_state.company} P&L for the year ended {st.session_state.period}",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            font=dict(size=26, color="#222222"),
        ),
        font=dict(size=11, color="#666"),
        margin=dict(l=40, r=40, t=120, b=80),
        height=650,
    )

    st.plotly_chart(fig, use_container_width=True)
