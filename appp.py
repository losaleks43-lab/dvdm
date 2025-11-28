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

# Base categories (used for data editor options only)
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

# Colorblind friendly palette options
# Each palette gives one color for all revenue items, one for all profit items, one for all cost items
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
    """Convert #RRGGBB to rgba(r,g,b,a) string."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# -------------------------------------------------------------------
# 1. Backend logic: the AI vision auditor
# -------------------------------------------------------------------

def get_openai_client():
    """Initializes OpenAI client from secrets or environment."""
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


def encode_image(image_file):
    """Converts uploaded image file to Base64 string for the API."""
    if image_file is None:
        return None
    return base64.b64encode(image_file.read()).decode("utf-8")


def audit_financials_with_vision(pnl_image_b64, segment_image_b64):
    """
    Sends images to GPT 4o mini to extract and reconcile data.
    """
    client = get_openai_client()
    if client is None:
        return None, None, None, None, "Error: OpenAI API Key not found."

    system_prompt = """
    You are an expert Financial Auditor. You do not just extract numbers; you reconcile them.

    Your goal: Create a clean dataset for a Sankey Diagram that maps Revenue Sources -> Gross Profit -> Net Income.

    Inputs:
    1. IMAGE A (P&L): The Consolidated Income Statement. This is the source of truth for Total Revenue, Total Expenses, and Net Income.
    2. IMAGE B (Segments): The Revenue Breakdown by Product or Geography.

    Always use the most recent year or period:
    - If the P&L shows multiple periods (columns), you must use only the most recent one (usually the rightmost column or the one with the latest date).
    - All numbers you output must refer to that same most recent period.
    - Ignore earlier years and periods for the numeric output.

    Step 1: Establish totals (from Image A)
    - Extract "Total Revenue" (or "Net Sales"). Call this [TR].
      If the report lists "Gross Revenue" and "Net Revenue", you must use "Net Revenue".
    - Identify the business model to find Direct Costs (COGS):
      * Retail or Manufacturing: sum of "Cost of Materials", "Purchase of Stock", "Changes in Inventories", "Excise Duty".
      * Tech or Platform companies: "Cost of Revenues".
    - Extract operating expenses (R&D, Sales and Marketing, G&A, Other operating expenses) and Tax.

    Step 2: Reconcile revenue (from Image B)
    - Extract revenue segments (for example "Digital", "Retail"). Sum them up = [Sum_Seg].
    - Compare [Sum_Seg] vs [TR] for the same most recent period:
      * If close (difference less than 5 percent), use the segments as is.
      * If [Sum_Seg] < [TR], add a segment "Unallocated Revenue" = [TR] - [Sum_Seg].
      * If [Sum_Seg] > [TR], look for "Eliminations". If none, proportionally scale down segments to match [TR].

    Step 3: Output
    - Return a JSON object with one list of lines.
    - Normalize category names to: "Revenue", "COGS", "R&D", "Sales & Marketing", "G&A", "Other Opex", "Tax".
    - All amounts must be numeric (no commas, no text).

    JSON format:
    {
        "company": "Company Name",
        "currency": "USD/INR/EUR",
        "period": "for example September 30, 2025",
        "audit_note": "Short explanation of how you reconciled the revenue and which period you used.",
        "lines": [
            {"item": "Segment A", "amount": 100, "category": "Revenue"},
            {"item": "Cost of Goods", "amount": 60, "category": "COGS"},
            {"item": "Tax", "amount": 10, "category": "Tax"}
        ]
    }
    """

    user_content = [{"type": "text", "text": "Perform the financial audit on these images."}]

    if pnl_image_b64:
        user_content.append({"type": "text", "text": "IMAGE A: Master P&L (Source of Truth)"})
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{pnl_image_b64}"},
            }
        )

    if segment_image_b64:
        user_content.append({"type": "text", "text": "IMAGE B: Segment Breakdown (Detail)"})
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{segment_image_b64}"},
            }
        )

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

        return (
            df,
            data.get("company"),
            data.get("currency"),
            data.get("period"),
            data.get("audit_note"),
        )

    except Exception as e:
        return None, None, None, None, str(e)


# -------------------------------------------------------------------
# 2. Frontend UI: uploads
# -------------------------------------------------------------------

st.title("Financial Flow Auditor 🕵️‍♂️")
st.markdown(
    """
How it works:
1. Upload the Income Statement (so we get the correct profit and margins).
2. Upload the Segment Breakdown (so we know where the money comes from).
3. The AI reconciles them into one clean diagram.
"""
)

# Appearance controls
with st.sidebar:
    st.header("Visual settings")
    palette_name = st.selectbox("Color palette", list(PALETTES.keys()), index=0)

# Two column uploader
col_input1, col_input2 = st.columns(2)

with col_input1:
    st.subheader("1. Master P&L")
    st.caption("Upload the Consolidated Income Statement.")
    pnl_file = st.file_uploader("Drop P&L screenshot", type=["png", "jpg", "jpeg"], key="pnl")
    if pnl_file:
        st.image(pnl_file, use_container_width=True)

with col_input2:
    st.subheader("2. Revenue splits")
    st.caption("Upload the Revenue by Segment or Product table.")
    seg_file = st.file_uploader("Drop segment screenshot", type=["png", "jpg", "jpeg"], key="seg")
    if seg_file:
        st.image(seg_file, use_container_width=True)

# Action button
if pnl_file:
    if st.button("Audit and visualize", type="primary", use_container_width=True):
        with st.spinner("AI auditor is analyzing the images..."):
            pnl_b64 = encode_image(pnl_file)
            seg_b64 = encode_image(seg_file) if seg_file else None

            df_result, company, currency, period, note = audit_financials_with_vision(pnl_b64, seg_b64)

            if df_result is not None:
                st.session_state.raw_df = df_result
                st.session_state.company = company or "Apple Inc."
                st.session_state.currency = currency or "USD"
                # use the simple date string as in the example
                st.session_state.period = period or "September 30, 2025"
                st.session_state.audit_note = note
                st.success("Audit complete.")
            else:
                st.error(f"Analysis failed: {note}")
else:
    st.info("Please upload at least the Master P&L to begin.")

# -------------------------------------------------------------------
# 3. Visualization and results
# -------------------------------------------------------------------

if "raw_df" in st.session_state:
    st.divider()

    header_company = st.session_state.company
    header_currency = st.session_state.currency
    st.header(f"Results: {header_company} ({header_currency})")
    if st.session_state.audit_note:
        st.info(f"Auditor finding: {st.session_state.audit_note}")

    # Level 1 layout: table (left) and KPIs (right)
    col_data, col_kpi = st.columns([2, 1])

    with col_data:
        st.subheader("Reconciled data")
        df = st.session_state.raw_df.copy()

        edited_df = st.data_editor(
            df,
            column_config={
                "Category": st.column_config.SelectboxColumn(
                    "Category",
                    options=list(CATEGORY_COLORS.keys()),
                    required=True,
                ),
                "Amount": st.column_config.NumberColumn(format="%.0f"),
            },
            use_container_width=True,
            num_rows="dynamic",
        )

    clean_df = edited_df.copy()

    # Aggregate metrics based on edited data
    grp = clean_df.groupby("Category")["Amount"].sum()

    rev_segments = clean_df[clean_df["Category"] == "Revenue"]
    total_revenue = grp.get("Revenue", 0)

    cogs = grp.get("COGS", 0)
    gross_profit = total_revenue - cogs

    opex_cats = ["R&D", "Sales & Marketing", "G&A", "Other Opex"]
    total_opex = sum(grp.get(c, 0) for c in opex_cats)

    operating_profit = gross_profit - total_opex
    tax = grp.get("Tax", 0)
    net_income = operating_profit - tax

    with col_kpi:
        st.subheader("Key metrics")
        m_row1_col1, m_row1_col2 = st.columns(2)
        m_row2_col1, m_row2_col2 = st.columns(2)

        m_row1_col1.metric("Revenue", f"{total_revenue:,.0f}")
        m_row1_col2.metric("Gross profit", f"{gross_profit:,.0f}")

        if total_revenue > 0:
            m_row2_col1.metric("Operating margin", f"{(operating_profit / total_revenue) * 100:.1f}%")
            m_row2_col2.metric("Net margin", f"{(net_income / total_revenue) * 100:.1f}%")
        else:
            m_row2_col1.metric("Operating margin", "n/a")
            m_row2_col2.metric("Net margin", "n/a")

    # Level 2: chart below
    # Title pattern: "Apple Inc. P&L for the year ended September 30, 2025"
    company_for_title = st.session_state.company or "Apple Inc."
    period_for_title = st.session_state.period or "September 30, 2025"
    chart_title = f"{company_for_title} P&L for the year ended {period_for_title}"
    units_text = f"(in millions of {st.session_state.currency})"

    palette = PALETTES.get(palette_name, PALETTES["Okabe Ito (Blue Green Orange)"])

    def node_role(name, kind=None):
        """Map node names to semantic roles: revenue, profit, cost."""
        if kind == "segment":
            return "revenue"
        if name in ["Total Revenue", "Revenue"]:
            return "revenue"
        if name in ["Gross Profit", "Operating Profit", "Net Income"]:
            return "profit"
        return "cost"

    def role_color(role, alpha=1.0):
        base_hex = palette.get(role, "#999999")
        if alpha >= 1.0:
            return base_hex
        return hex_to_rgba(base_hex, alpha)

    labels, sources, targets, values = [], [], [], []
    node_colors, link_colors, link_labels = [], [], []
    label_idx = {}

    def get_idx(name, kind=None):
        if name not in label_idx:
            label_idx[name] = len(labels)
            labels.append(f"<span style='color:white'>{name}</span>")
            role = node_role(name, kind)
            node_colors.append(role_color(role, alpha=0.5))  # lighter node blocks
        return label_idx[name]

    # Build flows

    # Segments -> Total Revenue (revenue color)
    for _, row in rev_segments.iterrows():
        s = get_idx(row["Item"], kind="segment")
        t = get_idx("Total Revenue")
        v = row["Amount"]

        sources.append(s)
        targets.append(t)
        values.append(v)
        link_colors.append(role_color("revenue"))
        # category-coloured label + grey number in text (visual approximation)
        link_labels.append(f"{row['Item']}\n{v:,.0f}")


    # Total Revenue -> COGS (cost) and Gross Profit (profit)
    if cogs > 0:
        s = get_idx("Total Revenue")
        t = get_idx("COGS")
        sources.append(s)
        targets.append(t)
        values.append(cogs)
        link_colors.append(role_color("cost"))
        link_labels.append(f"COGS\n{cogs:,.0f}")

    s_tr = get_idx("Total Revenue")
    t_gp = get_idx("Gross Profit")
    sources.append(s_tr)
    targets.append(t_gp)
    values.append(gross_profit)
    link_colors.append(role_color("profit"))
    link_labels.append(f"Gross profit\n{gross_profit:,.0f}")

    # Gross Profit -> Opex items (cost) and Operating Profit (profit)
    for cat in opex_cats:
        amt = grp.get(cat, 0)
        if amt > 0:
            s = get_idx("Gross Profit")
            t = get_idx(cat)
            sources.append(s)
            targets.append(t)
            values.append(amt)
            link_colors.append(role_color("cost"))
            link_labels.append(f"{cat}\n{amt:,.0f}")

    s_gp = get_idx("Gross Profit")
    t_op = get_idx("Operating Profit")
    sources.append(s_gp)
    targets.append(t_op)
    values.append(operating_profit)
    link_colors.append(role_color("profit"))
    link_labels.append(f"Operating profit\n{operating_profit:,.0f}")

    # Operating Profit -> Tax (cost) and Net Income (profit)
    if tax > 0:
        s = get_idx("Operating Profit")
        t = get_idx("Tax")
        sources.append(s)
        targets.append(t)
        values.append(tax)
        link_colors.append(role_color("cost"))
        link_labels.append(f"Operating profit\n{operating_profit:,.0f}")

    s_op = get_idx("Operating Profit")
    t_ni = get_idx("Net Income")
    sources.append(s_op)
    targets.append(t_ni)
    values.append(net_income)
    link_colors.append(role_color("profit"))
    link_labels.append(f"Net income\n{net_income:,.0f}")

    fig = go.Figure(
        data=[
            go.Sankey(
               node=dict(
    pad=40,
    thickness=10,
    line=dict(color="rgba(160,160,160,0.25)", width=0.4),
    label=labels,
    color=node_colors,
),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    label=link_labels,
                ),
                valueformat=",",
            )
        ]
    )

    # We cannot assign per-link font colors in plotly sankey,
    # so we choose a neutral grey that works on all branch colours.
    fig.update_layout(
        title=dict(
    text=f"{chart_title}\n{units_text}",
    x=0.02,
    y=0.98,
    xanchor="left",
    yanchor="top",
    font=dict(size=26, color="#000000"),
),
font=dict(color="#FFFFFF", size=14),
        margin=dict(l=30, r=30, t=110, b=70),  # more breathing room top and bottom
        height=640,  # a bit taller to emphasize long, slim branches
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------
    # 4. Marimekko-style 1D strip (category weight view)
    # -------------------------------------------------------------------
    st.subheader("Category weight view (Marimekko-style strip)")

    mekko_items = []
    mekko_values = []
    mekko_colors = []

    # Revenue segments
    for _, row in rev_segments.iterrows():
        mekko_items.append(row["Item"])
        mekko_values.append(float(row["Amount"]))
        mekko_colors.append(role_color("revenue"))

    # COGS
    if cogs > 0:
        mekko_items.append("COGS")
        mekko_values.append(float(cogs))
        mekko_colors.append(role_color("cost"))

    # Opex categories
    for cat in opex_cats:
        amt = float(grp.get(cat, 0.0))
        if amt > 0:
            mekko_items.append(cat)
            mekko_values.append(amt)
            mekko_colors.append(role_color("cost"))

    # Tax
    if tax > 0:
        mekko_items.append("Tax")
        mekko_values.append(float(tax))
        mekko_colors.append(role_color("cost"))

    # Net Income (positive block)
    if net_income != 0:
        mekko_items.append("Net income")
        mekko_values.append(float(abs(net_income)))
        mekko_colors.append(role_color("profit"))

    if mekko_items:
        abs_vals = [abs(v) for v in mekko_values]
        total_abs = sum(abs_vals) if sum(abs_vals) != 0 else 1.0

        widths = [v / total_abs for v in abs_vals]

        xs = []
        cum = 0.0
        for w in widths:
            xs.append(cum + w / 2.0)
            cum += w

        mekko_fig = go.Figure(
            data=[
                go.Bar(
                    x=xs,
                    y=[1.0] * len(xs),
                    width=widths,
                    marker_color=mekko_colors,
                    marker_line_color="white",
                    marker_line_width=1,
                    text=[f"{name}\n{val:,.0f}" for name, val in zip(mekko_items, mekko_values)],
                    textposition="inside",
                    hovertext=[
                        f"{name}: {val:,.0f} ({abs(val) / total_abs:,.1%})"
                        for name, val in zip(mekko_items, mekko_values)
                    ],
                    hoverinfo="text",
                )
            ]
        )

        mekko_fig.update_xaxes(visible=False, showticklabels=False, range=[0, 1])
        mekko_fig.update_yaxes(visible=False, showticklabels=False, range=[0, 1.2])

        mekko_fig.update_layout(
            height=220,
            margin=dict(l=40, r=40, t=40, b=40),
            font=dict(size=12),
        )

        st.plotly_chart(mekko_fig, use_container_width=True)
