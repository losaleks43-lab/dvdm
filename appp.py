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

# Try importing OpenAI (Handles missing library gracefully)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# -------------------------------------------------------------------
# 0. App Configuration & Styles
# -------------------------------------------------------------------
st.set_page_config(page_title="Financial Flow Auditor", layout="wide")

# Base categories (used for data editor options)
CATEGORY_COLORS = {
    "Revenue": "#4285F4",       # Blue (Sources)
    "COGS": "#DB4437",          # Red (Direct Costs)
    "Gross Profit": "#BDBDBD",  # Grey (Calculated Node)
    "R&D": "#AB47BC",           # Purple
    "Sales & Marketing": "#F4B400",  # Yellow
    "G&A": "#00ACC1",           # Teal
    "Other Opex": "#8D6E63",    # Brown
    "Tax": "#E91E63",           # Pink
    "Net Income": "#0F9D58",    # Green
    "Unallocated": "#9E9E9E",   # Grey (Reconciliation Gaps)
    "Eliminations": "#5f6368"   # Dark Grey (Inter-segment)
}

# Colorblind friendly palette options
PALETTES = {
    "Okabe Ito (default)": {
        "Revenue": "#0072B2",
        "Revenue segment": "#56B4E9",
        "COGS": "#D55E00",
        "R&D": "#CC79A7",
        "Sales & Marketing": "#E69F00",
        "G&A": "#F0E442",
        "Other Opex": "#999999",
        "Tax": "#000000",
        "Gross Profit": "#009E73",
        "Operating Profit": "#009E73",
        "Net Income": "#009E73",
        "Unallocated": "#999999",
        "Eliminations": "#555555",
        "Total Revenue": "#0072B2"
    },
    "Blue Green Contrast": {
        "Revenue": "#1f77b4",
        "Revenue segment": "#6baed6",
        "COGS": "#d62728",
        "R&D": "#9467bd",
        "Sales & Marketing": "#ff7f0e",
        "G&A": "#2ca02c",
        "Other Opex": "#8c564b",
        "Tax": "#7f7f7f",
        "Gross Profit": "#17becf",
        "Operating Profit": "#17becf",
        "Net Income": "#17becf",
        "Unallocated": "#bdbdbd",
        "Eliminations": "#636363",
        "Total Revenue": "#1f77b4"
    }
}


def hex_to_rgba(hex_color, alpha=1.0):
    """Convert #RRGGBB to rgba(r,g,b,a) string."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# -------------------------------------------------------------------
# 1. Backend Logic: The AI "Vision Auditor"
# -------------------------------------------------------------------

def get_openai_client():
    """Initializes OpenAI client from Secrets or Environment."""
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
    The Brain: Sends images to GPT-4o-mini to extract and reconcile data.
    """
    client = get_openai_client()
    if client is None:
        return None, None, None, "Error: OpenAI API Key not found."

    system_prompt = """
    You are an expert Financial Auditor. You do not just extract numbers; you RECONCILE them.

    YOUR GOAL: Create a clean dataset for a 'Sankey Diagram' that maps Revenue Sources -> Gross Profit -> Net Income.

    INPUTS:
    1. IMAGE A (P&L): The Consolidated Income Statement. This is the SOURCE OF TRUTH for Total Revenue, Total Expenses, and Net Income.
    2. IMAGE B (Segments): The Revenue Breakdown by Product or Geography.

    IMPORTANT - ALWAYS USE THE MOST RECENT YEAR / PERIOD:
    - If the P&L shows multiple periods (columns), you must use ONLY the most recent one (usually the rightmost column or the one with the latest date).
    - All numbers you output must refer to that same most recent period.
    - Ignore earlier years and periods for the numeric output.

    --- AUDIT ALGORITHM ---

    STEP 1: ESTABLISH TOTALS (From Image A)
    - Extract "Total Revenue" (or "Net Sales"). Call this [TR].
      CRITICAL: If the report lists "Gross Revenue" (with Excise/GST) and "Net Revenue", YOU MUST USE "NET REVENUE".
    - Identify the Business Model to find Direct Costs (COGS):
      * Retail or Manufacturing: Sum of "Cost of Materials", "Purchase of Stock", "Changes in Inventories", "Excise Duty".
      * Tech or Platform companies: "Cost of Revenues" (TAC, Data Centers, content acquisition, cloud infrastructure).
    - Extract Operating Expenses (R&D, Sales and Marketing, G&A, Other operating expenses) and Tax.

    STEP 2: RECONCILE REVENUE (From Image B)
    - Extract revenue segments (for example "Digital", "Retail"). Sum them up = [Sum_Seg].
    - Compare [Sum_Seg] vs [TR] for the same most recent period:
      * MATCH: If they are close (<5 percent difference), use the segments as is.
      * GAP: If [Sum_Seg] < [TR], add a segment "Unallocated Revenue" = [TR] - [Sum_Seg].
      * OVERFLOW: If [Sum_Seg] > [TR] (usually due to inter segment sales), look for "Eliminations". If not found, proportionally scale down segments to match [TR].

    STEP 3: OUTPUT
    - Return a JSON object with one list of lines.
    - Normalize category names to: "Revenue", "COGS", "R&D", "Sales & Marketing", "G&A", "Other Opex", "Tax".
    - All amounts must be numeric (no commas, no text).

    JSON FORMAT:
    {
        "company": "Company Name",
        "currency": "USD/INR/EUR",
        "period": "for example FY 2024 or 12M ended Dec 31, 2024",
        "audit_note": "Short explanation of how you reconciled the revenue and which period you used.",
        "lines": [
            {"item": "Segment A", "amount": 100, "category": "Revenue"},
            {"item": "Cost of Goods", "amount": 60, "category": "COGS"},
            {"item": "Tax", "amount": 10, "category": "Tax"}
        ]
    }
    """

    # Construct the visual payload
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

    # API Call
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

        # Post-processing to DataFrame
        df = pd.DataFrame(data.get("lines", []))
        if "category" not in df.columns:
            df["category"] = "Other Opex"
        df = df.rename(columns={"item": "Item", "amount": "Amount", "category": "Category"})

        return df, data.get("company"), data.get("currency"), data.get("audit_note")

    except Exception as e:
        return None, None, None, str(e)


# -------------------------------------------------------------------
# 2. Frontend UI: The "Place for Screenshots"
# -------------------------------------------------------------------

st.title("Financial Flow Auditor 🕵️‍♂️")
st.markdown(
    """
**How it works:**
1. Upload the **Income Statement** (so we get the correct Profit & Margins).
2. Upload the **Segment Breakdown** (so we know where the money comes from).
3. The AI reconciles them into one clean diagram.
"""
)

# Appearance controls
with st.sidebar:
    st.header("Visual settings")
    palette_name = st.selectbox("Color palette", list(PALETTES.keys()), index=0)

# --- THE TWO-COLUMN UPLOADER ---
col_input1, col_input2 = st.columns(2)

with col_input1:
    st.subheader("1. Master P&L")
    st.caption("Upload the Consolidated Income Statement.")
    pnl_file = st.file_uploader("Drop P&L Screenshot", type=["png", "jpg", "jpeg"], key="pnl")
    if pnl_file:
        st.image(pnl_file, use_container_width=True)

with col_input2:
    st.subheader("2. Revenue Splits")
    st.caption("Upload the Revenue by Segment/Product table.")
    seg_file = st.file_uploader("Drop Segment Screenshot", type=["png", "jpg", "jpeg"], key="seg")
    if seg_file:
        st.image(seg_file, use_container_width=True)

# --- ACTION BUTTON ---
if pnl_file:
    if st.button("Audit & Visualize", type="primary", use_container_width=True):
        with st.spinner("AI Auditor is analyzing the images..."):
            # 1. Encode
            pnl_b64 = encode_image(pnl_file)
            seg_b64 = encode_image(seg_file) if seg_file else None

            # 2. Analyze
            df_result, company, currency, note = audit_financials_with_vision(pnl_b64, seg_b64)

            if df_result is not None:
                st.session_state.raw_df = df_result
                st.session_state.company_info = f"{company} ({currency})"
                st.session_state.audit_note = note
                st.success("Audit Complete!")
            else:
                st.error(f"Analysis Failed: {note}")
else:
    st.info("👆 Please upload at least the Master P&L to begin.")

# -------------------------------------------------------------------
# 3. Visualization & Results
# -------------------------------------------------------------------

if "raw_df" in st.session_state:
    st.divider()

    # Header & Note
    st.header(f"Results: {st.session_state.company_info}")
    if st.session_state.audit_note:
        st.info(f"📝 **Auditor's Finding:** {st.session_state.audit_note}")

    # Layout: Data on Left, Chart on Right
    col_data, col_viz = st.columns([1, 2])

    with col_data:
        st.subheader("Reconciled Data")
        df = st.session_state.raw_df.copy()

        # Editable Table
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

    with col_viz:
        # --- CALCULATE SANKEY FLOWS ---
        grp = clean_df.groupby("Category")["Amount"].sum()

        # 1. Revenue
        rev_segments = clean_df[clean_df["Category"] == "Revenue"]
        total_revenue = grp.get("Revenue", 0)

        # 2. Costs
        cogs = grp.get("COGS", 0)
        gross_profit = total_revenue - cogs

        # 3. Opex
        opex_cats = ["R&D", "Sales & Marketing", "G&A", "Other Opex"]
        total_opex = sum(grp.get(c, 0) for c in opex_cats)

        # 4. Profit
        operating_profit = gross_profit - total_opex
        tax = grp.get("Tax", 0)
        net_income = operating_profit - tax

        # --- METRICS ROW ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Revenue", f"{total_revenue:,.0f}")
        if total_revenue > 0:
            m2.metric("Gross Margin", f"{(gross_profit / total_revenue) * 100:.1f}%")
            m3.metric("Op Margin", f"{(operating_profit / total_revenue) * 100:.1f}%")
            m4.metric("Net Margin", f"{(net_income / total_revenue) * 100:.1f}%")

        # --- PALETTE SELECTION FOR THIS CHART ---
        palette = PALETTES.get(palette_name, PALETTES["Okabe Ito (default)"])

        def color_for(name, kind=None):
            """
            kind can be 'segment', 'profit', 'cost' etc.
            We map segments to Revenue colors unless a more specific color exists.
            """
            if kind == "segment":
                base = palette.get("Revenue segment", palette.get("Revenue", "#0072B2"))
            else:
                base = palette.get(name, None)
                if base is None:
                    # Map profit style nodes
                    if name in ["Gross Profit", "Operating Profit", "Net Income", "Total Revenue"]:
                        base = palette.get("Gross Profit", "#009E73")
                    # Map cost like nodes
                    elif name in [
                        "COGS",
                        "R&D",
                        "Sales & Marketing",
                        "G&A",
                        "Other Opex",
                        "Tax",
                        "Unallocated",
                        "Eliminations",
                    ]:
                        base = palette.get("Other Opex", "#999999")
                    else:
                        base = "#999999"
            return base

        labels, sources, targets, values = [], [], [], []
        node_colors, link_colors = [], []
        label_idx = {}

        def get_idx(name, kind=None):
            if name not in label_idx:
                label_idx[name] = len(labels)
                labels.append(name)

                base_hex = color_for(name, kind=kind)
                # Nodes slightly transparent, links full color
                node_colors.append(hex_to_rgba(base_hex, 0.6))
            return label_idx[name]

        # Node: Segments -> Total Revenue
        for _, row in rev_segments.iterrows():
            s = get_idx(row["Item"], kind="segment")
            t = get_idx("Total Revenue")
            v = row["Amount"]

            sources.append(s)
            targets.append(t)
            values.append(v)
            link_colors.append(color_for("Revenue"))

        # Node: Total Revenue -> COGS and Gross Profit
        if cogs > 0:
            s = get_idx("Total Revenue")
            t = get_idx("COGS")
            sources.append(s)
            targets.append(t)
            values.append(cogs)
            link_colors.append(color_for("COGS"))

        s_tr = get_idx("Total Revenue")
        t_gp = get_idx("Gross Profit")
        sources.append(s_tr)
        targets.append(t_gp)
        values.append(gross_profit)
        link_colors.append(color_for("Gross Profit"))

        # Node: Gross Profit -> Opex and Operating Profit
        for cat in opex_cats:
            amt = grp.get(cat, 0)
            if amt > 0:
                s = get_idx("Gross Profit")
                t = get_idx(cat)
                sources.append(s)
                targets.append(t)
                values.append(amt)
                link_colors.append(color_for(cat))

        s_gp = get_idx("Gross Profit")
        t_op = get_idx("Operating Profit")
        sources.append(s_gp)
        targets.append(t_op)
        values.append(operating_profit)
        link_colors.append(color_for("Operating Profit"))

        # Node: Operating Profit -> Tax and Net Income
        if tax > 0:
            s = get_idx("Operating Profit")
            t = get_idx("Tax")
            sources.append(s)
            targets.append(t)
            values.append(tax)
            link_colors.append(color_for("Tax"))

        s_op = get_idx("Operating Profit")
        t_ni = get_idx("Net Income")
        sources.append(s_op)
        targets.append(t_ni)
        values.append(net_income)
        link_colors.append(color_for("Net Income"))

        # Render
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=20,
                        thickness=20,
                        line=dict(color="rgba(150,150,150,0.4)", width=0.5),
                        label=labels,
                        color=node_colors,
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color=link_colors,
                    ),
                    valueformat=",",
                )
            ]
        )

        # Labels in grey, flows carry the color
        fig.update_layout(
            font=dict(color="#555555", size=12),
            margin=dict(l=10, r=10, t=40, b=10),
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)


