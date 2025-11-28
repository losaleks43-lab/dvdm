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

# Palette: one colour for all revenues, one for profits, one for costs
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
    """
    Sends images to GPT-4o-mini to extract and reconcile data.
    Returns: df, company, currency, period, audit_note OR (None, ..., error_msg)
    """
    client = get_openai_client()
    if client is None:
        return None, None, None, None, "Error: OpenAI API Key not found."

    system_prompt = """
    You are an expert Financial Auditor. You do not just extract numbers; you reconcile them.

    Goal: Create a clean dataset for a Sankey diagram:
    Revenue segments -> Total Revenue -> Gross Profit -> Operating Profit -> Net Income.

    Always use ONLY the most recent period (latest year / right-most column).

    Output JSON:
    {
      "company": "Company Name",
      "currency": "USD/EUR/INR",
      "period": "for example September 30, 2025",
      "audit_note": "Short note on reconciliation and period used",
      "lines": [
        {"item": "iPhone", "amount": 209586, "category": "Revenue"},
        {"item": "Total net sales", "amount": 416161, "category": "Revenue"},
        {"item": "Cost of sales", "amount": 220960, "category": "COGS"},
        {"item": "Gross margin", "amount": 195201, "category": "Gross Profit"},
        {"item": "Research and development", "amount": 34550, "category": "R&D"},
        {"item": "Selling, general and administrative", "amount": 27501, "category": "Sales & Marketing"},
        {"item": "Total operating expenses", "amount": 62151, "category": "Other Opex"},
        {"item": "Provision for income taxes", "amount": 20719, "category": "Tax"},
        {"item": "Net income", "amount": 112010, "category": "Net Income"}
      ]
    }
    All amounts must be numeric only (no commas, no currency symbols).
    """

    user_content = [{"type": "text", "text": "Perform the financial audit on these images."}]

    if pnl_image_b64:
        user_content.append({"type": "text", "text": "IMAGE A: Master P&L (Source of Truth)"})
        user_content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pnl_image_b64}"}}
        )

    if segment_image_b64:
        user_content.append({"type": "text", "text": "IMAGE B: Segment Breakdown (Detail)"})
        user_content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{segment_image_b64}"}}
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

        if df.empty:
            return None, None, None, None, "AI returned no data lines."

        # Robust column normalisation (handles amount/Amount, category/Category, item/Item)
        lower_map = {c.lower(): c for c in df.columns}

        if "amount" in lower_map and "Amount" not in df.columns:
            df["Amount"] = df[lower_map["amount"]]
        if "category" in lower_map and "Category" not in df.columns:
            df["Category"] = df[lower_map["category"]]
        if "item" in lower_map and "Item" not in df.columns:
            df["Item"] = df[lower_map["item"]]

        if not {"Item", "Amount", "Category"}.issubset(df.columns):
            return None, None, None, None, f"AI response missing required fields. Got columns: {list(df.columns)}"

        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)

        return (
            df[["Item", "Amount", "Category"]],
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
1. Upload the Income Statement (P&L).  
2. Optionally upload the revenue by segment table.  
3. The AI reconciles them into one flow diagram.  
"""
)

with st.sidebar:
    st.header("Visual settings")
    palette_name = st.selectbox("Color palette", list(PALETTES.keys()), index=0)

col_input1, col_input2 = st.columns(2)

with col_input1:
    st.subheader("1. Master P&L")
    st.caption("Upload the Consolidated Income Statement.")
    pnl_file = st.file_uploader("Drop P&L screenshot", type=["png", "jpg", "jpeg"], key="pnl")
    if pnl_file:
        st.image(pnl_file, use_container_width=True)

with col_input2:
    st.subheader("2. Revenue splits")
    st.caption("Upload the revenue by segment/product table.")
    seg_file = st.file_uploader("Drop segment screenshot", type=["png", "jpg", "jpeg"], key="seg")
    if seg_file:
        st.image(seg_file, use_container_width=True)

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
                st.session_state.period = period or "September 30, 2025"
                st.session_state.audit_note = note
                st.success("Audit complete.")
            else:
                st.session_state.pop("raw_df", None)
                st.error(f"Analysis failed: {note}")
else:
    st.info("Please upload at least the Master P&L to begin.")

# -------------------------------------------------------------------
# 3. Visualization and results
# -------------------------------------------------------------------

if "raw_df" in st.session_state:
    st.divider()

    df = st.session_state.raw_df.copy()

    st.header(f"Results: {st.session_state.company} ({st.session_state.currency})")
    if st.session_state.audit_note:
        st.info(f"Auditor finding: {st.session_state.audit_note}")

    # Guard
    if not {"Category", "Amount", "Item"}.issubset(df.columns):
        st.error(f"Reconciled data is missing required columns. Got: {list(df.columns)}")
        st.dataframe(df)
    else:
        # Level 1 layout: table (left) and KPIs (right)
        col_data, col_kpi = st.columns([2, 1])

        with col_data:
            st.subheader("Reconciled data")
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

        # -------------------------------------------------------------------
        # Apple-safe aggregation: avoid double-counting "Total net sales"
        # -------------------------------------------------------------------
        clean_df["item_norm"] = clean_df["Item"].str.lower().str.strip()

        total_row_mask = clean_df["item_norm"].str.contains("total net sales") | \
                         clean_df["item_norm"].str.contains("total revenue")

        total_row = clean_df[total_row_mask].sort_values("Amount", ascending=False)

        # Revenue segments: all revenue rows except the explicit total line
        rev_segments = clean_df[
            (clean_df["Category"] == "Revenue") & (~total_row_mask)
        ].copy()

        if not total_row.empty:
            total_revenue = float(total_row["Amount"].iloc[0])
        else:
            total_revenue = float(rev_segments["Amount"].sum())

        # Other aggregates based on categories
        grp = clean_df.groupby("Category")["Amount"].sum()

        cogs = float(grp.get("COGS", 0.0))
        gross_profit = total_revenue - cogs

        opex_cats = ["R&D", "Sales & Marketing", "G&A", "Other Opex"]
        total_opex = float(sum(grp.get(c, 0.0) for c in opex_cats))

        operating_profit = gross_profit - total_opex
        tax = float(grp.get("Tax", 0.0))
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

        # -------------------------------------------------------------------
        # 3b. Flow diagram
        # -------------------------------------------------------------------
        company_for_title = st.session_state.company or "Apple Inc."
        period_for_title = st.session_state.period or "September 30, 2025"
        chart_title = f"{company_for_title} P&L for the year ended {period_for_title}"

        palette = PALETTES.get(palette_name, PALETTES["Okabe Ito (Blue Green Orange)"])

        def node_role(name, kind=None):
            if kind == "segment":
                return "revenue"
            if name in ["Total Revenue", "Revenue"]:
                return "revenue"
            if name in ["Gross Profit", "Operating Profit", "Net Income"]:
                return "profit"
            return "cost"

        def role_color(role, alpha=1.0):
            base_hex = palette.get(role, "#999999")
            return hex_to_rgba(base_hex, alpha) if alpha < 1 else base_hex

        labels, sources, targets, values = [], [], [], []
        node_colors, link_colors, link_labels = [], [], []
        label_idx = {}

        def get_idx(name, kind=None):
            if name not in label_idx:
                label_idx[name] = len(labels)
                labels.append(name)
                node_colors.append(role_color(node_role(name, kind), alpha=0.5))
            return label_idx[name]

        # Segments -> Total Revenue
        for _, row in rev_segments.iterrows():
            s = get_idx(row["Item"], kind="segment")
            t = get_idx("Total Revenue")
            v = float(row["Amount"])
            sources.append(s)
            targets.append(t)
            values.append(v)
            link_colors.append(role_color("revenue"))
            link_labels.append(f"{row['Item']} · {v:,.0f}")

        # Total Revenue -> COGS and Gross Profit
        if cogs > 0:
            s = get_idx("Total Revenue")
            t = get_idx("COGS")
            sources.append(s)
            targets.append(t)
            values.append(cogs)
            link_colors.append(role_color("cost"))
            link_labels.append(f"COGS · {cogs:,.0f}")

        s_tr = get_idx("Total Revenue")
        t_gp = get_idx("Gross Profit")
        sources.append(s_tr)
        targets.append(t_gp)
        values.append(gross_profit)
        link_colors.append(role_color("profit"))
        link_labels.append(f"Gross profit · {gross_profit:,.0f}")

        # Gross Profit -> Opex and Operating Profit
        for cat in opex_cats:
            amt = float(grp.get(cat, 0.0))
            if amt > 0:
                s = get_idx("Gross Profit")
                t = get_idx(cat)
                sources.append(s)
                targets.append(t)
                values.append(amt)
                link_colors.append(role_color("cost"))
                link_labels.append(f"{cat} · {amt:,.0f}")

        s_gp = get_idx("Gross Profit")
        t_op = get_idx("Operating Profit")
        sources.append(s_gp)
        targets.append(t_op)
        values.append(operating_profit)
        link_colors.append(role_color("profit"))
        link_labels.append(f"Operating profit · {operating_profit:,.0f}")

        # Operating Profit -> Tax and Net Income
        if tax > 0:
            s = get_idx("Operating Profit")
            t = get_idx("Tax")
            sources.append(s)
            targets.append(t)
            values.append(tax)
            link_colors.append(role_color("cost"))
            link_labels.append(f"Tax · {tax:,.0f}")

        s_op = get_idx("Operating Profit")
        t_ni = get_idx("Net Income")
        sources.append(s_op)
        targets.append(t_ni)
        values.append(net_income)
        link_colors.append(role_color("profit"))
        link_labels.append(f"Net income · {net_income:,.0f}")

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
                    ),
                    valueformat=",",
                )
            ]
        )

        fig.update_layout(
            title=dict(
                text=chart_title,
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                font=dict(size=24, color="#222222"),
            ),
            font=dict(color="#555555", size=11),
            margin=dict(l=30, r=30, t=110, b=70),
            height=640,
        )

        st.plotly_chart(fig, use_container_width=True)
