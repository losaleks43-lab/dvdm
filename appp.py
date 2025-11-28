# app.py
# "How X Makes Money" – AI-assisted P&L Sankey
# With: most recent year logic + color-blind-safe themes

import os
import re
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# PDF text extraction
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# OpenAI client
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# -------------------------------------------------------------------
# 0. App configuration
# -------------------------------------------------------------------

st.set_page_config(page_title="How X Makes Money", layout="wide")

CATEGORIES = [
    "Revenue",
    "COGS",
    "R&D",
    "Sales & Marketing",
    "G&A",
    "Other Opex",
    "Tax",
    "Ignore",
]

# -------------------------------------------------------------------
# 1. Helpers: guessing categories, cleaning, colors, column choice
# -------------------------------------------------------------------


def guess_category(name: str) -> str:
    """Keyword-based guess for a line item category."""
    if not isinstance(name, str):
        return "Ignore"
    n = name.lower()

    # Revenue
    if any(
        w in n
        for w in [
            "revenue",
            "revenues",
            "net sales",
            "sales",
            "subscriptions",
            "subscription",
            "licensing",
            "license",
            "ads",
            "advertising",
            "cloud",
            "services",
            "membership",
        ]
    ):
        return "Revenue"

    # COGS / cost of revenues
    if any(
        w in n
        for w in [
            "cost of revenues",
            "cost of revenue",
            "cost of goods",
            "cost of sales",
            "cogs",
        ]
    ):
        return "COGS"

    # R&D
    if any(w in n for w in ["research", "r&d", "development"]):
        return "R&D"

    # Sales & Marketing
    if any(
        w in n
        for w in [
            "selling",
            "sales and marketing",
            "sales & marketing",
            "marketing",
            "advertising",
        ]
    ):
        return "Sales & Marketing"

    # G&A
    if any(
        w in n for w in ["general and administrative", "g&a", "administrative", "admin"]
    ):
        return "G&A"

    # Tax
    if "tax" in n:
        return "Tax"

    # Fallback
    return "Other Opex"


def load_sample_df() -> pd.DataFrame:
    """Sample income statement."""
    data = {
        "Item": [
            "Search advertising revenue",
            "YouTube advertising revenue",
            "Cloud services revenue",
            "Other revenue",
            "Cost of revenues",
            "R&D",
            "Sales and marketing",
            "General and administrative",
            "Other operating expenses",
            "Income tax expense",
        ],
        "Amount": [
            72000,
            33000,
            35000,
            6000,
            80000,
            35000,
            25000,
            15000,
            4000,
            8000,
        ],
    }
    return pd.DataFrame(data)


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df = df.dropna(subset=["Amount"])
    return df


def lighten(hex_color: str, factor: float = 0.3) -> str:
    """
    Blend a hex color with white by `factor` (0..1).
    Used to make nodes lighter than links.
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return "#cccccc"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)

    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB to rgba(r,g,b,a)."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(150,150,150,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


COLOR_THEMES = {
    "Classic (Blue / Red / Green)": {
        "revenue_seg": "#4C78A8",
        "revenue_total": "#3069E0",
        "cogs": "#E15759",
        "opex_rnd": "#B279A2",
        "opex_sm": "#F28E2B",
        "opex_ga": "#76B7B2",
        "opex_other": "#FFBE7D",
        "profit": "#59A14F",
        "tax": "#EDC948",
        "other": "#9C755F",
        "text": "#444444",
        "background": "white",
    },
    # Okabe–Ito color-blind-safe
    "Colorblind-safe": {
        "revenue_seg": "#0072B2",
        "revenue_total": "#005AB5",
        "cogs": "#D55E00",
        "opex_rnd": "#E69F00",
        "opex_sm": "#009E73",
        "opex_ga": "#CC79A7",
        "opex_other": "#56B4E9",
        "profit": "#009E73",
        "tax": "#F0E442",
        "other": "#999999",
        "text": "#363636",
        "background": "white",
    },
    "Muted Teal / Orange": {
        "revenue_seg": "#2A9D8F",
        "revenue_total": "#1E7A70",
        "cogs": "#E76F51",
        "opex_rnd": "#F4A261",
        "opex_sm": "#E9C46A",
        "opex_ga": "#A8DADC",
        "opex_other": "#457B9D",
        "profit": "#2A9D8F",
        "tax": "#F4A261",
        "other": "#264653",
        "text": "#3C3C3C",
        "background": "white",
    },
}


def choose_amount_column(df_no_item: pd.DataFrame) -> str:
    """
    Try to pick the 'Amount' column automatically.
    Priority:
    1. Column explicitly called 'Amount' / 'amount' / 'value'
    2. Column whose header contains a year (take the most recent year)
    3. Last numeric column as a fallback
    """
    cols = list(df_no_item.columns)

    # 1) explicit amount column
    for c in cols:
        if str(c).strip().lower() in ("amount", "value"):
            return c

    # 2) year columns like "2022", "FY 2023", "Year ended 2024"
    year_candidates = []
    for c in cols:
        m = re.search(r"(19|20)\d{2}", str(c))
        if m:
            year_candidates.append((int(m.group(0)), c))
    if year_candidates:
        year_candidates.sort()
        return year_candidates[-1][1]  # highest year

    # 3) last numeric column
    numeric_cols = []
    for c in cols:
        if pd.to_numeric(df_no_item[c], errors="coerce").notna().sum() > 0:
            numeric_cols.append(c)
    if numeric_cols:
        return numeric_cols[-1]

    raise ValueError("Could not automatically determine the amount column.")


# -------------------------------------------------------------------
# 2. Session defaults
# -------------------------------------------------------------------

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "detected_company" not in st.session_state:
    st.session_state.detected_company = "Example Corp"

# -------------------------------------------------------------------
# 3. OpenAI client and AI extraction helpers
# -------------------------------------------------------------------


def get_openai_client():
    """Create OpenAI client using Streamlit secrets or env var."""
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key or OpenAI is None:
        return None

    return OpenAI(api_key=api_key)


def extract_pnl_with_llm(raw_text: str):
    """
    Use GPT to extract an income statement from raw text.
    Two calls: revenue lines, then cost/expense/tax lines.
    """
    client = get_openai_client()
    if client is None:
        raise RuntimeError(
            "OpenAI client is not configured. "
            "Install openai and set OPENAI_API_KEY in secrets or env."
        )

    def call_llm(system_prompt: str, text: str) -> dict:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()

        # Remove optional ```json fences
        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
                if content.lower().startswith("json"):
                    content = content[4:]
            content = content.strip()

        return json.loads(content)

    revenue_system_prompt = """
You are a meticulous financial analyst.

Task: From the provided text, extract ONLY revenue / net sales lines that
belong to the income statement.

You MUST:
- Use ONLY numbers that clearly appear in the text (no guessing).
- If multiple years are shown in columns, ALWAYS use the MOST RECENT year,
  even if that year column is not the rightmost.
- If there is a breakdown (by segment, geography, product, etc.),
  return a separate line for each component, e.g.:
  "Net sales - Walmart U.S.", "Net sales - Sam's Club", etc.
- If there is only one revenue line (e.g. "Net sales"), return just that one.
- DO NOT include subtotals like "Total net sales", "Total revenues" if that
  subtotal just sums the other lines.
- DO NOT include cost or expense lines here.

Output ONLY valid JSON:

{
  "company": "Company name or null",
  "currency": "3 letter currency code or null",
  "lines": [
    {"item": "Net sales - Segment A", "amount": 1234.56},
    {"item": "Net sales - Segment B", "amount": 2345.67}
  ]
}
""".strip()

    data_rev = call_llm(revenue_system_prompt, raw_text)

    cost_system_prompt = """
You are a meticulous financial analyst.

Task: From the provided text, extract ONLY COST / EXPENSE / TAX line items
that belong to the income statement (profit and loss).

You MUST include, when present:
- Cost of sales / Cost of revenues / Cost of goods sold.
- Operating expenses such as:
    - Selling, general and administrative
    - Marketing, advertising
    - Research and development
    - Other operating expenses
- Income tax expense.
- Other clearly identified income-statement expenses.

Very important:
- Use ONLY numbers that appear in the text (no guessing).
- If multiple years are shown in columns, ALWAYS use the MOST RECENT year,
  even if that year column is not the rightmost.
- DO NOT include subtotals like "Total operating expenses",
  "Total costs and expenses", "Gross profit", "Operating income",
  "Net income", etc.
- DO NOT include revenue or net sales here.

Output ONLY valid JSON:

{
  "lines": [
    {"item": "Cost of sales", "amount": 999.99},
    {"item": "Selling, general and administrative", "amount": 888.88},
    {"item": "Income tax expense", "amount": 777.77}
  ]
}
""".strip()

    data_cost = call_llm(cost_system_prompt, raw_text)

    lines = []
    detected_company = data_rev.get("company")
    detected_currency = data_rev.get("currency")

    for src in (data_rev, data_cost):
        for line in src.get("lines", []):
            try:
                item = line["item"]
                amount = float(line["amount"])
                lines.append({"Item": item, "Amount": amount})
            except Exception:
                continue

    df = pd.DataFrame(lines)
    return df, detected_company, detected_currency


def extract_text_from_uploaded_file(uploaded_file) -> str:
    """
    Turn uploaded PDF/TXT into plain text for the LLM.

    For PDFs:
    - Read all pages
    - Keep only pages that look like income statement / revenue note
    - Then truncate to keep token usage (and cost) low.
    """
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()

    # TXT - just read everything (then truncate)
    if name.endswith(".txt") or uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        return text[:20000]  # ~5k tokens

    # PDF
    if name.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError(
                "pypdf is not installed. Add 'pypdf' to requirements.txt."
            )

        reader = PdfReader(uploaded_file)
        all_pages_text = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            all_pages_text.append(t)

        keywords = [
            "income statement",
            "statement of income",
            "statement of operations",
            "statement of earnings",
            "consolidated statements of income",
            "consolidated statement of operations",
            "profit and loss",
            "statement of profit",
            "net sales",
            "net revenue",
            "net revenues",
            "segment information",
            "disaggregation of revenue",
        ]

        candidate_indices = []
        for i, t in enumerate(all_pages_text):
            low = t.lower()
            if any(kw in low for kw in keywords):
                candidate_indices.append(i)

        expanded_indices = set()
        for i in candidate_indices:
            for j in [i - 1, i, i + 1]:
                if 0 <= j < len(all_pages_text):
                    expanded_indices.add(j)

        if expanded_indices:
            selected_pages = [all_pages_text[i] for i in sorted(expanded_indices)[:8]]
            text = "\n\n".join(selected_pages)
        else:
            text = "\n\n".join(all_pages_text)

        return text[:20000]

    # Fallback: try decoding anything else as text
    try:
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        return text[:20000]
    except Exception:
        return ""


# -------------------------------------------------------------------
# 4. Sidebar: settings and input mode
# -------------------------------------------------------------------

st.sidebar.header("Settings")

theme_name = st.sidebar.selectbox(
    "Color theme",
    list(COLOR_THEMES.keys()),
    index=0,
)
theme = COLOR_THEMES[theme_name]

min_share = st.sidebar.slider(
    "Min revenue share for separate node",
    0.0,
    0.20,
    0.05,
    0.01,
    help="Revenue items smaller than this share of total revenue "
    "are grouped into 'Other revenue'.",
)

logo_file = st.sidebar.file_uploader(
    "Company logo (PNG/JPG, optional)",
    type=["png", "jpg", "jpeg"],
)

st.sidebar.markdown("---")
st.sidebar.header("Input data")

input_mode = st.sidebar.radio(
    "How do you want to provide data?",
    ["AI (upload/paste statement)", "Upload CSV/Excel", "Use sample data"],
)

company_name_override = st.sidebar.text_input(
    "Company name (optional override)",
    value=st.session_state.detected_company,
)

# -------------------------------------------------------------------
# 5. Main title and introductory text
# -------------------------------------------------------------------

if logo_file is not None:
    st.image(logo_file, width=140)

st.title("How X Makes Money")
st.write(
    "Upload a company's financial statement and let AI extract the income statement, "
    "or upload a ready CSV/Excel. Review the line items and generate a Sankey diagram "
    "showing how the company makes and spends money."
)

# -------------------------------------------------------------------
# 6. Input modes → st.session_state.raw_df
# -------------------------------------------------------------------

raw_df = None

if input_mode == "Use sample data":
    raw_df = load_sample_df()
    st.session_state.raw_df = raw_df
    st.session_state.detected_company = company_name_override or "Example Corp"

elif input_mode == "Upload CSV/Excel":
    uploaded_csv = st.sidebar.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="File must have an 'Item' column and one or more numeric columns (years or 'Amount').",
        key="csv_uploader",
    )
    if uploaded_csv is not None:
        try:
            raw_df = pd.read_csv(uploaded_csv)
        except Exception:
            uploaded_csv.seek(0)
            raw_df = pd.read_excel(uploaded_csv)
        st.session_state.raw_df = raw_df
        st.session_state.detected_company = company_name_override or "Example Corp"

elif input_mode == "AI (upload/paste statement)":
    st.subheader("Step 0 – Provide income statement")

    uploaded_stmt = st.file_uploader(
        "Upload financial statement (PDF or TXT)",
        type=["pdf", "txt"],
        key="stmt_uploader",
    )

    raw_text_manual = st.text_area(
        "Or paste the income statement text here (income statement section is enough).",
        height=260,
        key="raw_text_area",
    )

    if st.button("Extract with AI"):
        try:
            raw_text_for_ai = extract_text_from_uploaded_file(uploaded_stmt)
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            st.stop()

        if not raw_text_for_ai.strip():
            raw_text_for_ai = raw_text_manual

        if not raw_text_for_ai.strip():
            st.warning("Please upload a PDF/TXT file or paste some text first.")
            st.stop()

        with st.spinner("Calling GPT to extract the income statement..."):
            try:
                df_ai, detected_company, detected_currency = extract_pnl_with_llm(
                    raw_text_for_ai
                )
            except Exception as e:
                st.error(
                    "AI extraction failed. Check your API key or try a simpler snippet. "
                    f"Technical detail: {e}"
                )
                st.stop()

        if df_ai.empty:
            st.error("AI did not return any line items. Try a clearer statement.")
        else:
            st.session_state.raw_df = df_ai
            if detected_company:
                st.session_state.detected_company = detected_company

            if detected_company:
                st.success(f"Detected company: {detected_company}")
            if detected_currency:
                st.caption(f"Detected currency: {detected_currency}")

# After all branches, use whatever we have in session_state
raw_df = st.session_state.raw_df

if raw_df is None:
    st.info(
        "Provide data via AI extraction, CSV upload, or the sample option in the sidebar."
    )
    st.stop()

df = raw_df.copy()

# -------------------------------------------------------------------
# 7. Column validation and auto categorization
# -------------------------------------------------------------------

cols_lower = {c.lower(): c for c in df.columns}
if "item" not in cols_lower:
    st.error(
        "Your data must contain a column with line item names called 'Item' "
        f"(case-insensitive). Found columns: {', '.join(df.columns)}"
    )
    st.stop()

item_col = cols_lower["item"]

try:
    amount_col = choose_amount_column(df.drop(columns=[item_col]))
except Exception as e:
    st.error(
        "I couldn't automatically choose which column is the most recent year / amount.\n\n"
        "Please rename your amount column to 'Amount' or keep only one numeric column.\n\n"
        f"Technical detail: {e}"
    )
    st.stop()

df = df[[item_col, amount_col]].rename(
    columns={item_col: "Item", amount_col: "Amount"}
)
df = ensure_numeric(df)

if df.empty:
    st.error("No valid numeric 'Amount' values found.")
    st.stop()

# We always infer Category locally
df["Category"] = df["Item"].apply(guess_category)

st.subheader("Step 1 – Review and adjust categories")
st.write(
    "We guessed a category for each line based on its name. "
    "You can change the Category column below. Lines marked as Ignore "
    "will not appear in the visualization."
)

edited_df = st.data_editor(
    df,
    num_rows="dynamic",
    column_config={
        "Category": st.column_config.SelectboxColumn(
            "Category",
            options=CATEGORIES,
            help="How this line item should be treated in the Sankey diagram.",
        ),
        "Amount": st.column_config.NumberColumn("Amount", format="%.2f"),
    },
    use_container_width=True,
    key="data_editor",
)

df = edited_df.copy()
df = ensure_numeric(df)
df = df[df["Category"] != "Ignore"]

if df.empty:
    st.error("All rows are marked as Ignore. Please assign some categories.")
    st.stop()

# -------------------------------------------------------------------
# 8. Aggregation and metrics
# -------------------------------------------------------------------

cat_sums = df.groupby("Category")["Amount"].sum().to_dict()

total_revenue = cat_sums.get("Revenue", 0.0)
cogs = cat_sums.get("COGS", 0.0)
rnd = cat_sums.get("R&D", 0.0)
sm = cat_sums.get("Sales & Marketing", 0.0)
ga = cat_sums.get("G&A", 0.0)
other_opex = cat_sums.get("Other Opex", 0.0)
tax = cat_sums.get("Tax", 0.0)

# If there are revenues but absolutely no costs/tax, something is off.
if total_revenue > 0 and (cogs + rnd + sm + ga + other_opex + tax) == 0:
    st.error(
        "The AI extraction found revenue but no cost or tax lines. "
        "This would make all margins 100 %, so the visualization would be misleading.\n\n"
        "Please either:\n"
        "• paste or upload a more complete income statement (including 'Cost of sales', expenses, tax), or\n"
        "• manually add the main cost lines in the table above and set their Category (e.g. COGS, G&A, Tax)."
    )
    st.stop()

gross_profit = max(total_revenue - cogs, 0)
total_opex = rnd + sm + ga + other_opex
operating_profit = max(gross_profit - total_opex, 0)
net_income = max(operating_profit - tax, 0)

company_name = (
    company_name_override or st.session_state.detected_company or "This company"
)

# -------------------------------------------------------------------
# 9. Sankey builder using color themes
# -------------------------------------------------------------------


def build_sankey(df: pd.DataFrame, theme: dict, min_share: float, company_name: str):
    # 1. Define nodes
    labels = []
    node_colors = []

    def add_node(label: str, color_hex: str):
        if label not in labels:
            labels.append(label)
            node_colors.append(lighten(color_hex, 0.35))

    # Core nodes
    add_node("Total revenue", theme["revenue_total"])
    add_node("Cost of revenues", theme["cogs"])
    add_node("Gross profit", theme["profit"])
    add_node("R&D", theme["opex_rnd"])
    add_node("Sales & marketing", theme["opex_sm"])
    add_node("G&A", theme["opex_ga"])
    add_node("Other opex", theme["opex_other"])
    add_node("Operating profit", theme["profit"])
    add_node("Tax", theme["tax"])
    add_node("Net income", theme["profit"])

    # Revenue segments
    revenue_rows = df[df["Category"] == "Revenue"].copy()
    revenue_nodes = []
    if not revenue_rows.empty:
        total_rev_seg = revenue_rows["Amount"].sum()
        threshold = total_rev_seg * min_share

        major = revenue_rows[revenue_rows["Amount"] >= threshold]
        minor = revenue_rows[revenue_rows["Amount"] < threshold]

        for _, row in major.iterrows():
            revenue_nodes.append((row["Item"], float(row["Amount"])))

        if not minor.empty:
            revenue_nodes.append(("Other revenue", float(minor["Amount"].sum())))

    for name, _ in revenue_nodes:
        add_node(name, theme["revenue_seg"])

    idx = {lab: i for i, lab in enumerate(labels)}

    # 2. Build links
    sources, targets, values, link_colors = [], [], [], []

    def add_link(src_label: str, tgt_label: str, value: float, color_hex: str):
        if value <= 0:
            return
        sources.append(idx[src_label])
        targets.append(idx[tgt_label])
        values.append(value)
        link_colors.append(hex_to_rgba(color_hex, 0.85))

    cat_sums_local = df.groupby("Category")["Amount"].sum().to_dict()
    total_revenue_l = cat_sums_local.get("Revenue", 0.0)
    cogs_l = cat_sums_local.get("COGS", 0.0)
    rnd_l = cat_sums_local.get("R&D", 0.0)
    sm_l = cat_sums_local.get("Sales & Marketing", 0.0)
    ga_l = cat_sums_local.get("G&A", 0.0)
    other_l = cat_sums_local.get("Other Opex", 0.0)
    tax_l = cat_sums_local.get("Tax", 0.0)

    gross_profit_l = max(total_revenue_l - cogs_l, 0)
    total_opex_l = rnd_l + sm_l + ga_l + other_l
    operating_profit_l = max(gross_profit_l - total_opex_l, 0)
    net_income_l = max(operating_profit_l - tax_l, 0)

    # 2.1 Revenue segments -> Total revenue
    for name, amount in revenue_nodes:
        add_link(name, "Total revenue", amount, theme["revenue_seg"])

    # 2.2 Total revenue -> COGS + Gross profit
    add_link("Total revenue", "Cost of revenues", cogs_l, theme["cogs"])
    add_link("Total revenue", "Gross profit", gross_profit_l, theme["revenue_total"])

    # 2.3 Gross profit -> Opex categories + Operating profit
    add_link("Gross profit", "R&D", rnd_l, theme["opex_rnd"])
    add_link("Gross profit", "Sales & marketing", sm_l, theme["opex_sm"])
    add_link("Gross profit", "G&A", ga_l, theme["opex_ga"])
    add_link("Gross profit", "Other opex", other_l, theme["opex_other"])
    add_link("Gross profit", "Operating profit", operating_profit_l, theme["profit"])

    # 2.4 Operating profit -> Tax + Net income
    add_link("Operating profit", "Tax", tax_l, theme["tax"])
    add_link("Operating profit", "Net income", net_income_l, theme["profit"])

    # 3. Build figure
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=20,
                    thickness=22,
                    line=dict(color="rgba(0,0,0,0.15)", width=0.4),
                    label=labels,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"How {company_name} Makes Money",
        font=dict(size=13, color=theme["text"]),
        paper_bgcolor=theme["background"],
        plot_bgcolor=theme["background"],
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return fig


# -------------------------------------------------------------------
# 10. Show metrics + Sankey
# -------------------------------------------------------------------

st.subheader("Step 2 – Key figures")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total revenue", f"{total_revenue:,.0f}")
with col2:
    gross_margin = (gross_profit / total_revenue * 100) if total_revenue else 0
    st.metric("Gross margin", f"{gross_margin:.1f} %")
with col3:
    op_margin = (operating_profit / total_revenue * 100) if total_revenue else 0
    st.metric("Operating margin", f"{op_margin:.1f} %")
with col4:
    net_margin = (net_income / total_revenue * 100) if total_revenue else 0
    st.metric("Net margin", f"{net_margin:.1f} %")

st.subheader("Step 3 – Visualization")
st.write(
    "Click **Generate chart** after you are happy with the categories above. "
    "Hover over the flows to see exact amounts."
)

if st.button("Generate chart"):
    fig = build_sankey(df, theme, min_share, company_name)
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "Tip: use the camera icon in the top right of the chart to download it as a PNG."
    )
else:
    st.caption("Press the button above to build the Sankey diagram.")

