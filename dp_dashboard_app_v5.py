
import hashlib
import hmac
import json
import math
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

DATA_PATH = "retail_offer_events_demo.csv"
APP_SECRET = os.getenv("DP_APP_SECRET", "CHANGE_ME_DEMO_SECRET")
NOISE_VERSION = "v1"

st.set_page_config(page_title="DP Dashboard Teaching Demo", layout="wide")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["event_timestamp", "event_date"])
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    return df


def get_counts(
    df: pd.DataFrame,
    start_date: date,
    end_date: date,
    region: str,
    city: str,
    age_band: str,
    customer_type: str,
    campaign_id: str,
    offer_type: str,
    campaign_channel: str,
    event_type: str,
) -> pd.DataFrame:
    x = df
    mask = (x["event_date"] >= start_date) & (x["event_date"] <= end_date)
    if region != "All":
        mask &= x["region"] == region
    if city != "All":
        mask &= x["city"] == city
    if age_band != "All":
        mask &= x["age_band"] == age_band
    if customer_type != "All":
        mask &= x["customer_type"] == customer_type
    if campaign_id != "All":
        mask &= x["campaign_id"] == campaign_id
    if offer_type != "All":
        mask &= x["offer_type"] == offer_type
    if campaign_channel != "All":
        mask &= x["campaign_channel"] == campaign_channel
    if event_type != "All":
        mask &= x["event_type"] == event_type
    return x.loc[mask].copy()


def canonical_query_key(filters: dict, start_date: date, end_date: date) -> str:
    payload = {
        "version": NOISE_VERSION,
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "filters": {k: filters[k] for k in sorted(filters)},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def deterministic_uniform(key: str) -> float:
    digest = hmac.new(APP_SECRET.encode(), key.encode(), hashlib.sha256).digest()
    n = int.from_bytes(digest[:8], "big")
    u = (n + 0.5) / (2**64)
    return min(max(u, 1e-12), 1 - 1e-12)


def laplace_inverse_cdf(u: float, b: float) -> float:
    if u < 0.5:
        return b * math.log(2 * u)
    return -b * math.log(2 * (1 - u))


def fresh_noisy_count(true_count: int, epsilon: float) -> int:
    b = 1.0 / epsilon
    return max(0, int(round(true_count + np.random.laplace(0, b))))


def deterministic_noisy_count(true_count: int, epsilon: float, key: str) -> int:
    b = 1.0 / epsilon
    u = deterministic_uniform(key)
    noise = laplace_inverse_cdf(u, b)
    return max(0, int(round(true_count + noise)))


def all_filter_options(df: pd.DataFrame):
    return {
        "region": ["All"] + sorted(df["region"].dropna().unique().tolist()),
        "city": ["All"] + sorted(df["city"].dropna().unique().tolist()),
        "age_band": ["All"] + sorted(df["age_band"].dropna().unique().tolist()),
        "customer_type": ["All"] + sorted(df["customer_type"].dropna().unique().tolist()),
        "campaign_id": ["All"] + sorted(df["campaign_id"].dropna().unique().tolist()),
        "offer_type": ["All"] + sorted(df["offer_type"].dropna().unique().tolist()),
        "campaign_channel": ["All"] + sorted(df["campaign_channel"].dropna().unique().tolist()),
        "event_type": ["All"] + sorted(df["event_type"].dropna().unique().tolist()),
    }


def filters_ui(df: pd.DataFrame, prefix: str):
    opts = all_filter_options(df)
    cols = st.columns(4)
    values = {}
    keys = ["region", "city", "age_band", "customer_type", "campaign_id", "offer_type", "campaign_channel", "event_type"]
    labels = {
        "region": "Region",
        "city": "City",
        "age_band": "Age band",
        "customer_type": "Customer type",
        "campaign_id": "Campaign",
        "offer_type": "Offer type",
        "campaign_channel": "Channel",
        "event_type": "Event type",
    }
    for i, k in enumerate(keys):
        with cols[i % 4]:
            values[k] = st.selectbox(labels[k], opts[k], key=f"{prefix}_{k}")
    return values


def true_count_for(df, filters, start_date, end_date):
    return len(get_counts(df, start_date, end_date, **filters))


def split_range_ui(label_prefix: str):
    split = st.checkbox("Split this time range into two sub-ranges", key=f"{label_prefix}_split")
    if not split:
        return split, None

    start = st.session_state[f"{label_prefix}_start"]
    end = st.session_state[f"{label_prefix}_end"]
    if start >= end:
        st.warning("Need at least 2 days in selected range to split.")
        return split, None

    midpoint = start + (end - start) / 2
    split_point = st.date_input(
        "Split point (left range ends here)",
        value=midpoint,
        min_value=start,
        max_value=end - timedelta(days=1),
        key=f"{label_prefix}_split_point",
    )
    left = (start, split_point)
    right = (split_point + timedelta(days=1), end)
    st.caption(f"Sub-ranges: {left[0]} to {left[1]}, and {right[0]} to {right[1]}")
    return split, [left, right]


def add_record(state_key: str, row: dict):
    if state_key not in st.session_state:
        st.session_state[state_key] = []
    st.session_state[state_key].append(row)


def records_df(state_key: str):
    vals = st.session_state.get(state_key, [])
    return pd.DataFrame(vals) if vals else pd.DataFrame()


def summarize_average(df_rec: pd.DataFrame, value_col: str):
    if df_rec.empty:
        return None
    return {
        "n": len(df_rec),
        "avg": float(df_rec[value_col].mean()),
        "min": float(df_rec[value_col].min()),
        "max": float(df_rec[value_col].max()),
    }


def daterange(start_date, end_date):
    d = start_date
    while d <= end_date:
        yield d
        d += timedelta(days=1)


# ===== Correct atomic logic: noise is attached ONLY to fixed daily buckets =====
def atomic_day_key(filters: dict, day: date) -> str:
    payload = {
        "version": NOISE_VERSION,
        "mode": "atomic_day",
        "day": day.isoformat(),
        "filters": {k: filters[k] for k in sorted(filters)},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def true_count_for_day(df, filters, day):
    return true_count_for(df, filters, day, day)


def noisy_atomic_day_count(df, filters, day, epsilon):
    tc = true_count_for_day(df, filters, day)
    key = atomic_day_key(filters, day)
    noisy = deterministic_noisy_count(tc, epsilon, key)
    return tc, noisy, key


def atomic_day_breakdown(df, filters, start_date, end_date, epsilon):
    rows = []
    for day in daterange(start_date, end_date):
        tc, noisy, key = noisy_atomic_day_count(df, filters, day, epsilon)
        rows.append(
            {
                "day": day,
                "true_count": tc,
                "noisy_count": noisy,
                "atomic_key": key,
            }
        )
    return pd.DataFrame(rows)


def noisy_count_atomic(df, filters, start_date, end_date, epsilon):
    detail = atomic_day_breakdown(df, filters, start_date, end_date, epsilon)
    return int(detail["noisy_count"].sum()), detail


df = load_data(DATA_PATH)
min_date = df["event_date"].min()
max_date = df["event_date"].max()

epsilon = st.sidebar.slider("ε (epsilon)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
st.sidebar.markdown("Smaller ε = more noise, stronger privacy.")
st.sidebar.caption("Demo note: set DP_APP_SECRET in your environment for a real secret.")

st.title("Differential Privacy Dashboard Teaching Demo")
st.write("Compare fresh noise, stable deterministic noise, and atomic time buckets.")

tabs = st.tabs(
    [
        "1. True count",
        "2. Fresh noise + averaging attack",
        "3. Stable deterministic noise",
        "4. Stable noise without atomic buckets",
        "5. Stable noise with atomic buckets",
    ]
)

# Tab 1 (unchanged)
with tabs[0]:
    st.subheader("Tab 1: True count")
    c1, c2 = st.columns([1, 1])
    with c1:
        start_date = st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date, key="t1_start")
    with c2:
        end_date = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date, key="t1_end")
    filters = filters_ui(df, "t1")
    if start_date > end_date:
        st.error("Start date must be <= end date.")
    else:
        tc = true_count_for(df, filters, start_date, end_date)
        st.metric("True count", tc)

# Tab 2 (unchanged)
with tabs[1]:
    st.subheader("Tab 2: Fresh Laplace noise (changes on every draw)")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date, key="t2_start")
    with c2:
        st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date, key="t2_end")
    filters = filters_ui(df, "t2")
    start_date = st.session_state["t2_start"]
    end_date = st.session_state["t2_end"]
    if start_date <= end_date:
        tc = true_count_for(df, filters, start_date, end_date)
        st.metric("True count", tc)
        noisy = fresh_noisy_count(tc, epsilon)
        st.metric("Fresh noisy answer", noisy)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Record this noisy answer", key="t2_record"):
                add_record("t2_records", {"true_count": tc, "noisy_answer": noisy})
        with col_b:
            if st.button("Clear recorded answers", key="t2_clear"):
                st.session_state["t2_records"] = []
        rdf = records_df("t2_records")
        if not rdf.empty:
            st.dataframe(rdf, use_container_width=True)
            s = summarize_average(rdf, "noisy_answer")
            st.write(f"Averaging attack estimate after **{s['n']}** recorded answers: **{s['avg']:.2f}**")
            st.write(f"True count = **{tc}**; error of average = **{s['avg'] - tc:.2f}**")

# Tab 3 (unchanged)
with tabs[2]:
    st.subheader("Tab 3: Stable deterministic noise")
    st.caption("Refresh or rerun the app with the same filters and dates: the answer stays the same.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date, key="t3_start")
    with c2:
        st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date, key="t3_end")
    filters = filters_ui(df, "t3")
    start_date = st.session_state["t3_start"]
    end_date = st.session_state["t3_end"]
    if start_date <= end_date:
        tc = true_count_for(df, filters, start_date, end_date)
        key = canonical_query_key(filters, start_date, end_date)
        noisy = deterministic_noisy_count(tc, epsilon, key)
        st.metric("True count", tc)
        st.metric("Stable deterministic noisy answer", noisy)
        st.code(key, language="json")
        st.caption("In production, only the canonical query is visible; the secret key stays server-side.")
        if st.button("Record this stable answer", key="t3_record"):
            add_record("t3_records", {"true_count": tc, "stable_noisy_answer": noisy, "query_key": key})
        if st.button("Clear stable records", key="t3_clear"):
            st.session_state["t3_records"] = []
        rdf = records_df("t3_records")
        if not rdf.empty:
            st.dataframe(rdf, use_container_width=True)
            s = summarize_average(rdf, "stable_noisy_answer")
            st.write(f"Average of recorded stable answers: **{s['avg']:.2f}**")
            st.write("Notice this does not improve with repeated recordings if the query is unchanged.")

# Tab 4 (unchanged)
with tabs[3]:
    st.subheader("Tab 4: Stable deterministic noise without atomic buckets")
    st.caption("Each full date range gets its own deterministic noise. If you ask different sub-ranges, they get different deterministic noises.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date, key="t4_start")
    with c2:
        st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date, key="t4_end")
    filters = filters_ui(df, "t4")
    start_date = st.session_state["t4_start"]
    end_date = st.session_state["t4_end"]
    if start_date <= end_date:
        true_total = true_count_for(df, filters, start_date, end_date)
        direct_key = canonical_query_key(filters, start_date, end_date)
        direct_noisy = deterministic_noisy_count(true_total, epsilon, direct_key)
        st.metric("True count", true_total)
        st.metric("Direct stable noisy answer", direct_noisy)
        split, ranges = split_range_ui("t4")
        if split and ranges:
            parts = []
            sum_noisy = 0
            for idx, (a, b) in enumerate(ranges, start=1):
                t = true_count_for(df, filters, a, b)
                k = canonical_query_key(filters, a, b)
                n = deterministic_noisy_count(t, epsilon, k)
                parts.append({"part": idx, "start": a, "end": b, "true_count": t, "noisy_count": n})
                sum_noisy += n
            part_df = pd.DataFrame(parts)
            st.dataframe(part_df, use_container_width=True)
            st.write(f"Sum of split noisy answers = **{sum_noisy}**")
            st.write(f"Direct noisy answer = **{direct_noisy}**")
            st.write(f"True total = **{true_total}**")
            st.info(
                "These can differ because the full range and each sub-range are treated as different deterministic queries, "
                "so they get different deterministic noise values."
            )
            if st.button("Record this split attack attempt", key="t4_record"):
                add_record(
                    "t4_records",
                    {
                        "true_total": true_total,
                        "direct_noisy": direct_noisy,
                        "split_sum_noisy": sum_noisy,
                        "difference_from_true": sum_noisy - true_total,
                    },
                )
            if st.button("Clear split records", key="t4_clear"):
                st.session_state["t4_records"] = []
            rdf = records_df("t4_records")
            if not rdf.empty:
                st.dataframe(rdf, use_container_width=True)
                s = summarize_average(rdf, "split_sum_noisy")
                st.write(f"Average reconstructed count from recorded split attempts = **{s['avg']:.2f}**")
                st.write(f"Average error vs true count = **{s['avg'] - true_total:.2f}**")

# Tab 5 (fixed atomic logic)
with tabs[4]:
    st.subheader("Tab 5: Stable deterministic noise with atomic daily buckets")
    st.caption(
        "Here noise is attached only to fixed daily buckets. Any larger range is answered by summing the same noisy daily buckets."
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date, key="t5_start")
    with c2:
        st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date, key="t5_end")
    filters = filters_ui(df, "t5")
    start_date = st.session_state["t5_start"]
    end_date = st.session_state["t5_end"]
    if start_date <= end_date:
        true_total = true_count_for(df, filters, start_date, end_date)
        direct_atomic_noisy, direct_details = noisy_count_atomic(df, filters, start_date, end_date, epsilon)

        st.metric("True count", true_total)
        st.metric("Direct atomic noisy answer", direct_atomic_noisy)

        with st.expander("Show daily atomic blocks used for direct range"):
            st.dataframe(direct_details, use_container_width=True)

        split, ranges = split_range_ui("t5")
        if split and ranges:
            left_noisy, left_details = noisy_count_atomic(df, filters, ranges[0][0], ranges[0][1], epsilon)
            right_noisy, right_details = noisy_count_atomic(df, filters, ranges[1][0], ranges[1][1], epsilon)
            split_total = left_noisy + right_noisy

            st.write(f"Sum of split atomic noisy answers = **{split_total}**")
            st.write(f"Direct atomic noisy answer = **{direct_atomic_noisy}**")
            st.write(f"True total = **{true_total}**")

            with st.expander("Show daily atomic blocks used for split ranges"):
                left_show = left_details.copy()
                left_show["part"] = 1
                right_show = right_details.copy()
                right_show["part"] = 2
                st.dataframe(pd.concat([left_show, right_show], ignore_index=True), use_container_width=True)

            st.success(
                "These should match exactly (or very closely if you later change post-processing), because both calculations reuse the same noisy daily building blocks."
            )
            st.write(f"Difference (direct atomic - split atomic) = **{direct_atomic_noisy - split_total}**")

            if st.button("Record this atomic split attempt", key="t5_record"):
                add_record(
                    "t5_records",
                    {
                        "true_total": true_total,
                        "direct_atomic_noisy": direct_atomic_noisy,
                        "split_atomic_sum": split_total,
                        "difference_from_true": split_total - true_total,
                        "direct_minus_split": direct_atomic_noisy - split_total,
                    },
                )
            if st.button("Clear atomic records", key="t5_clear"):
                st.session_state["t5_records"] = []
            rdf = records_df("t5_records")
            if not rdf.empty:
                st.dataframe(rdf, use_container_width=True)
                s = summarize_average(rdf, "split_atomic_sum")
                st.write(f"Average split reconstruction = **{s['avg']:.2f}**")
                st.write(f"Average error vs true count = **{s['avg'] - true_total:.2f}**")
                st.write(f"Average (direct atomic - split atomic) = **{rdf['direct_minus_split'].mean():.2f}**")
