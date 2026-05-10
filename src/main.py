import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import gradio as gr
import pandas as pd
import numpy as np
import pyarrow
import plotly.express as px
import tensorflow as tf
import joblib
import warnings

warnings.filterwarnings("ignore")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, "data", "Churn_Data_Final.parquet")
model_path = os.path.join(BASE_DIR, "models", "churn_champion_model.h5")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

data = pd.read_parquet(data_path)
model = tf.keras.models.load_model(model_path, compile=False)
scaler = joblib.load(scaler_path)


FEATURE_COLUMNS = [
    "Tenure",
    "Total Spend",
    "Usage Frequency",
    "Payment Delay",
    "Support Calls",
    "Last Interaction",
]


def filter_data(generation, subscription, contract, segment):
    df = data.copy()

    if generation != "All":
        df = df[df["Generations"].astype(str) == generation]

    if subscription != "All":
        df = df[df["Subscription Type"].astype(str) == subscription]

    if contract != "All":
        df = df[df["Contract Length"].astype(str) == contract]

    if segment != "All":
        df = df[df["Segment_Labels"].astype(str) == segment]

    return df


def style_plot(fig):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig

def format_currency(value):
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.2f}K"
    else:
        return f"${value:.2f}"
def create_charts(df):
    churn_segment_fig = px.histogram(
        df,
        x="Segment_Labels",
        color="Churn_status",
        barmode="group",
        title="Churn by Customer Segment",
        labels={
            "Segment_Labels": "Customer Segment",
            "Churn_status": "Churn Status",
        },
    )
    churn_segment_fig = style_plot(churn_segment_fig)

    payment_delay_fig = px.box(
        df,
        x="Churn_status",
        y="Payment Delay",
        title="Payment Delay vs Churn",
        labels={
            "Churn_status": "Churn Status",
            "Payment Delay": "Payment Delay",
        },
    )
    payment_delay_fig = style_plot(payment_delay_fig)

    tenure_fig = px.box(
        df,
        x="Churn_status",
        y="Tenure",
        title="Tenure vs Churn",
        labels={
            "Churn_status": "Churn Status",
            "Tenure": "Tenure",
        },
    )
    tenure_fig = style_plot(tenure_fig)

    usage_fig = px.box(
        df,
        x="Churn_status",
        y="Usage Frequency",
        title="Usage Frequency vs Churn",
        labels={
            "Churn_status": "Churn Status",
            "Usage Frequency": "Usage Frequency",
        },
    )
    usage_fig = style_plot(usage_fig)

    support_fig = px.box(
        df,
        x="Churn_status",
        y="Support Calls",
        title="Support Calls vs Churn",
        labels={
            "Churn_status": "Churn Status",
            "Support Calls": "Support Calls",
        },
    )
    support_fig = style_plot(support_fig)

    spend_fig = px.box(
        df,
        x="Churn_status",
        y="Total Spend",
        title="Total Spend vs Churn",
        labels={
            "Churn_status": "Churn Status",
            "Total Spend": "Total Spend",
        },
    )
    spend_fig = style_plot(spend_fig)

    return churn_segment_fig, payment_delay_fig, tenure_fig, usage_fig, support_fig, spend_fig


def generate_insights(df):
    if len(df) == 0:
        return "### Key Insights\nNo data available for the selected filters."

    churn_rate = (
        df["Churn_status"]
        .value_counts(normalize=True)
        .get("Churned", 0) * 100
    )

    avg_payment_delay = df["Payment Delay"].mean()
    avg_tenure = df["Tenure"].mean()
    avg_spend = df["Total Spend"].mean()
    avg_support = round(df["Support Calls"].mean(),0)
    total_revenue_loss = df[df["Churn_status"]=="Churned"]["Customer_Lifetime_value"].sum()
    avg_cltv = df["Customer_Lifetime_value"].mean()
    avg_revenue_loss = total_revenue_loss/len(df)


    return f"""
### Key Business Insights

- Current filtered customer base has a churn rate of **{churn_rate:.2f}%**.
- Average payment delay is **{avg_payment_delay:.2f}**, indicating billing-related churn risk.
- Average tenure is **{avg_tenure:.2f}** months, helping identify long-term customer retention patterns.
- Average support calls is **{avg_support:.2f}**, suggesting that frequent support interactions may contribute to customer dissatisfaction.
- Average customer spend is **${avg_spend:.2f}**, which helps estimate customer revenue contribution.
- Estimated average Customer Lifetime Value (CLTV) is **{format_currency(avg_cltv)}**, representing the projected long-term value generated per customer.
- Total estimated revenue loss from churned customers is **{format_currency(total_revenue_loss)}***, with an average revenue loss of **{format_currency(avg_revenue_loss)}** per churned customer.
"""


def update_dashboard(generation, subscription, contract, segment):
    df = filter_data(generation, subscription, contract, segment)

    length_of_customer = len(df)

    churn_rate = (
        df["Churn_status"]
        .value_counts(normalize=True)
        .get("Churned", 0) * 100
        if len(df) > 0
        else 0
    )

    avg_tenure = df["Tenure"].mean() if len(df) > 0 else 0
    avg_contract = round(df["Numeric Contract Length"].mean(), 0) if len(df) > 0 else 0
    avg_spend = df["Total Spend"].mean() if len(df) > 0 else 0

    (
        churn_segment_fig,
        payment_delay_fig,
        tenure_fig,
        usage_fig,
        support_fig,
        spend_fig,
    ) = create_charts(df)

    insights = generate_insights(df)

    return (
        length_of_customer,
        f"{churn_rate:.2f}%",
        f"{avg_tenure:.2f}",
        avg_contract,
        f"${avg_spend:.2f}",
        churn_segment_fig,
        payment_delay_fig,
        tenure_fig,
        usage_fig,
        support_fig,
        spend_fig,
        insights,
    )


def predict_churn(
    tenure,
    total_spend,
    usage_frequency,
    payment_delay,
    support_calls,
    last_interaction,
):
    input_df = pd.DataFrame([{
        "Tenure": tenure,
        "Total Spend": total_spend,
        "Usage Frequency": usage_frequency,
        "Payment Delay": payment_delay,
        "Support Calls": support_calls,
        "Last Interaction": last_interaction,
    }])

    input_df = input_df[FEATURE_COLUMNS]
    scaled_input = scaler.transform(input_df)

    probability = model.predict(scaled_input, verbose=0)[0][0]
    prediction = "Churned" if probability >= 0.5 else "Not Churned"

    return prediction, f"{probability * 100:.2f}%"


css = """
    .metric-box label {
        text-align: center !important;
        font-weight: bold !important;
        font-size:12px !important;
    }
    .metric-box input {
        text-align: center !important;
        font-size: 24px !important;
        font-weight: bold !important;
        color: #1f77b4 !important;
    }
"""


def create_interface():
    with gr.Blocks(title="Churn Predictor & Segmentation App", css=css) as demo:
        gr.Markdown("# Telecom Churn Prediction & Dashboard")
        gr.Markdown("""
This tool combines exploratory analytics and machine learning to understand and predict telecom customer churn. 
Use the dashboard to analyze churn patterns by customer generation, subscription type, contract length, and customer segment.
        """)

        with gr.Tabs():

            with gr.Tab("Analytics Dashboard"):
                with gr.Row():
                    gen = gr.Dropdown(
                        ["All", "Baby Boomers", "Generation X", "Millennials", "Generation Z"],
                        label="Generation",
                        value="All",
                    )

                    sub = gr.Dropdown(
                        ["All", "Basic", "Standard", "Premium"],
                        label="Subscription Type",
                        value="All",
                    )

                    con = gr.Dropdown(
                        ["All"] + sorted(data["Contract Length"].dropna().astype(str).unique().tolist()),
                        label="Contract Length",
                        value="All",
                    )

                    seg = gr.Dropdown(
                        ["All"] + sorted(data["Segment_Labels"].dropna().astype(str).unique().tolist()),
                        label="Segment",
                        value="All",
                    )

                gr.Markdown("## Overview KPIs")

                with gr.Row():
                    customer_count = gr.Textbox(
                        label="Number of Customers",
                        interactive=False,
                        elem_classes=["metric-box"],
                    )
                    churn_rate_display = gr.Textbox(
                        label="Churn Rate",
                        interactive=False,
                        elem_classes=["metric-box"],
                    )
                    tenure_display = gr.Textbox(
                        label="Average Tenure",
                        interactive=False,
                        elem_classes=["metric-box"],
                    )
                    contract_display = gr.Textbox(
                        label="Average Number of Contracts",
                        interactive=False,
                        elem_classes=["metric-box"],
                    )
                    spend_display = gr.Textbox(
                        label="Average Spend",
                        interactive=False,
                        elem_classes=["metric-box"],
                    )

                gr.Markdown("## Who is Churning?")

                with gr.Row():
                    churn_segment_plot = gr.Plot()

                gr.Markdown("## Why Are Customers Churning?")

                with gr.Row():
                    payment_delay_plot = gr.Plot()
                    tenure_plot = gr.Plot()

                with gr.Row():
                    usage_plot = gr.Plot()
                    support_plot = gr.Plot()

                gr.Markdown("## Revenue Risk")

                with gr.Row():
                    spend_plot = gr.Plot()

                insight_box = gr.Markdown()

                inputs = [gen, sub, con, seg]

                outputs = [
                    customer_count,
                    churn_rate_display,
                    tenure_display,
                    contract_display,
                    spend_display,
                    churn_segment_plot,
                    payment_delay_plot,
                    tenure_plot,
                    usage_plot,
                    support_plot,
                    spend_plot,
                    insight_box,
                ]

                for dropdown in [gen, sub, con, seg]:
                    dropdown.change(update_dashboard, inputs=inputs, outputs=outputs)

                demo.load(update_dashboard, inputs=inputs, outputs=outputs)

            with gr.Tab("Churn Prediction"):
                gr.Markdown("## Customer Churn Prediction App")

                with gr.Row():
                    tenure = gr.Number(label="Tenure")
                    total_spend = gr.Number(label="Total Spend $")

                with gr.Row():
                    usage_frequency = gr.Number(label="Usage Frequency")
                    payment_delay = gr.Number(label="Payment Delay")

                with gr.Row():
                    support_calls = gr.Number(label="Support Calls")
                    last_interaction = gr.Number(label="Last Interaction")

                predict_btn = gr.Button("Predict Churn")

                with gr.Row():
                    prediction_output = gr.Textbox(
                        label="Prediction",
                        interactive=False,
                        elem_classes=["metric-box"],
                    )
                    probability_output = gr.Textbox(
                        label="Churn Probability",
                        interactive=False,
                        elem_classes=["metric-box"],
                    )

                predict_btn.click(
                    predict_churn,
                    inputs=[
                        tenure,
                        total_spend,
                        usage_frequency,
                        payment_delay,
                        support_calls,
                        last_interaction,
                    ],
                    outputs=[prediction_output, probability_output],
                )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 8080)),
        debug=True,
        show_error=True
    )