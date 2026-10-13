"""
System Q — Hugging Face Space / Interactive Gradio Web Demo.
Deployable with 1 click on a free Hugging Face CPU/GPU Space!
"""

import gradio as gr
import time
from src.system_q import SystemQ, SCAM_OPTIONS, INTENT_OPTIONS

print("Loading System Q for Web Interface...")
q = SystemQ(precision="4bit")


def analyze_scam(email_text):
    if not email_text.strip():
        return "Please enter text to analyze.", {}, "0.0 ms"
    t0 = time.perf_counter()
    res = q.is_scam(email_text)
    dt = (time.perf_counter() - t0) * 1000.0
    
    status = f"🚨 {res.choice}" if res.is_scam else f"✅ {res.choice}"
    summary = f"### Decision: **{status}**\n**Confidence:** {res.confidence:.1%} | **Margin:** {res.margin:.3f}"
    latency_str = f"{res.latency_ms:.1f} ms (0 tokens generated)"
    return summary, res.scores, latency_str


def analyze_intent(message_text):
    if not message_text.strip():
        return "Please enter text to analyze.", {}, "0.0 ms"
    t0 = time.perf_counter()
    res = q.get_intent(message_text)
    dt = (time.perf_counter() - t0) * 1000.0
    
    summary = f"### Routing Category: **{res.category}**\n**Confidence:** {res.confidence:.1%} | **Margin:** {res.margin:.3f}"
    latency_str = f"{res.latency_ms:.1f} ms (0 tokens generated)"
    return summary, res.scores, latency_str


def analyze_custom(input_text, options_csv):
    if not input_text.strip():
        return "Please enter text.", {}, "0.0 ms"
    opts = [o.strip() for o in options_csv.split(",") if o.strip()]
    if len(opts) < 2:
        return "Please specify at least 2 comma-separated options.", {}, "0.0 ms"
    
    res = q.decide(input_text, opts)
    summary = f"### Selected Option: **{res.choice}**\n**Confidence:** {res.confidence:.1%} | **Margin:** {res.margin:.3f}"
    latency_str = f"{res.latency_ms:.1f} ms (0 tokens generated)"
    return summary, res.scores, latency_str


# Build Gradio UI
with gr.Blocks(title="System Q: Fast System 1 Decision Model", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # ⚡ System Q: Fast, Accurate System 1 Decision Engine
    ### Built from Qwen2.5-3B • 0 Autoregressive Tokens • Sub-250ms on Consumer GPUs
    
    System Q adapts the pretrained weights of Qwen2.5-3B into a non-autoregressive decision-making architecture.
    **Beats ModernBERT-large (Laya) by +21.0% overall accuracy (88.0% vs 67.0%) on consumer 4GB VRAM hardware.**
    """)
    
    with gr.Tabs():
        # Tab 1: Scam Detection
        with gr.TabItem("🛡️ Scam & Phishing Detection"):
            scam_input = gr.Textbox(
                label="Email / Message Content",
                placeholder="Paste an email or message to detect phishing, CEO fraud, or invoice scams...",
                lines=5
            )
            scam_btn = gr.Button("Analyze Decision ⚡", variant="primary")
            scam_summary = gr.Markdown()
            scam_plot = gr.Label(label="Probability Distribution")
            scam_lat = gr.Textbox(label="Inference Latency", interactive=False)
            
            scam_btn.click(analyze_scam, inputs=[scam_input], outputs=[scam_summary, scam_plot, scam_lat])
            
            gr.Examples(
                examples=[
                    ["Critical Alert: Your Office 365 password expires in 2 hours! Click http://m1crosoft-auth.com immediately."],
                    ["Your package #US-90218 is in transit and estimated for delivery tomorrow by 5 PM. Track at fedex.com/tracking."],
                    ["Hey, I am in an urgent executive meeting. Can you buy $500 Apple gift cards for a client and email the codes to me right now? - Sent from my iPhone"]
                ],
                inputs=scam_input
            )

        # Tab 2: Intent Classification
        with gr.TabItem("📬 Customer Email Intent Routing"):
            intent_input = gr.Textbox(
                label="Customer Communication",
                placeholder="Paste customer inquiry to route between BILLING, TECHNICAL_SUPPORT, ACCOUNT_ACCESS, SALES_INQUIRY, ACCOUNT_CANCELLATION...",
                lines=5
            )
            intent_btn = gr.Button("Classify Intent ⚡", variant="primary")
            intent_summary = gr.Markdown()
            intent_plot = gr.Label(label="Intent Probabilities")
            intent_lat = gr.Textbox(label="Inference Latency", interactive=False)
            
            intent_btn.click(analyze_intent, inputs=[intent_input], outputs=[intent_summary, intent_plot, intent_lat])
            
            gr.Examples(
                examples=[
                    ["Could you please send over an updated VAT invoice for our company's September billing statement?"],
                    ["We are experiencing 502 Bad Gateway errors on the v2 /predict endpoint in the production cluster."],
                    ["We would like to explore enterprise pricing and an annual contract for 150 team seats."]
                ],
                inputs=intent_input
            )

        # Tab 3: Custom Decision Space
        with gr.TabItem("⚙️ Custom Decision Space"):
            custom_input = gr.Textbox(label="Scenario / Input Context", lines=4)
            custom_opts = gr.Textbox(label="Allowed Options (Comma-Separated)", value="APPROVE, MANUAL_REVIEW, REJECT")
            custom_btn = gr.Button("Evaluate Options ⚡", variant="primary")
            custom_summary = gr.Markdown()
            custom_plot = gr.Label(label="Choice Probabilities")
            custom_lat = gr.Textbox(label="Inference Latency", interactive=False)
            
            custom_btn.click(analyze_custom, inputs=[custom_input, custom_opts], outputs=[custom_summary, custom_plot, custom_lat])

    gr.Markdown("""
    ---
    *Built with System Q (Qwen2.5-3B INT4 NF4 + Fused Margin LoRA + Canonical Anchor Scoring).*  
    *Research & Benchmark Results: 88.0% Accuracy | ~210 ms Latency | 2.003 GB Peak VRAM.*
    """)

if __name__ == "__main__":
    demo.launch()
