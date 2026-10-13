"""
System Q Command-Line Interface (CLI)

Usage Examples:
    # 1. Quick Scam Detection
    python -m src.cli "Urgent: Your PayPal account has been suspended. Click here to verify." --task scam

    # 2. Email Intent Classification
    python -m src.cli "Hi, we are looking to purchase 100 enterprise seats for our engineering team." --task intent

    # 3. Custom Decision
    python -m src.cli "User has filed 3 chargebacks in 24 hours." --options "FLAG,APPROVE,BLOCK"

    # 4. Interactive Mode
    python -m src.cli --interactive
"""

import argparse
import sys
import os

from src.system_q import SystemQ, SCAM_OPTIONS, INTENT_OPTIONS


def print_banner():
    banner = r"""
  ___           _                   ___  
 / __|_  _  ___| |_ ___ _ __  ___  / _ \ 
 \__ \ || |(_-<|  _/ -_) '  \(_-< | (_) |
 |___/\_, |/__/ \__\___|_|_|_/__/  \__\_\
      |__/                               
    Fast, Accurate System 1 Decision Engine
"""
    print(banner)


def format_decision_card(text: str, decision, task_label: str = "Decision"):
    print("\n" + "=" * 65)
    print(f"  SYSTEM Q DECISION CARD [{task_label.upper()}]")
    print("=" * 65)
    preview = text.strip().replace("\n", " ")
    if len(preview) > 90:
        preview = preview[:87] + "..."
    print(f"  Input:       \"{preview}\"")
    print(f"  Decision:    [{decision.choice}]")
    print(f"  Confidence:  {decision.confidence:.1%} (Anchor: {decision.anchor}, Margin: {decision.margin:.3f})")
    print(f"  Latency:     {decision.latency_ms:.1f} ms (0 tokens generated)")
    print("-" * 65)
    print("  Probability Distribution:")
    for opt, prob in decision.scores.items():
        bar_len = int(prob * 25)
        bar = "#" * bar_len + "-" * (25 - bar_len)
        marker = " <--" if opt == decision.choice else ""
        print(f"    {opt:<22} [{bar}] {prob:>6.1%}{marker}")
    print("=" * 65 + "\n")


def run_interactive(q: SystemQ):
    print("\nEntering Interactive Decision Mode.")
    print("Tasks: [1] Scam Detection  [2] Email Intent  [3] Custom Options  [q] Quit")
    
    while True:
        try:
            mode = input("\nSelect task (1=Scam, 2=Intent, 3=Custom, q=Quit): ").strip()
            if mode.lower() in ["q", "quit", "exit"]:
                print("Exiting System Q. Goodbye!")
                break

            if mode == "1":
                task_label = "Scam Detection"
                options = SCAM_OPTIONS
            elif mode == "2":
                task_label = "Email Intent"
                options = INTENT_OPTIONS
            elif mode == "3":
                custom = input("Enter comma-separated options (e.g. YES,NO,MAYBE): ").strip()
                options = [o.strip() for o in custom.split(",") if o.strip()]
                if len(options) < 2:
                    print("Error: Need at least 2 options.")
                    continue
                task_label = "Custom Decision"
            else:
                print("Invalid selection.")
                continue

            text = input("Paste or type input text: ").strip()
            if not text:
                print("Empty input skipped.")
                continue

            decision = q.decide(text, options)
            format_decision_card(text, decision, task_label)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting System Q.")
            break


def main():
    parser = argparse.ArgumentParser(description="System Q: Fast System 1 Decision CLI")
    parser.add_argument("input_text", nargs="?", default=None, help="Input text or email body to evaluate")
    parser.add_argument("--task", choices=["scam", "intent"], default=None, help="Preset decision task")
    parser.add_argument("--options", type=str, default=None, help="Comma-separated options (e.g. YES,NO,REVIEW)")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive decision shell")
    parser.add_argument("--precision", choices=["4bit", "8bit", "16bit"], default="4bit", help="Model precision")

    args = parser.parse_args()
    print_banner()

    q = SystemQ(precision=args.precision)

    if args.interactive or (args.input_text is None and args.options is None and args.task is None):
        run_interactive(q)
        return

    text = args.input_text
    if not text:
        print("Error: Please provide input text to analyze.")
        sys.exit(1)

    if args.task == "scam":
        task_label = "Scam Detection"
        decision = q.is_scam(text)
    elif args.task == "intent":
        task_label = "Email Intent"
        decision = q.get_intent(text)
    elif args.options:
        task_label = "Custom Decision"
        opts = [o.strip() for o in args.options.split(",") if o.strip()]
        decision = q.decide(text, opts)
    else:
        # Default to scam check if not specified
        task_label = "Scam Detection"
        decision = q.is_scam(text)

    format_decision_card(text, decision, task_label)


if __name__ == "__main__":
    main()
