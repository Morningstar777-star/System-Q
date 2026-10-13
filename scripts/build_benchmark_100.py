"""
Dataset generator for 100-question benchmark:
- 50 Scam Mail Detection questions (SCAM vs LEGITIMATE)
- 50 Email Intent Classification questions (BILLING, TECHNICAL_SUPPORT, ACCOUNT_ACCESS, SALES_INQUIRY, ACCOUNT_CANCELLATION)
"""

import json
import os

BENCHMARK_100 = [
    # =========================================================================
    # PART 1: SCAM MAIL DETECTION (50 Questions)
    # Options: ["SCAM", "LEGITIMATE"]
    # =========================================================================
    # --- Phishing & Credential Theft ---
    {
        "id": "SCAM_001",
        "domain": "scam_detection",
        "input": "From: security@m1crosoft-auth.com\nSubject: Critical Alert: Your Office 365 password expires in 2 hours!\nPlease click here immediately to verify your credentials or your email mailbox will be permanently deactivated.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "phishing_credential_harvesting"
    },
    {
        "id": "SCAM_002",
        "domain": "scam_detection",
        "input": "From: account-security-noreply@accountprotection.microsoft.com\nSubject: Microsoft account security code\nYour single-use security code is 849201. If you did not request this code, you can safely ignore this email.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_2fa"
    },
    {
        "id": "SCAM_003",
        "domain": "scam_detection",
        "input": "From: service@paypa1-resolution-center.net\nSubject: Account Suspended: Unusual activity detected\nWe noticed an unauthorized charge of $849.00 on your account. Download the attached PDF form and enter your SSN and credit card number to dispute the transaction.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "phishing_fake_charge"
    },
    {
        "id": "SCAM_004",
        "domain": "scam_detection",
        "input": "From: service@paypal.com\nSubject: Receipt for your payment to Steam Games\nYou sent a payment of $19.99 USD to Valve Corporation. Transaction ID: 4920183021. You can review your transaction history in your PayPal account dashboard.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_receipt"
    },
    {
        "id": "SCAM_005",
        "domain": "scam_detection",
        "input": "From: ceo.office@executive-urgent-board.com\nSubject: Confidential wire transfer request\nHi, I am currently in an all-day board meeting. I need you to wire $48,500 immediately to this overseas vendor account for an urgent contract closing. Do not call me, just reply with confirmation.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "ceo_fraud_bec"
    },
    {
        "id": "SCAM_006",
        "domain": "scam_detection",
        "input": "From: sarah.jenkins@company.com\nSubject: Slides for Thursday client sync\nHi team, attached is the updated deck for Thursday's review meeting. Let me know if you have feedback on the Q3 roadmap slides before noon tomorrow.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_internal_work"
    },
    {
        "id": "SCAM_007",
        "domain": "scam_detection",
        "input": "From: payroll@internal-hr-portal-update.org\nSubject: ACTION REQUIRED: Update direct deposit account\nOur HR payroll software is migrating to a new banking system. Please login via http://bit.ly/3xHR-payroll-update and input your routing number and online banking login.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "payroll_fraud"
    },
    {
        "id": "SCAM_008",
        "domain": "scam_detection",
        "input": "From: notifications@github.com\nSubject: [GitHub] A personal access token has expired\nYour personal access token 'ci-deploy-token' (ID: 98124) expired on October 12, 2026. Visit github.com/settings/tokens to generate a replacement token.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_dev_notification"
    },
    {
        "id": "SCAM_009",
        "domain": "scam_detection",
        "input": "From: support@coinbase-wallets-kyc.info\nSubject: Your crypto wallet has been frozen!\nDue to AML regulations, you must send 0.05 BTC to the verification address 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa within 24 hours to unlock your funds.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "crypto_scam"
    },
    {
        "id": "SCAM_010",
        "domain": "scam_detection",
        "input": "From: auto-confirm@amazon.com\nSubject: Your Amazon.com order #112-948192-3810283\nThank you for shopping with us! We will send a confirmation when your items ship. Order total: $34.50. View or manage your order in Your Orders.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_order"
    },
    {
        "id": "SCAM_011",
        "domain": "scam_detection",
        "input": "From: tax-refund@irs-federal-gov-online.cc\nSubject: Internal Revenue Service: Unclaimed tax refund of $3,420.00\nYou have an outstanding federal tax refund waiting. Submit your debit card PIN and mother's maiden name at the link below to receive direct deposit within 1 hour.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "tax_scam"
    },
    {
        "id": "SCAM_012",
        "domain": "scam_detection",
        "input": "From: noreply@stripe.com\nSubject: Your monthly Stripe billing summary for September\nYour invoice summary is ready. Total processing fees charged: $142.18. You can download the complete breakdown PDF directly from your Stripe Dashboard.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_billing_statement"
    },
    {
        "id": "SCAM_013",
        "domain": "scam_detection",
        "input": "From: geeksquad-billing@invoicing-quickbooks-cloud.com\nSubject: INVOICE #83921: GeekSquad 3-Year Protection Plan auto-renewed ($499.99)\nIf you did not authorize this charge, call our toll-free fraud cancellation hotline at 1-800-555-0199 immediately. Do not hang up.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "call_center_invoice_scam"
    },
    {
        "id": "SCAM_014",
        "domain": "scam_detection",
        "input": "From: trackingupdates@fedex.com\nSubject: FedEx Shipment 789421890312: Delivered\nYour package was delivered on Monday at 2:15 PM at your front door. Signed for by: Signature not required. Track further shipments at fedex.com.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_delivery"
    },
    {
        "id": "SCAM_015",
        "domain": "scam_detection",
        "input": "From: usps-reschedule@redelivery-parcel-tracking.vip\nSubject: USPS Delivery Failed: Incorrect address details\nPackage #US-98214 could not be delivered due to missing house number. Pay $2.99 re-delivery fee using your credit card at the link below.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "smishing_phishing_parcel"
    },
    {
        "id": "SCAM_016",
        "domain": "scam_detection",
        "input": "From: alerts@chase.com\nSubject: Security alert: Card not present transaction of $42.50\nA transaction of $42.50 at Trader Joe's was approved on card ending in 4102. If you recognize this charge, no action is needed.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_bank_alert"
    },
    {
        "id": "SCAM_017",
        "domain": "scam_detection",
        "input": "From: barrister.adebayo@lagos-chamber-legal.com\nSubject: Confidential: Inheritance funds of $14.5 Million USD\nI am the personal attorney to a deceased client sharing your last name. We can transfer 40% of the funds to your account upon receiving a $500 clearance certificate fee.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "advance_fee_419"
    },
    {
        "id": "SCAM_018",
        "domain": "scam_detection",
        "input": "From: digest-noreply@quora.com\nSubject: Top questions for you: What are the best practices for CUDA optimization?\nCheck out answers from senior engineers discussing memory coalescing and warp scheduling on modern GPUs.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_newsletter"
    },
    {
        "id": "SCAM_019",
        "domain": "scam_detection",
        "input": "From: docusign@document-signature-secure-review.ru\nSubject: Document Ready: You have 1 document awaiting signature\nPlease review and execute the attached NDASigning.exe file to finalize your real estate disclosure.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "malware_exe_disguised"
    },
    {
        "id": "SCAM_020",
        "domain": "scam_detection",
        "input": "From: dse_na4@docusign.net\nSubject: Completed: Mutual Non-Disclosure Agreement between Acme and Beta\nAll parties have completed signing the Mutual Non-Disclosure Agreement. A copy of the finalized document is available for download in your DocuSign account.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_docusign"
    },
    {
        "id": "SCAM_021",
        "domain": "scam_detection",
        "input": "From: apple-id@verify-icloud-device-locked.biz\nSubject: Your Apple ID was used to sign in to an unauthorized device in Moscow\nYour iCloud account has been locked for safety. Click here within 10 minutes and enter your password and passcode to avoid permanent account termination.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "fake_device_alert"
    },
    {
        "id": "SCAM_022",
        "domain": "scam_detection",
        "input": "From: no_reply@email.apple.com\nSubject: Your receipt from Apple for iCloud+ 200GB Storage\nThank you for your purchase. Subscribed service: iCloud+ with 200 GB Storage ($2.99/month). Payment method: Apple Pay ending in 1920.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_apple_receipt"
    },
    {
        "id": "SCAM_023",
        "domain": "scam_detection",
        "input": "From: hr-lottery-rewards@employee-survey-bonus.top\nSubject: Congratulations! You won the $1,000 corporate anniversary gift card!\nYou have been selected in our annual workplace raffle. Click here and input your bank account details to claim your cash voucher immediately.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "fake_lottery_raffle"
    },
    {
        "id": "SCAM_024",
        "domain": "scam_detection",
        "input": "From: notifications@slack.com\nSubject: Marcus invited you to the workspace 'Project Pegasus'\nMarcus Vance (marcus@pegasus-ai.com) invited you to collaborate in Slack. Click below to accept the invitation and join your team.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_slack_invite"
    },
    {
        "id": "SCAM_025",
        "domain": "scam_detection",
        "input": "From: blackhat-hacker99@anonymous-onion-mail.org\nSubject: I have recorded your screen and webcam. Pay 0.1 Bitcoin or I leak it\nI installed a pegasus trojan on your system while you were browsing adult content. You have 48 hours to transfer $1,000 in BTC to my wallet or I send the video to all your contacts.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "sextortion_blackmail"
    },
    {
        "id": "SCAM_026",
        "domain": "scam_detection",
        "input": "From: alerts@pagerduty.com\nSubject: [Triggered] Incident #49281: High Latency on API Gateway\nSeverity: P2. Service: api-gateway-prod. Details: 99th percentile response time exceeded 1200ms over 5 minutes. Acknowledged by on-call engineer.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_pagerduty"
    },
    {
        "id": "SCAM_027",
        "domain": "scam_detection",
        "input": "From: billing@zoom-meeting-renewal-auth.cc\nSubject: Zoom Pro Monthly Invoice #Z-92819 - Failed Payment\nYour credit card payment failed. Download the attached invoice.js file and run it to update your payment details.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "malware_script_attachment"
    },
    {
        "id": "SCAM_028",
        "domain": "scam_detection",
        "input": "From: no-reply@zoom.us\nSubject: Cloud Recording - Product Architecture Sync is now available\nYour cloud recording of 'Product Architecture Sync' on October 12, 2026 is ready to view. Passcode: 849#19a. You can manage recordings in your Zoom dashboard.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_zoom_recording"
    },
    {
        "id": "SCAM_029",
        "domain": "scam_detection",
        "input": "From: job-offer@amazon-remote-careers-recruit.xyz\nSubject: Work From Home Job Offer: $85/hour simple data entry!\nYou have been selected for an online remote position. Send your $150 onboarding background check processing fee via Venmo to get started immediately.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "employment_scam"
    },
    {
        "id": "SCAM_030",
        "domain": "scam_detection",
        "input": "From: recruiting@google.com\nSubject: Interview Confirmation: Software Engineering, Core ML Infrastructure\nHi Yash, thank you for confirming your availability. Your technical interview is scheduled for Tuesday, October 20 at 3:00 PM IST via Google Meet.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_recruiter"
    },
    {
        "id": "SCAM_031",
        "domain": "scam_detection",
        "input": "From: netflix-payment@stream-account-billing-fix.top\nSubject: We could not process your Netflix membership fee\nYour membership is on hold. Update your card number, CVV, and billing address at http://netflix-secure-login-39.biz to continue watching.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "phishing_streaming"
    },
    {
        "id": "SCAM_032",
        "domain": "scam_detection",
        "input": "From: info@mailer.netflix.com\nSubject: Coming this Friday: New releases on Netflix\nCheck out the newest additions to your watchlist including sci-fi thrillers, documentary series, and award-winning dramas.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_marketing"
    },
    {
        "id": "SCAM_033",
        "domain": "scam_detection",
        "input": "From: sweepstakes@international-powerball-claims.com\nSubject: NOTICE: Winner of 500,000 GBP in British National Lottery\nYour email address was drawn by computer ballot. Send copy of your passport and Western Union transfer fee of 250 GBP to release your winning cheque.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "lottery_scam"
    },
    {
        "id": "SCAM_034",
        "domain": "scam_detection",
        "input": "From: support@uber.com\nSubject: Your Tuesday morning trip with Uber\nThanks for riding with us, Yash. Total: ₹340.00. Pickup: Koramangala. Dropoff: Indiranagar. A detailed PDF tax invoice is attached.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_ride_receipt"
    },
    {
        "id": "SCAM_035",
        "domain": "scam_detection",
        "input": "From: bank-support@wells-fargo-card-verify.cc\nSubject: Card Suspended: Immediate identity verification required\nYour Wells Fargo debit card has been restricted due to suspicious POS transactions. Call 1-888-028-1920 or reply with your 16-digit card number and PIN.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "banking_phishing"
    },
    {
        "id": "SCAM_036",
        "domain": "scam_detection",
        "input": "From: statements@fidelity.com\nSubject: Your Fidelity brokerage account statement is now available online\nYour monthly statement for period ending September 30, 2026 is ready to review. Log in to your Fidelity account at fidelity.com to view your portfolio summary.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_financial_statement"
    },
    {
        "id": "SCAM_037",
        "domain": "scam_detection",
        "input": "From: customer-care@mcafee-antivirus-support-online.net\nSubject: Auto-Renewal Notice: McAfee Total Protection billed $699.00\nThank you for renewing your antivirus. The charge will appear on your bank statement within 24 hours. If this was not you, call 1-844-392-1029 to cancel immediately.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "refund_scam_call_bait"
    },
    {
        "id": "SCAM_038",
        "domain": "scam_detection",
        "input": "From: order-update@doordash.com\nSubject: Order confirmation from Chipotle Mexican Grill\nYour order has been received and the restaurant is preparing your food. Estimated arrival: 12:45 PM. You can track your Dasher live in the DoorDash app.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_food_order"
    },
    {
        "id": "SCAM_039",
        "domain": "scam_detection",
        "input": "From: admin@customs-border-protection-impound.com\nSubject: Legal Warning: Package seized by Customs containing illegal substances\nWe intercepted a parcel under your name. Pay $800 clearance bond in Google Play gift cards to prevent an arrest warrant from being issued by local police.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "law_enforcement_impersonation"
    },
    {
        "id": "SCAM_040",
        "domain": "scam_detection",
        "input": "From: calendar-notification@google.com\nSubject: Reminder: Weekly ML Engineering Sync @ Wed Oct 14 10am - 11am (IST)\nYou have an upcoming calendar event. Google Meet link: meet.google.com/abc-defg-hij. Organized by dev-leads@company.com.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_calendar"
    },
    {
        "id": "SCAM_041",
        "domain": "scam_detection",
        "input": "From: support@steam-community-free-wallet-codes.org\nSubject: Claim free $100 Steam Wallet card!\nSteam is celebrating its summer event. Sign in with your Steam username and password at our promotional portal to get your redeemable code.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "gaming_phishing"
    },
    {
        "id": "SCAM_042",
        "domain": "scam_detection",
        "input": "From: noreply@medium.com\nSubject: Top stories for you on Machine Learning and PyTorch\nDiscover the top 5 trending articles on implementing custom PyTorch kernels, Triton compilation, and LLM inference optimization.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_newsletter"
    },
    {
        "id": "SCAM_043",
        "domain": "scam_detection",
        "input": "From: secure@docusign-contracts-sign-now.click\nSubject: URGENT: Sign real estate closing papers\nClick the link below and upload photos of your passport and banking login to unlock the deed transfer agreement.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "fake_docusign"
    },
    {
        "id": "SCAM_044",
        "domain": "scam_detection",
        "input": "From: support@digitalocean.com\nSubject: Maintenance Notification: NYC1 datacenter network maintenance\nOur engineering team will perform scheduled maintenance on network switches in NYC1 on Sunday, Oct 18 between 02:00 and 04:00 UTC. No droplet downtime is expected.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_cloud_maintenance"
    },
    {
        "id": "SCAM_045",
        "domain": "scam_detection",
        "input": "From: bill-pay@electricity-power-shutoff-notice.info\nSubject: FINAL NOTICE: Power disconnect in 30 minutes due to unpaid bill\nYour electric utility service will be cut off today. Pay the past-due balance of $184.20 immediately using Bitcoin ATM at your nearest convenience store.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "utility_shutoff_scam"
    },
    {
        "id": "SCAM_046",
        "domain": "scam_detection",
        "input": "From: no-reply@spotify.com\nSubject: Your Spotify Premium receipt for October\nWe have received your payment of $10.99 for Spotify Premium Individual. Next billing date: November 12, 2026. Manage your plan in Account Overview.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_subscription_receipt"
    },
    {
        "id": "SCAM_047",
        "domain": "scam_detection",
        "input": "From: hr-review@performance-salary-bonus-appraisal.top\nSubject: Confidential: Your 2026 Salary Raise Schedule\nPlease review the attached macro-enabled Excel document 'Bonus_Calculations_2026.xlsm' to approve your year-end performance raise.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "macro_malware_excel"
    },
    {
        "id": "SCAM_048",
        "domain": "scam_detection",
        "input": "From: news@ycombinator.com\nSubject: Hacker News: Daily Digest\nStories with the highest points today: Fast inference architectures for edge transformers, new compiler optimizations in LLVM 21, and database indexing strategies.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_digest"
    },
    {
        "id": "SCAM_049",
        "domain": "scam_detection",
        "input": "From: binance-rewards@airdrop-crypto-claim-web3.cc\nSubject: You have 5,000 USDT waiting to be claimed in the Web3 Airdrop!\nConnect your MetaMask or TrustWallet private key at our decentralized claim portal to receive your token allocation before the window closes tonight.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "SCAM",
        "type": "crypto_airdrop_seed_phrase"
    },
    {
        "id": "SCAM_050",
        "domain": "scam_detection",
        "input": "From: noreply@github.com\nSubject: [GitHub] Security advisory: New vulnerability reported in dependency\nA high severity vulnerability (CVE-2026-4921) was detected in repository 'system-q-core' dependency 'urllib3'. Dependabot has created a pull request to update to 2.6.4.",
        "options": ["SCAM", "LEGITIMATE"],
        "ground_truth": "LEGITIMATE",
        "type": "legitimate_dependabot"
    },

    # =========================================================================
    # PART 2: EMAIL INTENT CLASSIFICATION (50 Questions)
    # Options: ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"]
    # =========================================================================
    # --- Category: BILLING (10 questions) ---
    {
        "id": "INTENT_001",
        "domain": "intent_classification",
        "input": "Subject: Double charge on invoice #9201\nHi, I checked my bank statement today and saw that my credit card was billed twice ($49.00 each) for my October subscription. Could you please refund the duplicate charge?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "BILLING",
        "type": "intent_billing"
    },
    {
        "id": "INTENT_002",
        "domain": "intent_classification",
        "input": "Subject: Updating company VAT number\nCan you please re-issue invoice #INV-4928 with our updated corporate VAT ID 'DE982140281' so our accounting team can claim tax deductions?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "BILLING",
        "type": "intent_billing"
    },
    {
        "id": "INTENT_003",
        "domain": "intent_classification",
        "input": "Subject: Need W-9 form for vendor onboarding\nOur accounts payable department requires a signed W-9 form and ACH payment instructions before we can remit payment for invoice #8392.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "BILLING",
        "type": "intent_billing"
    },
    {
        "id": "INTENT_004",
        "domain": "intent_classification",
        "input": "Subject: Overcharge on metered API tokens\nWe noticed a spike in API billing charges this month. According to our internal telemetry we consumed 12M tokens, but the invoice billed us for 24M tokens. Please review usage logs.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "BILLING",
        "type": "intent_billing"
    },
    {
        "id": "INTENT_005",
        "domain": "intent_classification",
        "input": "Subject: Change credit card on file\nOur corporate Amex card expired at the end of September. How can our billing administrator update the payment method without disrupting our active services?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "BILLING",
        "type": "intent_billing"
    },
    {
        "id": "INTENT_006",
        "domain": "intent_classification",
        "input": "Subject: Annual prepayment discount inquiry\nWe currently pay monthly for 25 seats. Do you offer a discount if we switch to an annual upfront invoice rather than monthly card billing?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "BILLING",
        "type": "intent_billing"
    },
    {
        "id": "INTENT_007",
        "domain": "intent_classification",
        "input": "Subject: Refund request for unused days\nI upgraded our plan yesterday by mistake and immediately reverted back to the starter tier within 10 minutes. Can you issue a prorated refund for the difference?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "BILLING",
        "type": "intent_billing"
    },
    {
        "id": "INTENT_008",
        "domain": "intent_classification",
        "input": "Subject: Currency conversion surcharge\nWhy was our euro transaction charged an additional 3.5% foreign exchange fee on our receipt? Can we set our billing currency directly to EUR?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "BILLING",
        "type": "intent_billing"
    },
    {
        "id": "INTENT_009",
        "domain": "intent_classification",
        "input": "Subject: Past due invoice notice - payment already sent\nWe received an automated email stating our invoice #9012 is past due, but our bank wire cleared 3 business days ago. Here is the wire transfer reference code #WT-8942.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "BILLING",
        "type": "intent_billing"
    },
    {
        "id": "INTENT_010",
        "domain": "intent_classification",
        "input": "Subject: Where can I download all past PDF invoices?\nOur finance auditors need copies of all monthly invoices from January to December 2025. Is there a way to bulk export all receipt PDFs from the portal?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "BILLING",
        "type": "intent_billing"
    },

    # --- Category: TECHNICAL_SUPPORT (10 questions) ---
    {
        "id": "INTENT_011",
        "domain": "intent_classification",
        "input": "Subject: REST API returning HTTP 502 Bad Gateway\nSince 09:30 UTC, approximately 30% of our POST requests to /v1/chat/completions are failing with 502 Bad Gateway from Cloudflare. Are your servers experiencing degradation?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "TECHNICAL_SUPPORT",
        "type": "intent_tech_support"
    },
    {
        "id": "INTENT_012",
        "domain": "intent_classification",
        "input": "Subject: CUDA out of memory error when loading model weights\nWhen calling torch.load on our RTX 3050 GPU, the process crashes with 'torch.cuda.OutOfMemoryError: CUDA out of memory'. How can we enable 4-bit NF4 quantization?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "TECHNICAL_SUPPORT",
        "type": "intent_tech_support"
    },
    {
        "id": "INTENT_013",
        "domain": "intent_classification",
        "input": "Subject: Webhook signature verification failing in Python SDK\nWe implemented HMAC-SHA256 signature verification as described in your docs, but every incoming webhook event throws an invalid signature exception. Here is our snippet.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "TECHNICAL_SUPPORT",
        "type": "intent_tech_support"
    },
    {
        "id": "INTENT_014",
        "domain": "intent_classification",
        "input": "Subject: Database migration script stuck on index creation\nRunning 'python manage.py migrate' hangs indefinitely on migration 0048_add_gin_index. There is an exclusive table lock blocking all client queries. Please advise.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "TECHNICAL_SUPPORT",
        "type": "intent_tech_support"
    },
    {
        "id": "INTENT_015",
        "domain": "intent_classification",
        "input": "Subject: Docker container keeps restarting with exit code 137\nOur worker pod on Kubernetes is crashing every 10 minutes with exit code 137 (OOMKilled). Could you provide guidance on optimal memory limits?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "TECHNICAL_SUPPORT",
        "type": "intent_tech_support"
    },
    {
        "id": "INTENT_016",
        "domain": "intent_classification",
        "input": "Subject: WebSocket connection dropping after 60 seconds\nOur frontend client connects fine, but exactly 60 seconds into every session the socket closes with code 1006. Is there a keepalive heartbeat ping we must configure?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "TECHNICAL_SUPPORT",
        "type": "intent_tech_support"
    },
    {
        "id": "INTENT_017",
        "domain": "intent_classification",
        "input": "Subject: SSL handshake error when connecting from Node.js 22\nOur microservice throws 'UNABLE_TO_VERIFY_LEAF_SIGNATURE' when calling your staging endpoints. Did your team recently rotate intermediate CA certificates?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "TECHNICAL_SUPPORT",
        "type": "intent_tech_support"
    },
    {
        "id": "INTENT_018",
        "domain": "intent_classification",
        "input": "Subject: Tokenizer mismatch on special tokens in Hugging Face\nWhen tokenizing inputs using AutoTokenizer, the EOS token ID is encoded as 151645 instead of 151643, causing generation stopping criteria to fail. How do we patch this?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "TECHNICAL_SUPPORT",
        "type": "intent_tech_support"
    },
    {
        "id": "INTENT_019",
        "domain": "intent_classification",
        "input": "Subject: Latency spike on vector search queries\nOur similarity search queries usually complete in 15ms, but over the last 2 hours response times jumped to 850ms. Can your infrastructure team check the cluster shard health?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "TECHNICAL_SUPPORT",
        "type": "intent_tech_support"
    },
    {
        "id": "INTENT_020",
        "domain": "intent_classification",
        "input": "Subject: iOS app crashing on launch after updating to v3.2.0\nSeveral users report that after installing yesterday's App Store update, the application crashes immediately on the splash screen on iOS 18. Crash symbolication attached.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "TECHNICAL_SUPPORT",
        "type": "intent_tech_support"
    },

    # --- Category: ACCOUNT_ACCESS (10 questions) ---
    {
        "id": "INTENT_021",
        "domain": "intent_classification",
        "input": "Subject: Locked out of account - lost 2FA phone\nI recently lost my phone and do not have access to my Google Authenticator app or backup recovery codes. How can I verify my identity to reset two-factor authentication?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_ACCESS",
        "type": "intent_account_access"
    },
    {
        "id": "INTENT_022",
        "domain": "intent_classification",
        "input": "Subject: Password reset link not arriving in email\nI clicked 'Forgot Password' three times over the last hour, but no reset email is arriving in my inbox or spam folder. My login email is alex@startup.io.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_ACCESS",
        "type": "intent_account_access"
    },
    {
        "id": "INTENT_023",
        "domain": "intent_classification",
        "input": "Subject: Transferring team ownership to new administrator\nOur current workspace owner has left the company. As the HR director, I need to transfer primary ownership of our enterprise account to our new VP of Engineering.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_ACCESS",
        "type": "intent_account_access"
    },
    {
        "id": "INTENT_024",
        "domain": "intent_classification",
        "input": "Subject: Account suspended after too many failed login attempts\nMy account was locked after I entered an old password several times. It says 'Contact Support to unlock your account'. Could an admin please unlock my profile?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_ACCESS",
        "type": "intent_account_access"
    },
    {
        "id": "INTENT_025",
        "domain": "intent_classification",
        "input": "Subject: Changing primary email address on my profile\nOur company recently rebranded and my email changed from john@oldcorp.com to john@newbrand.com. How do I update my primary login email without losing past projects?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_ACCESS",
        "type": "intent_account_access"
    },
    {
        "id": "INTENT_026",
        "domain": "intent_classification",
        "input": "Subject: SAML SSO login loop with Okta\nWhen our employees attempt to log in using Okta Single Sign-On, the browser redirects back to the login page without signing in. Can you check our IdP metadata XML configuration?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_ACCESS",
        "type": "intent_account_access"
    },
    {
        "id": "INTENT_027",
        "domain": "intent_classification",
        "input": "Subject: How to enable role-based permissions for contractors?\nWe want to invite 3 external design contractors into our workspace but restrict them from viewing billing settings or accessing administrative API keys. How do we configure roles?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_ACCESS",
        "type": "intent_account_access"
    },
    {
        "id": "INTENT_028",
        "domain": "intent_classification",
        "input": "Subject: Suspicious login from unexpected country alert\nI received an email stating my account was accessed from an IP in Brazil. I live in Seattle and never traveled there. Please revoke all active sessions immediately.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_ACCESS",
        "type": "intent_account_access"
    },
    {
        "id": "INTENT_029",
        "domain": "intent_classification",
        "input": "Subject: Resending team invite link to colleague\nI invited my colleague emily@team.com yesterday but the invitation token expired before she could open it. How do I resend the activation link from the admin console?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_ACCESS",
        "type": "intent_account_access"
    },
    {
        "id": "INTENT_030",
        "domain": "intent_classification",
        "input": "Subject: Deactivating former employee account\nAn engineer resigned yesterday and we need to immediately revoke their access tokens, OAuth sessions, and workspace seat to maintain SOC2 compliance.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_ACCESS",
        "type": "intent_account_access"
    },

    # --- Category: SALES_INQUIRY (10 questions) ---
    {
        "id": "INTENT_031",
        "domain": "intent_classification",
        "input": "Subject: Enterprise pricing for 500+ developers\nHello, our organization is evaluating your platform for our global software engineering division (~500 engineers). Can we speak with an account executive regarding volume pricing tiers?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "SALES_INQUIRY",
        "type": "intent_sales"
    },
    {
        "id": "INTENT_032",
        "domain": "intent_classification",
        "input": "Subject: Scheduling a product demo for leadership team\nWe would like to book a 45-minute technical demonstration for our VP of Product and CTO next Tuesday to assess your enterprise analytics capabilities.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "SALES_INQUIRY",
        "type": "intent_sales"
    },
    {
        "id": "INTENT_033",
        "domain": "intent_classification",
        "input": "Subject: Custom SLA and dedicated support contract\nDoes your enterprise plan include a 99.99% uptime guarantee with a 15-minute response time SLA and a dedicated customer success manager?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "SALES_INQUIRY",
        "type": "intent_sales"
    },
    {
        "id": "INTENT_034",
        "domain": "intent_classification",
        "input": "Subject: Reseller and partnership program inquiry\nWe are an IT systems integrator in Germany servicing Fortune 500 clients. We would like to explore joining your authorized partner program to resell your software.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "SALES_INQUIRY",
        "type": "intent_sales"
    },
    {
        "id": "INTENT_035",
        "domain": "intent_classification",
        "input": "Subject: RFP submission for government healthcare portal\nWe are issuing an official Request For Proposal (RFP) for a secure decision intelligence platform. Please review the attached requirements matrix and provide commercial terms.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "SALES_INQUIRY",
        "type": "intent_sales"
    },
    {
        "id": "INTENT_036",
        "domain": "intent_classification",
        "input": "Subject: On-premise air-gapped deployment option\nOur defense compliance rules prohibit cloud hosting. Do you offer an on-premise, air-gapped deployment license for your System 1 decision engine?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "SALES_INQUIRY",
        "type": "intent_sales"
    },
    {
        "id": "INTENT_037",
        "domain": "intent_classification",
        "input": "Subject: Academic discount for research lab\nWe are a research group at Stanford University studying non-autoregressive transformers. Do you provide educational or academic research grants for API credits?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "SALES_INQUIRY",
        "type": "intent_sales"
    },
    {
        "id": "INTENT_038",
        "domain": "intent_classification",
        "input": "Subject: Competitor comparison sheet vs Datadog\nOur procurement committee is deciding between your monitoring suite and Datadog. Could your sales engineering team share a comparative feature and cost matrix?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "SALES_INQUIRY",
        "type": "intent_sales"
    },
    {
        "id": "INTENT_039",
        "domain": "intent_classification",
        "input": "Subject: Proof of Concept (POC) trial extension request\nOur team has been evaluating your platform on a 14-day trial. We need another 2 weeks to complete load testing with our security architects before signing the annual agreement.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "SALES_INQUIRY",
        "type": "intent_sales"
    },
    {
        "id": "INTENT_040",
        "domain": "intent_classification",
        "input": "Subject: Startup program application\nWe recently closed a $2M seed round from Y Combinator. Are we eligible for your startup acceleration credits package ($100k API credits)?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "SALES_INQUIRY",
        "type": "intent_sales"
    },

    # --- Category: ACCOUNT_CANCELLATION (10 questions) ---
    {
        "id": "INTENT_041",
        "domain": "intent_classification",
        "input": "Subject: Cancel our Pro subscription immediately\nWe no longer require your service as our project has concluded. Please terminate our subscription effective immediately and do not renew our credit card next month.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_CANCELLATION",
        "type": "intent_cancellation"
    },
    {
        "id": "INTENT_042",
        "domain": "intent_classification",
        "input": "Subject: How to delete my personal account and all associated data?\nUnder GDPR right to be forgotten, I request the complete deletion of my account, profile information, and all stored telemetry from your database.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_CANCELLATION",
        "type": "intent_cancellation"
    },
    {
        "id": "INTENT_043",
        "domain": "intent_classification",
        "input": "Subject: Discontinue auto-renew on annual agreement\nOur contract renews on November 15. Please consider this formal written notice that we elect not to renew for the upcoming year and wish to terminate service at contract end.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_CANCELLATION",
        "type": "intent_cancellation"
    },
    {
        "id": "INTENT_044",
        "domain": "intent_classification",
        "input": "Subject: Switching to an in-house solution - closing workspace\nOur engineering division has built an internal tool that replaces this product. What are the steps to export our raw logs and dismantle our organization workspace?",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_CANCELLATION",
        "type": "intent_cancellation"
    },
    {
        "id": "INTENT_045",
        "domain": "intent_classification",
        "input": "Subject: Too expensive for our budget - need to cancel\nDue to company-wide budget reductions, we cannot justify the monthly cost. Please downgrade us to the free community tier and cancel our paid plan.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_CANCELLATION",
        "type": "intent_cancellation"
    },
    {
        "id": "INTENT_046",
        "domain": "intent_classification",
        "input": "Subject: Terminate trial before billing starts\nI signed up for the 7-day free trial yesterday but realized this doesn't fit our tech stack. Please cancel the trial immediately so my card is not charged when the trial ends.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_CANCELLATION",
        "type": "intent_cancellation"
    },
    {
        "id": "INTENT_047",
        "domain": "intent_classification",
        "input": "Subject: Shutting down startup operations\nUnfortunately our company is winding down business operations at the end of the month. We need to close our account and remove our payment methods permanently.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_CANCELLATION",
        "type": "intent_cancellation"
    },
    {
        "id": "INTENT_048",
        "domain": "intent_classification",
        "input": "Subject: Cancel membership - poor customer service experience\nAfter waiting 5 days for a response from your support team, I have decided to migrate our workflows to a competitor. Please cancel our account today.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_CANCELLATION",
        "type": "intent_cancellation"
    },
    {
        "id": "INTENT_049",
        "domain": "intent_classification",
        "input": "Subject: Revoke all licenses and end agreement\nPlease accept this notification to terminate our service subscription as of the end of the current billing cycle. Confirm once cancellation is confirmed in your system.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_CANCELLATION",
        "type": "intent_cancellation"
    },
    {
        "id": "INTENT_050",
        "domain": "intent_classification",
        "input": "Subject: Unsubscribe from all services and purge account\nI no longer use this software. Please close my account, cancel any recurring billing authorizations, and remove my contact information from your systems.",
        "options": ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS", "SALES_INQUIRY", "ACCOUNT_CANCELLATION"],
        "ground_truth": "ACCOUNT_CANCELLATION",
        "type": "intent_cancellation"
    }
]

def build_benchmark_100(output_file: str = "data/benchmark_100.json"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(BENCHMARK_100, f, indent=2)
    print(f"Generated 100-question benchmark dataset at {output_file}")
    scam_count = sum(1 for q in BENCHMARK_100 if q['domain'] == 'scam_detection')
    intent_count = sum(1 for q in BENCHMARK_100 if q['domain'] == 'intent_classification')
    print(f"Total Questions: {len(BENCHMARK_100)} (Scam Detection: {scam_count}, Intent Classification: {intent_count})")
    return BENCHMARK_100

if __name__ == "__main__":
    build_benchmark_100()
