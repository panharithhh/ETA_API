import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ── Configure these in .env ──────────────────────────────────────────────────
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")       # your Gmail address
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")  # Gmail app password (not your account password)

_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{ margin: 0; padding: 0; background: #f4f4f5; font-family: Arial, sans-serif; }}
    .wrap {{ max-width: 600px; margin: 40px auto; background: #ffffff;
             border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
    .header {{ background: #1e293b; padding: 28px 32px; }}
    .header h1 {{ margin: 0; color: #ffffff; font-size: 20px; }}
    .body {{ padding: 32px; }}
    .body p {{ color: #374151; line-height: 1.7; margin: 0 0 16px; }}
    .tracking {{ background: #f8fafc; border-left: 4px solid #6366f1;
                 padding: 12px 16px; border-radius: 4px; margin: 20px 0;
                 font-size: 14px; color: #475569; }}
    .actions {{ text-align: center; margin: 32px 0; }}
    .btn {{ display: inline-block; padding: 14px 32px; border-radius: 8px;
            text-decoration: none; font-weight: bold; font-size: 15px; margin: 8px; }}
    .btn-confirm {{ background: #16a34a; color: #ffffff; }}
    .btn-reject  {{ background: #dc2626; color: #ffffff; }}
    .notice {{ font-size: 13px; color: #9ca3af; margin-top: 8px; }}
    .footer {{ background: #f8fafc; padding: 20px 32px;
               font-size: 12px; color: #9ca3af; text-align: center; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1>Chonchoun Courier</h1>
    </div>
    <div class="body">
      <p>Hi there,</p>
      <p>Your package has arrived at our warehouse and is awaiting your confirmation before we route it for delivery.</p>
      <div class="tracking">
        <strong>Package ID:</strong> {package_id}
      </div>
      <p>Please click one of the buttons below. These links are single-use and expire in <strong>48 hours</strong>.</p>
      <div class="actions">
        <a href="{confirm_url}" class="btn btn-confirm">✓ &nbsp;Confirm Delivery</a>
        <a href="{reject_url}" class="btn btn-reject">✗ &nbsp;Reject Delivery</a>
      </div>
      <p class="notice">
        Confirming will schedule your package for last-mile delivery.<br>
        Rejecting will halt routing and flag the package for review.
      </p>
    </div>
    <div class="footer">
      You received this because you are the registered customer for package #{package_id}.<br>
      Chonchoun Courier &middot; Links expire 48 hours after this email was sent.
    </div>
  </div>
</body>
</html>
"""


def send_warehouse_arrival_email(
    to_email: str,
    package_id: int,
    confirm_url: str,
    reject_url: str,
) -> None:
    """Send the warehouse-arrival confirmation email with Confirm / Reject buttons."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Action required: your package #{package_id} is at the warehouse"
    msg["From"] = SMTP_USER
    msg["To"] = to_email

    msg.attach(MIMEText(
        _HTML.format(package_id=package_id, confirm_url=confirm_url, reject_url=reject_url),
        "html",
    ))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_USER, to_email, msg.as_string())
