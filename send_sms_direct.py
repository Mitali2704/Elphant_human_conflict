from twilio.rest import Client

# Twilio credentials
account_sid = 'AC63f94d3aaba7d73b40e13bee610264d0'
auth_token = '6d4f2f2781c5f6f0dd66d243be040307'
client = Client(account_sid, auth_token)

def send_sms_alert():
    message = client.messages.create(
        body="Alert! Elephant detected near the area.",
        from_='+18566444281',  # Your Twilio trial number
        to='+18566444281'       # Your verified personal number
    )
    print(f"SMS sent! SID: {message.sid}")

send_sms_alert()
