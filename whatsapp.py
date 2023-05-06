import os
from twilio.rest import TwilioRestClient
from custom.credentials import token, account

def whatsapp_message(token, account, to_number, message):
	client = Client(account, token)
	from_number = 'whatsapp:+14155238886'
	to_number = 'whatsapp:'+to_number
	client.messages.create(body=message, to= to_number, from_ = from_number)

	   message = client.messages.create(body="Hey this is a Test SMS",
        to="+919551771607",    # Replace with your phone number
        from_="+14066234282") # Replace with your Twilio number
