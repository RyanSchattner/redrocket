from flask import Flask
from redclient.models import RedRocketDaily
from datetime import datetime, timedelta, timezone
import time
import alpaca_trade_api as tradeapi

app = Flask(__name__)

#app.testing=True
#############################################################################
############ CONFIGURATIONS (CAN BE SEPARATE CONFIG.PY FILE) ###############
###########################################################################

# Remember you need to set your environment variables at the command line
# when you deploy this to a real website.
# export SECRET_KEY=mysecret
# set SECRET_KEY=mysecret
app.config['SECRET_KEY'] = 'mysecret'
#app.config['RECAPTCHA_PUBLIC_KEY'] = '6LdXpVwaAAAAAK3jEmi-WzkD4rP628WHx3An7Zr5'
#app.config['RECAPTCHA_PRIVATE_KEY'] = '6LdXpVwaAAAAAGA4SsGc8tOOAwmr_POXZjJQLesx'

#################################
### run models ############
###############################
base_url = 'https://paper-api.alpaca.markets'
api_key_id = 'PKHGQBATAM09YY8RZ0M8'
api_secret = 'SDJ2Fx0MjOzJ8dNFpQ6c3clRnRkepUDGRAaxQXI6'

api = tradeapi.REST(
    base_url=base_url,
    key_id=api_key_id,
    secret_key=api_secret
)
alpaca = tradeapi.REST(api_key_id, api_secret, base_url, 'v2')
print('starting up!')
isOpen = alpaca.get_clock().is_open

#rrd=RedRocketDaily()
#rrd.run()
while not isOpen:
	clock = alpaca.get_clock()
	openingTime = clock.next_open
	currTime = clock.timestamp
	timeToOpen = openingTime-currTime
	print(str(timeToOpen) + " minutes til market open.")
	time.sleep(60)
	isOpen = alpaca.get_clock().is_open
while isOpen==True:
	#pass
	print('market open!')
	rrd=RedRocketDaily()
	rrd.run()
