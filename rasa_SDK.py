from rasa.core.agent import Agent
from rasa.utils.endpoints import EndpointConfig
import asyncio

url = 'http://localhost:5056/webhook'

print('action endpoint: "{}"'.format(url))
model = 'C:/Users/gsl/Desktop/rasa_alt/20220310-113317.tar.gz'

agent = Agent.load(
    model,
    action_endpoint=EndpointConfig(url)
)

async def pharse(user_input):
    nlu = await agent.parse_message_using_nlu_interpreter(user_input)
    return nlu

nlu = asyncio.run(pharse('12365478'))
print(nlu)
